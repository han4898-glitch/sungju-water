"""기상청 자료 연계 (공공데이터포털)

1) 관측: 기상청_지상(종관, ASOS) 일자료 조회서비스
   - 저수지 주변 ASOS 지점 3곳을 거리 역제곱(IDW)으로 묶어 저수지 기상 시계열을 만듭니다.
   - data/weather_kma_auto_<표준코드>.csv 에 저장하고, 다음부터는 마지막 날짜 이후만 받습니다.
2) 예보: 기상청_단기예보 ((구)_동네예보) 조회서비스
   - 저수지 위치를 기상청 격자(nx, ny)로 바꿔 향후 약 3일 강수량·기온을 일 단위로 묶습니다.

인증키 (둘 중 하나):
  - 기상청 API허브(apihub.kma.go.kr) authKey   → KMA_APIHUB_KEY   (우선 사용)
  - 공공데이터포털 '일반 인증키(Decoding)'       → DATA_GO_KR_KEY
  환경변수 또는 .streamlit/secrets.toml. 각 사이트에서 두 서비스를 모두 '활용신청'해야 합니다.
"""
from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .geocode import UA, _env

ROOT = Path(__file__).resolve().parents[1]
# 같은 서비스를 두 곳에서 제공: 공공데이터포털(serviceKey) / 기상청 API허브(authKey)
DATAGO_BASE = "https://apis.data.go.kr/1360000/"
APIHUB_BASE = "https://apihub.kma.go.kr/api/typ02/openApi/"
ASOS_URL = "AsosDalyInfoService/getWthrDataList"          # 공공데이터포털용
APIHUB_SFCDD3 = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"  # API허브: 지상관측 > ASOS > 1.4 일자료(기간 조회)
# kma_sfcdd3 출력 열 순서 (API허브 명세)
SFCDD3_COLS = ["TM", "STN", "WS_AVG", "WR_DAY", "WD_MAX", "WS_MAX", "WS_MAX_TM", "WD_INS", "WS_INS", "WS_INS_TM",
               "TA_AVG", "TA_MAX", "TA_MAX_TM", "TA_MIN", "TA_MIN_TM", "TD_AVG", "TS_AVG", "TG_MIN", "HM_AVG",
               "HM_MIN", "HM_MIN_TM", "PV_AVG", "EV_S", "EV_L", "FG_DUR", "PA_AVG", "PS_AVG", "PS_MAX",
               "PS_MAX_TM", "PS_MIN", "PS_MIN_TM", "CA_TOT", "SS_DAY", "SS_DUR", "SS_CMB", "SI_DAY",
               "SI_60M_MAX", "SI_60M_MAX_TM", "RN_DAY", "RN_D99", "RN_DUR", "RN_60M_MAX", "RN_60M_MAX_TM",
               "RN_10M_MAX", "RN_10M_MAX_TM", "RN_POW_MAX", "RN_POW_MAX_TM", "SD_NEW", "SD_NEW_TM", "SD_MAX",
               "SD_MAX_TM", "TE_05", "TE_10", "TE_15", "TE_30", "TE_50"]
FCST_URL = "VilageFcstInfoService_2.0/getVilageFcst"
FIELD_MAP = {"tm": "date", "sumRn": "rain", "maxTa": "tmax", "minTa": "tmin",
             "avgRhm": "rh", "avgWs": "wind", "sumSsHr": "sunshine"}

# 영남·충북 남부 ASOS 지점 (지점번호: 이름, 위도, 경도) — 위치는 기상청 지점정보 기준 근사값
ASOS_STATIONS = {
    130: ("울진", 36.9918, 129.4128), 136: ("안동", 36.5729, 128.7073), 137: ("상주", 36.4084, 128.1574),
    138: ("포항", 36.0326, 129.3796), 143: ("대구", 35.8780, 128.6530), 221: ("제천", 37.1593, 128.1943),
    226: ("보은", 36.4876, 127.7341), 271: ("봉화", 36.9436, 128.9145), 272: ("영주", 36.8718, 128.5167),
    273: ("문경", 36.6273, 128.1488), 276: ("청송군", 36.4351, 129.0401), 277: ("영덕", 36.5333, 129.4093),
    278: ("의성", 36.3561, 128.6886), 279: ("구미", 36.1306, 128.3206), 281: ("영천", 35.9774, 128.9514),
    283: ("경주시", 35.8172, 129.2009), 284: ("거창", 35.6712, 127.9109), 285: ("합천", 35.5650, 128.1699),
    288: ("밀양", 35.4914, 128.7441), 289: ("산청", 35.4130, 127.8790), 152: ("울산", 35.5601, 129.3200),
    192: ("진주", 35.1638, 128.0400), 248: ("장수", 35.6570, 127.5203), 135: ("추풍령", 36.2203, 127.9946),
}


def get_key() -> tuple[str, str] | None:
    """(키, 출처) — 출처는 apihub 또는 datago"""
    k = _env("KMA_APIHUB_KEY")
    if k:
        return k, "apihub"
    k = _env("DATA_GO_KR_KEY")
    if k:
        if "%" in k:  # '인코딩' 키를 넣은 경우 → 디코딩 키로 변환
            k = urllib.parse.unquote(k)
        return k, "datago"
    return None


def _get(service: str, params: dict, key: tuple[str, str], timeout: int = 30) -> dict:
    k, src = key
    base = APIHUB_BASE if src == "apihub" else DATAGO_BASE
    params = {("authKey" if src == "apihub" else "serviceKey"): k, **params}
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{base}{service}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:  # 키 오류 등은 XML로 옴
        msg = raw[:300].replace("\n", " ")
        raise RuntimeError(f"기상청 API 오류 응답: {msg}")
    head = data.get("response", {}).get("header", {})
    if head.get("resultCode") not in ("00", None):
        if head.get("resultCode") == "03":  # NO_DATA
            return {"items": []}
        raise RuntimeError(f"기상청 API 오류 {head.get('resultCode')}: {head.get('resultMsg')}")
    items = data.get("response", {}).get("body", {}).get("items") or {}
    items = items.get("item", []) if isinstance(items, dict) else items
    return {"items": items if isinstance(items, list) else [items]}


# ------------------------------------------------------------- 지점 선택 --
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))


def nearest_stations(lat: float, lon: float, n: int = 3) -> list[dict]:
    d = [{"stn": k, "name": v[0], "km": haversine_km(lat, lon, v[1], v[2])} for k, v in ASOS_STATIONS.items()]
    d.sort(key=lambda r: r["km"])
    sel = d[:n]
    w = np.array([1 / max(s["km"], 1.0) ** 2 for s in sel])
    for s, wi in zip(sel, w / w.sum()):
        s["weight"] = float(wi)
    return sel


# ------------------------------------------------------------- ASOS 관측 --
def fetch_asos(key: tuple[str, str], stn: int, start: str, end: str) -> pd.DataFrame:
    """한 지점 일자료. API허브 키면 kma_sfcdd3(텍스트), 공공데이터포털 키면 JSON 서비스"""
    if key[1] == "apihub":
        return fetch_asos_apihub(key[0], stn, start, end)
    rows = []
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for y in range(s.year, e.year + 1):
        ys = max(s, pd.Timestamp(f"{y}-01-01")).strftime("%Y%m%d")
        ye = min(e, pd.Timestamp(f"{y}-12-31")).strftime("%Y%m%d")
        if ys > ye:
            continue
        r = _get(ASOS_URL, {"pageNo": 1, "numOfRows": 400, "dataType": "JSON",
                            "dataCd": "ASOS", "dateCd": "DAY", "startDt": ys, "endDt": ye, "stnIds": stn}, key)
        rows += r["items"]
        time.sleep(0.15)
    return parse_asos(rows)


def parse_sfcdd3(text: str) -> pd.DataFrame:
    """API허브 kma_sfcdd3 텍스트 → 표준 열 (rain, tmax, tmin, rh, wind, sunshine)"""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace(",", " ").split()]
        if len(parts) < 39 or not parts[0][:8].isdigit():
            continue
        rows.append(parts[: len(SFCDD3_COLS)])
    if not rows:
        if "authKey" in text or "error" in text.lower() or "인증" in text:
            raise RuntimeError(f"기상청 API허브 오류 응답: {text[:200]}")
        return pd.DataFrame(columns=["rain", "tmax", "tmin", "rh", "wind", "sunshine"])
    df = pd.DataFrame(rows, columns=SFCDD3_COLS[: len(rows[0])])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TM"].str[:8], format="%Y%m%d"),
        "rain": pd.to_numeric(df["RN_DAY"], errors="coerce"),
        "tmax": pd.to_numeric(df["TA_MAX"], errors="coerce"),
        "tmin": pd.to_numeric(df["TA_MIN"], errors="coerce"),
        "rh": pd.to_numeric(df["HM_AVG"], errors="coerce"),
        "wind": pd.to_numeric(df["WS_AVG"], errors="coerce"),
        "sunshine": pd.to_numeric(df["SS_DAY"], errors="coerce"),
    }).set_index("date").sort_index()
    out = out.mask(out <= -9)              # 결측(-9, -99 등)
    out["rain"] = out["rain"].fillna(0.0)  # 무강수일
    # 열 순서가 명세와 다르면 값이 엉뚱해지므로 점검
    chk = out.dropna(subset=["tmax", "tmin", "rh"])
    if len(chk) and ((chk["tmax"] < chk["tmin"]).mean() > 0.2 or (chk["rh"] > 100).any()):
        raise RuntimeError("API허브 일자료 열 순서가 예상과 다릅니다. help=1 출력으로 열 순서를 확인해야 합니다.")
    return out


def fetch_asos_apihub(auth: str, stn: int, start: str, end: str, chunk_days: int = 180) -> pd.DataFrame:
    frames = []
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    while s <= e:
        ce = min(e, s + pd.Timedelta(days=chunk_days - 1))
        q = urllib.parse.urlencode({"tm1": s.strftime("%Y%m%d"), "tm2": ce.strftime("%Y%m%d"),
                                    "stn": stn, "help": 0, "disp": 0, "authKey": auth})
        req = urllib.request.Request(f"{APIHUB_SFCDD3}?{q}", headers={"User-Agent": UA})
        for attempt in range(5):  # 인터넷이 잠깐 끊겨도 이어서 받도록 재시도 (5·10·20·40초 대기)
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    raw = r.read()
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                if attempt == 4:
                    raise
                time.sleep(5 * 2 ** attempt)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("euc-kr", errors="replace")
        frames.append(parse_sfcdd3(text))
        s = ce + pd.Timedelta(days=1)
        time.sleep(0.15)
    frames = [f for f in frames if len(f)]
    return pd.concat(frames) if frames else parse_sfcdd3("")


def parse_asos(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=list(FIELD_MAP.values())[1:])
    df = pd.DataFrame(items)
    df = df[[c for c in FIELD_MAP if c in df.columns]].rename(columns=FIELD_MAP)
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    df["rain"] = df["rain"].fillna(0.0)  # 무강수일은 빈칸
    return df.set_index("date").sort_index()


def combine_stations(frames: dict[int, pd.DataFrame], weights: dict[int, float]) -> pd.DataFrame:
    """지점별 자료를 가중 평균 (결측인 지점은 그날 가중치에서 빠짐)"""
    cols = ["rain", "tmax", "tmin", "rh", "wind", "sunshine"]
    idx = sorted(set().union(*[f.index for f in frames.values() if len(f)]))
    out = pd.DataFrame(index=pd.DatetimeIndex(idx, name="date"), columns=cols, dtype=float)
    for c in cols:
        num = pd.Series(0.0, index=out.index)
        den = pd.Series(0.0, index=out.index)
        for stn, f in frames.items():
            if c not in f:
                continue
            v = f[c].reindex(out.index)
            m = v.notna()
            num[m] += v[m] * weights[stn]
            den[m] += weights[stn]
        out[c] = (num / den.where(den > 0)).round(2)
    return out.asfreq("D").interpolate(limit=3)


def update_weather(code: str, lat: float, lon: float, start: str = "2015-01-01",
                   key: tuple[str, str] | None = None, progress=None) -> tuple[Path, dict]:
    """저수지 기상 시계열을 최신(어제)까지 갱신하고 파일 경로·정보를 돌려줍니다."""
    key = key or get_key()
    if not key:
        raise RuntimeError("기상청 인증키(KMA_APIHUB_KEY 또는 DATA_GO_KR_KEY)가 없습니다.")
    path = ROOT / "data" / f"weather_kma_auto_{code}.csv"
    stations = nearest_stations(lat, lon, 3)
    old = pd.read_csv(path, parse_dates=["date"], index_col="date") if path.exists() else None
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    ranges = []
    if old is None or old.empty:
        ranges.append((start, yesterday))
    else:
        if pd.Timestamp(start) < old.index.min() - pd.Timedelta(days=1):  # 과거로 늘리기 (장기 자료)
            ranges.append((start, (old.index.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
        tail_from = (old.index.max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        if tail_from <= yesterday:
            ranges.append((tail_from, yesterday))
    if not ranges:
        return path, {"stations": stations, "new_days": 0, "first": str(old.index.min().date()),
                      "last": str(old.index.max().date())}
    weights = {s["stn"]: s["weight"] for s in stations}
    merged = old
    n_new = 0
    for k, (a, b) in enumerate(ranges):
        frames = {}
        for i, s in enumerate(stations):
            if progress:
                progress((k * len(stations) + i) / (len(ranges) * len(stations)), f"{s['name']}({s['stn']}) {a[:4]}~{b[:4]} 받는 중…")
            frames[s["stn"]] = fetch_asos(key, s["stn"], a, b)
        if not any(len(f) for f in frames.values()):
            continue
        new = combine_stations(frames, weights)
        n_new += len(new)
        if merged is None:
            merged = new
        else:  # 새 자료가 겹치는 날짜를 덮어씀
            merged = pd.concat([merged[(merged.index < new.index.min()) | (merged.index > new.index.max())], new]).sort_index()
    merged.index.name = "date"
    merged.to_csv(path)
    return path, {"stations": stations, "new_days": int(n_new), "first": str(merged.index.min().date()),
                  "last": str(merged.index.max().date())}


# ------------------------------------------------------------- 단기예보 --
def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    """위경도 → 기상청 동네예보 격자 (Lambert Conformal Conic, 기상청 공개 변환식)"""
    RE, GRID, SLAT1, SLAT2, OLON, OLAT, XO, YO = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0, 43, 136
    D = math.pi / 180.0
    re = RE / GRID
    s1, s2, olon, olat = SLAT1 * D, SLAT2 * D, OLON * D, OLAT * D
    sn = math.log(math.cos(s1) / math.cos(s2)) / math.log(math.tan(math.pi / 4 + s2 / 2) / math.tan(math.pi / 4 + s1 / 2))
    sf = math.tan(math.pi / 4 + s1 / 2) ** sn * math.cos(s1) / sn
    ro = re * sf / math.tan(math.pi / 4 + olat / 2) ** sn
    ra = re * sf / math.tan(math.pi / 4 + lat * D / 2) ** sn
    theta = lon * D - olon
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    theta *= sn
    return int(ra * math.sin(theta) + XO + 0.5), int(ro - ra * math.cos(theta) + YO + 0.5)


def parse_pcp(v: str) -> float:
    """단기예보 1시간 강수량 문자열 → mm"""
    v = str(v).strip()
    if v in ("", "강수없음", "-", "0"):
        return 0.0
    if "미만" in v:
        return 0.5
    if "이상" in v:
        return float(v.replace("mm", "").replace("이상", "").strip())
    if "~" in v:
        a, b = v.replace("mm", "").split("~")
        return (float(a) + float(b)) / 2
    try:
        return float(v.replace("mm", ""))
    except ValueError:
        return 0.0


def latest_base(now: datetime | None = None) -> tuple[str, str]:
    """발표 후 약 10분 뒤부터 조회 가능 → 가장 최근 발표시각"""
    now = (now or datetime.now()) - timedelta(minutes=15)
    for h in (23, 20, 17, 14, 11, 8, 5, 2):
        if now.hour >= h:
            return now.strftime("%Y%m%d"), f"{h:02d}00"
    prev = now - timedelta(days=1)
    return prev.strftime("%Y%m%d"), "2300"


def parse_forecast(items: list[dict]) -> pd.DataFrame:
    """단기예보 항목 → 일별 (rain 합, tmax, tmin, rh 평균, wind 평균)"""
    if not items:
        return pd.DataFrame(columns=["rain", "tmax", "tmin", "rh", "wind"])
    df = pd.DataFrame(items)
    df["date"] = pd.to_datetime(df["fcstDate"])
    out = pd.DataFrame(index=sorted(df["date"].unique()))
    pcp = df[df["category"] == "PCP"].assign(v=lambda d: d["fcstValue"].map(parse_pcp))
    out["rain"] = pcp.groupby("date")["v"].sum()
    tmp = df[df["category"] == "TMP"].assign(v=lambda d: pd.to_numeric(d["fcstValue"], errors="coerce"))
    out["tmax"] = df[df["category"] == "TMX"].assign(v=lambda d: pd.to_numeric(d["fcstValue"], errors="coerce")).groupby("date")["v"].max()
    out["tmin"] = df[df["category"] == "TMN"].assign(v=lambda d: pd.to_numeric(d["fcstValue"], errors="coerce")).groupby("date")["v"].min()
    out["tmax"] = out["tmax"].fillna(tmp.groupby("date")["v"].max())
    out["tmin"] = out["tmin"].fillna(tmp.groupby("date")["v"].min())
    for cat, col in (("REH", "rh"), ("WSD", "wind")):
        out[col] = df[df["category"] == cat].assign(v=lambda d: pd.to_numeric(d["fcstValue"], errors="coerce")).groupby("date")["v"].mean()
    out["rain"] = out["rain"].fillna(0.0)
    out.index.name = "date"
    return out


def fetch_forecast(lat: float, lon: float, key: tuple[str, str] | None = None) -> tuple[pd.DataFrame, dict]:
    key = key or get_key()
    if not key:
        raise RuntimeError("기상청 인증키(KMA_APIHUB_KEY 또는 DATA_GO_KR_KEY)가 없습니다.")
    nx, ny = latlon_to_grid(lat, lon)
    bd, bt = latest_base()
    r = _get(FCST_URL, {"pageNo": 1, "numOfRows": 1000, "dataType": "JSON",
                        "base_date": bd, "base_time": bt, "nx": nx, "ny": ny}, key)
    return parse_forecast(r["items"]), {"base": f"{bd} {bt}", "grid": (nx, ny), "source": key[1]}


# ------------------------------------------------------------- 누적 강수 --
def cumulative_rain(weather: pd.DataFrame, end: pd.Timestamp | None = None,
                    windows=(7, 30, 90, 180, 365)) -> pd.DataFrame:
    """최근 N일 누적강수와 평년(같은 기간 과거 평균) 대비 비율 — 가뭄 판단 지표"""
    r = weather["rain"].dropna()
    end = pd.Timestamp(end or r.index.max()).normalize()
    rows = []
    for n in windows:
        cur = r.loc[end - pd.Timedelta(days=n - 1): end].sum()
        past = []
        for y in sorted(set(r.index.year)):
            e = end - pd.DateOffset(years=end.year - y)
            if y == end.year or e - pd.Timedelta(days=n - 1) < r.index.min():
                continue
            past.append(r.loc[e - pd.Timedelta(days=n - 1): e].sum())
        normal = float(np.mean(past)) if past else np.nan
        rows.append({"기간": f"최근 {n}일", "누적강수(mm)": float(cur), "평년(mm)": normal,
                     "평년 대비(%)": float(cur / normal * 100) if normal and normal > 0 else np.nan,
                     "비교 연수": len(past)})
    return pd.DataFrame(rows).set_index("기간")
