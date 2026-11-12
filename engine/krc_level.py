"""한국농어촌공사_농촌용수 저수지 수위정보 조회 (공공데이터포털 15099919) — 일별 수위·저수율

엔드포인트: https://apis.data.go.kr/B552149/reserviorWaterLevel/reservoirlevel/
  요청: serviceKey, pageNo, numOfRows, fac_code(표준코드 10자리), date_s, date_e (YYYYMMDD, 최대 365일)
  응답(XML): fac_code, fac_name, county, check_date, water_level(m), rate(저수율 %)
인증키: 공공데이터포털 일반 인증키(Decoding) → DATA_GO_KR_KEY (환경변수 또는 .streamlit/secrets.toml)

받은 자료는 data/obs_rate_<표준코드>.csv 에 쌓고, 다음부터는 마지막 날짜 이후만 받습니다.
또 모델 검증용 data/observed_storage_<표준코드>.csv (저수량 천㎥)도 함께 만듭니다.
※ 저수율은 유효저수량 기준으로 보고 저수량 = 사수량 + 저수율 × 유효저수량 으로 환산합니다. [확인 필요]
"""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .geocode import UA, _env

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://apis.data.go.kr/B552149/reserviorWaterLevel/"


def get_key() -> str | None:
    k = _env("DATA_GO_KR_KEY")
    if k and "%" in k:
        k = urllib.parse.unquote(k)
    return k


def obs_path(code: str) -> Path:
    return ROOT / "data" / f"obs_rate_{code}.csv"


def parse_xml(text: str) -> tuple[list[dict], int]:
    """응답 XML → (레코드 목록, 전체 건수). 오류 응답이면 예외"""
    try:
        root = ET.fromstring(text.strip())
    except ET.ParseError:
        raise RuntimeError(f"저수지 수위 API 응답을 읽을 수 없습니다: {text[:200]}")
    code = (root.findtext(".//returnReasonCode") or root.findtext(".//resultCode") or "").strip()
    msg = (root.findtext(".//returnAuthMsg") or root.findtext(".//resultMsg") or "").strip()
    if code and code not in ("00", "0", "000"):
        if code == "03" or "NODATA" in re.sub(r"[^A-Z]", "", msg.upper()):
            return [], 0
        raise RuntimeError(f"저수지 수위 API 오류 {code}: {msg}")
    recs = []
    for it in root.iter("item"):
        recs.append({c.tag: (c.text or "").strip() for c in it})
    total = int((root.findtext(".//totalCount") or len(recs)) or 0)
    return recs, total


def fetch_rate(key: str, fac_code: str, start: str, end: str, rows: int = 400) -> pd.DataFrame:
    """일별 수위·저수율 (365일씩 나눠 요청)"""
    recs = []
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    while s <= e:
        ce = min(e, s + pd.Timedelta(days=364))
        page = 1
        while True:
            q = urllib.parse.urlencode({"serviceKey": key, "pageNo": page, "numOfRows": rows,
                                        "fac_code": fac_code, "date_s": s.strftime("%Y%m%d"),
                                        "date_e": ce.strftime("%Y%m%d")})
            req = urllib.request.Request(f"{BASE}reservoirlevel/?{q}", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("euc-kr", errors="replace")
            items, total = parse_xml(text)
            recs += items
            if not items or page * rows >= total:
                break
            page += 1
        s = ce + pd.Timedelta(days=1)
        time.sleep(0.15)
    return to_frame(recs)


def to_frame(recs: list[dict]) -> pd.DataFrame:
    if not recs:
        return pd.DataFrame(columns=["level", "rate"], index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(recs)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["check_date"].str.replace(r"[^\d]", "", regex=True).str[:8], format="%Y%m%d"),
        "level": pd.to_numeric(df.get("water_level"), errors="coerce"),
        "rate": pd.to_numeric(df.get("rate"), errors="coerce"),
    })
    return out.groupby("date").last().sort_index()


def update(code: str, cfg: dict, start: str = "2015-01-01", key: str | None = None) -> tuple[pd.DataFrame, dict]:
    """관측 일별 저수율을 어제까지 갱신 → (전체 자료, 정보). 모델 검증용 저수량 파일도 갱신"""
    key = key or get_key()
    if not key:
        raise RuntimeError("공공데이터포털 인증키(DATA_GO_KR_KEY)가 없습니다.")
    path = obs_path(code)
    old = pd.read_csv(path, parse_dates=["date"], index_col="date") if path.exists() else None
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    frm = start if old is None or old.empty else (old.index.max() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    new = fetch_rate(key, code, frm, yesterday) if frm <= yesterday else old.iloc[0:0]
    if (old is None or old.empty) and new.empty:
        raise LookupError("이 저수지는 한국농어촌공사 수위 관측 자료가 없습니다 (자동 수위계가 설치된 저수지만 제공). "
                          "모델은 기상자료만으로 계속 실행할 수 있습니다.")
    full = new if old is None else pd.concat([old[old.index < (new.index.min() if len(new) else pd.Timestamp.max)], new])
    full = full[~full.index.duplicated(keep="last")].sort_index()
    full.index.name = "date"
    full.to_csv(path)
    write_observed_storage(code, full, cfg)
    return full, {"new_days": int(len(new)), "first": str(full.index.min().date()) if len(full) else None,
                  "last": str(full.index.max().date()) if len(full) else None}


def write_observed_storage(code: str, obs: pd.DataFrame, cfg: dict) -> Path:
    r = cfg["reservoir"]
    eff = r["total_storage"] - r["dead_storage"]
    st = (r["dead_storage"] + obs["rate"].clip(lower=0) / 100 * eff).clip(upper=r["total_storage"] * 1.05)
    out = pd.DataFrame({"storage": st.round(1)}).dropna()
    out.index.name = "date"
    p = ROOT / "data" / f"observed_storage_{code}.csv"
    out.to_csv(p)
    return p


def normal_curve(obs: pd.DataFrame, year: int) -> pd.DataFrame:
    """평년(해당 연도 제외한 같은 날짜 평균)·최소·최대 저수율 곡선 (1~366일)"""
    o = obs.dropna(subset=["rate"])
    o = o[o.index.year != year]
    if o.empty:
        return pd.DataFrame(columns=["normal", "min", "max", "years"])
    g = o.groupby(o.index.dayofyear)["rate"]
    out = pd.DataFrame({"normal": g.mean(), "min": g.min(), "max": g.max(), "years": g.count()})
    out = out.reindex(range(1, 367)).interpolate(limit_direction="both")
    return out


def status(obs: pd.DataFrame) -> dict:
    """최신 저수율, 평년 대비, 7일·30일 변화"""
    o = obs.dropna(subset=["rate"])
    if o.empty:
        return {}
    last_d = o.index.max()
    cur = float(o["rate"].iloc[-1])
    nc = normal_curve(o, last_d.year)
    normal = float(nc.loc[min(last_d.dayofyear, 366), "normal"]) if len(nc) else np.nan

    def chg(n):
        prev = o["rate"].loc[: last_d - pd.Timedelta(days=n)]
        return cur - float(prev.iloc[-1]) if len(prev) else np.nan

    return {"date": last_d, "rate": cur, "level": float(o["level"].iloc[-1]) if "level" in o else np.nan,
            "normal": normal, "pct_of_normal": cur / normal * 100 if normal and normal > 0 else np.nan,
            "chg7": chg(7), "chg30": chg(30)}
