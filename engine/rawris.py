"""농촌용수종합정보시스템(RAWRIS) 실시간 저수율 — https://rawris-am.ekr.or.kr/wrms/

공개 지도 화면이 쓰는 저수지 목록 자료(로그인 없음)를 하루 몇 번만 받아 캐시합니다.
  - 전국 약 2,600개 계측 저수지의 10분 단위 최신 저수율(cr_ratio), 평년 저수율(nr_ratio), 평년 대비(per_rate)
  - 저수지 위치(위경도) — 주소·이름 검색보다 정확 → 지도 위치 기준으로 사용
농업가뭄 단계는 ADMS 농업가뭄관리시스템(https://adms.ekr.or.kr) 저수율지도 기준(평년 대비 %)을 따릅니다.
서버 부담을 줄이려고 10분 이내에는 다시 받지 않고, 실패하면 마지막 저장본을 씁니다.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .operation import official_stage

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "rawris_snapshot.json"
URL = "https://rawris-am.ekr.or.kr/wrms/query/json?qid=rawris.wrms.gis.metagis_list_meas_reservoir"
UA = "Mozilla/5.0 (sungju-water student project; KNU)"
MIN_INTERVAL_S = 600


def fetch(timeout: int = 30) -> list[dict]:
    req = urllib.request.Request(URL, data=b"", method="POST",
                                 headers={"User-Agent": UA, "Referer": "https://rawris-am.ekr.or.kr/wrms/",
                                          "X-Requested-With": "XMLHttpRequest"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def to_frame(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["표준코드", "시설명", "본부", "지사", "lat", "lon", "시각", "저수율", "평년", "평년대비",
                                     "수위", "단계"])
    df = pd.DataFrame(items)
    out = pd.DataFrame({
        "표준코드": df["fac_code"].astype(str),
        "시설명": df.get("equip_name"),
        "본부": df.get("bonbu_name"),
        "지사": df.get("jisa_name"),
        "lat": pd.to_numeric(df.get("lat"), errors="coerce"),
        "lon": pd.to_numeric(df.get("lon"), errors="coerce"),
        "시각": pd.to_datetime(df.get("cr_date"), unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None),
        "저수율": pd.to_numeric(df.get("cr_ratio"), errors="coerce"),
        "평년": pd.to_numeric(df.get("nr_ratio"), errors="coerce"),
        "평년대비": pd.to_numeric(df.get("per_rate"), errors="coerce"),
        "수위": pd.to_numeric(df.get("cr_level"), errors="coerce"),
        "통신지연(일)": pd.to_numeric(df.get("comm_delay_2"), errors="coerce"),
    })
    # 통신이 하루 이상 끊긴 계측기는 실시간 값으로 쓰지 않음
    stale = out["통신지연(일)"].fillna(0) > 1
    out.loc[stale, ["저수율", "평년대비"]] = None
    out["단계"] = [official_stage(v) for v in out["평년대비"]]
    # 같은 저수지에 계측기가 여럿이면 최신 값 하나
    return out.sort_values("시각").drop_duplicates("표준코드", keep="last").reset_index(drop=True)


def snapshot(force: bool = False) -> tuple[pd.DataFrame, str]:
    """(자료, 출처 설명). 10분 이내 저장본이 있으면 그대로, 받기 실패 시 마지막 저장본"""
    if CACHE.exists() and not force and time.time() - CACHE.stat().st_mtime < MIN_INTERVAL_S:
        return to_frame(json.loads(CACHE.read_text(encoding="utf-8"))), "RAWRIS (10분 이내 저장본)"
    try:
        items = fetch()
        CACHE.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        return to_frame(items), "RAWRIS 실시간"
    except Exception as e:  # noqa: BLE001
        if CACHE.exists():
            ts = pd.Timestamp(CACHE.stat().st_mtime, unit="s", tz="UTC").tz_convert("Asia/Seoul")
            return to_frame(json.loads(CACHE.read_text(encoding="utf-8"))), f"RAWRIS 저장본({ts:%m-%d %H:%M}, 받기 실패: {e})"
        raise


def lookup(code: str, df: pd.DataFrame | None = None) -> dict | None:
    if df is None:
        try:
            df, _ = snapshot()
        except Exception:  # noqa: BLE001
            return None
    hit = df[df["표준코드"] == str(code)]
    return None if hit.empty else hit.iloc[0].to_dict()


def update_coords(df: pd.DataFrame) -> int:
    """RAWRIS 위치로 좌표 캐시를 채움 (직접 입력(manual)·DEM 수면 중심 좌표는 유지). 바뀐 개수"""
    from . import geocode

    cur = geocode.load_coords().set_index("표준코드")
    n = 0
    for _, r in df.dropna(subset=["lat", "lon"]).iterrows():
        if not (33 < r["lat"] < 39 and 124 < r["lon"] < 132):
            continue
        if r["표준코드"] in cur.index and str(cur.loc[r["표준코드"], "source"]).startswith(("manual", "DEM")):
            continue  # 직접 입력·DEM(만수위 수면 중심) 좌표는 유지
        if r["표준코드"] in cur.index and str(cur.loc[r["표준코드"], "source"]) == "RAWRIS" \
                and abs(cur.loc[r["표준코드"], "lat"] - r["lat"]) < 1e-5 and abs(cur.loc[r["표준코드"], "lon"] - r["lon"]) < 1e-5:
            continue
        cur.loc[r["표준코드"], ["시설명", "lat", "lon", "source", "query"]] = [r["시설명"], r["lat"], r["lon"], "RAWRIS", "rawris"]
        n += 1
    if n:
        geocode.write_coords(cur.reset_index())
    return n
