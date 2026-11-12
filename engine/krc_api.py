"""저수지 수위 API 클라이언트 (행정안전부 재난안전데이터 공유플랫폼, 원천: 한국농어촌공사)

호출 형식:  {base_url}{dataset_id}?serviceKey=...&returnType=json&pageNo=1&numOfRows=1000
응답의 body 레코드 필드명은 데이터셋마다 다를 수 있어, 아래 FIELD_HINTS로 자동 인식하고
config [api.fields] 에 직접 지정하면 그 값을 우선합니다.  필드 확인:  python data/fetch_reservoir.py inspect

일일 호출 한도가 작을 수 있으므로(예: 100회/일) 받은 자료는 data/cache_*.csv 로 저장해 재사용합니다.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# 표준 열 → 원본 필드명에 들어갈 법한 단어 (대소문자 무시, 앞쪽일수록 우선)
FIELD_HINTS = {
    "time": ["계측일시", "측정일시", "일시", "MSRM_DT", "MSRMT_DT", "OBSR_DT", "CHECK_DT", "DT", "TM", "DATE"],
    "code": ["표준코드", "시설코드", "저수지코드", "FCLT_CD", "FAC_CD", "RSVR_CD", "CD", "CODE"],
    "name": ["시설명", "저수지명", "FCLT_NM", "FAC_NM", "RSVR_NM", "NM", "NAME"],
    "level": ["저수위", "RSVR_WTLV", "WTLV", "WL", "LEVEL"],
    "spill_level": ["방수로수위", "SPWY", "OTLT_WTLV"],
    "storage": ["저수량", "STRG_QTY", "STRG", "STORAGE", "VOL"],
    "rate": ["저수율", "STRG_RT", "RATE", "RT"],
    "inflow": ["유입량", "INFLW", "INFL", "INFLOW"],
    "supply": ["공급량", "SPLY", "SUPPLY", "OTFL"],
}


def get_key() -> str | None:
    key = os.environ.get("SAFETYDATA_KEY")
    if key:
        return key
    try:  # Streamlit 배포 시 secrets
        import streamlit as st  # type: ignore

        return st.secrets.get("SAFETYDATA_KEY")  # type: ignore[attr-defined]
    except Exception:
        return None


def fetch_pages(api_cfg: dict, dataset_id: str, extra: dict | None = None,
                max_pages: int = 20, num_rows: int = 1000) -> list[dict]:
    key = get_key()
    if not key:
        raise RuntimeError("인증키가 없습니다. 환경변수 SAFETYDATA_KEY 를 설정하세요.")
    if not dataset_id:
        raise RuntimeError("config [api] 의 데이터셋 ID(dataset_10min/dataset_daily)를 입력하세요.")
    records: list[dict] = []
    for page in range(1, max_pages + 1):
        q = {"serviceKey": key, "returnType": "json", "pageNo": page, "numOfRows": num_rows, **(extra or {})}
        url = api_cfg["base_url"] + dataset_id + "?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        body = extract_body(data)
        if not body:
            break
        records += body
        if len(body) < num_rows:
            break
        time.sleep(0.2)
    return records


def extract_body(data) -> list[dict]:
    """응답 JSON에서 레코드 목록을 찾아냅니다 (body 리스트, body.items.item 등 여러 형태 대응)."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for path in (("body",), ("body", "items"), ("body", "items", "item"), ("response", "body", "items", "item"), ("items",)):
        cur = data
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if isinstance(cur, list):
            return cur
        if isinstance(cur, dict) and path[-1] == "item":
            return [cur]
    return []


def detect_fields(record: dict, override: dict | None = None) -> dict:
    """원본 필드명 → 표준 열 매핑을 추정합니다."""
    override = override or {}
    keys = list(record)
    mapping, used = {}, set()
    for std, hints in FIELD_HINTS.items():
        if std in override:
            mapping[std] = override[std]
            used.add(override[std])
            continue
        for h in hints:
            hit = next((k for k in keys if k not in used and (k == h or h.upper() in k.upper())), None)
            if hit:
                mapping[std] = hit
                used.add(hit)
                break
    return mapping


def normalize(records: list[dict], override: dict | None = None) -> pd.DataFrame:
    """API 레코드를 표준 열(time, code, name, level, storage, rate, inflow, supply …)로 변환"""
    if not records:
        return pd.DataFrame(columns=list(FIELD_HINTS))
    m = detect_fields(records[0], override)
    df = pd.DataFrame(records).rename(columns={v: k for k, v in m.items()})
    df = df[[c for c in FIELD_HINTS if c in df.columns]].copy()
    if "time" in df:
        t = df["time"].astype(str).str.replace(r"[^\d]", "", regex=True)
        fmt = {8: "%Y%m%d", 10: "%Y%m%d%H", 12: "%Y%m%d%H%M", 14: "%Y%m%d%H%M%S"}
        df["time"] = pd.to_datetime(t, format=fmt.get(int(t.str.len().mode()[0]), None), errors="coerce")
    for c in ("level", "spill_level", "storage", "rate", "inflow", "supply"):
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    if "code" in df:
        df["code"] = df["code"].astype(str).str.strip()
    return df.sort_values("time") if "time" in df else df


def filter_reservoir(df: pd.DataFrame, code: str | None = None, name: str | None = None) -> pd.DataFrame:
    if code and "code" in df and (df["code"] == str(code)).any():
        return df[df["code"] == str(code)]
    if name and "name" in df:
        n = re.sub(r"(저수지|댐)$", "", name)
        return df[df["name"].astype(str).str.replace(r"(저수지|댐)$", "", regex=True) == n]
    return df.iloc[0:0]


def to_observed(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """API 자료 → 모델 검증용 일자료 (storage 천㎥, release 천㎥/일).

    저수량이 없으면 저수율 × 유효저수량 + 사수량으로 환산합니다.
    공급량 단위는 데이터셋 문서로 확인하세요: ㎥/s 면 ×86.4, ㎥/일이면 ÷1000 (config [api] supply_unit)."""
    res = cfg["reservoir"]
    d = df.set_index("time")
    out = pd.DataFrame(index=d.resample("D").mean(numeric_only=True).index)
    if "storage" in d and d["storage"].notna().any():
        st_ = d["storage"].resample("D").last()
        if st_.median() > res["total_storage"] * 5:  # ㎥ 단위로 판단 → 천㎥
            st_ = st_ / 1000
        out["storage"] = st_
    elif "rate" in d:
        out["storage"] = res["dead_storage"] + d["rate"].resample("D").last() / 100 * res["effective_storage"]
    if "supply" in d and d["supply"].notna().any():
        unit = cfg.get("api", {}).get("supply_unit", "cms")
        daily_mean = d["supply"].resample("D").mean()
        out["release"] = daily_mean * 86.4 if unit == "cms" else daily_mean / 1000
    out.index.name = "date"
    return out.dropna(how="all")


def cache_path(code: str, kind: str) -> Path:
    return ROOT / "data" / f"cache_{kind}_{code}.csv"
