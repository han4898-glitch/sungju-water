"""한국농어촌공사 저수지 제원 카탈로그 (data/krc_reservoirs.csv, 2026-09-07 기준 3,402개소).

- 자연어에 가까운 검색:  "경북 성주", "문경 경천", "성주댐", "경산 저수지 큰 거"
- 저수지별 모델 설정 자동 생성: 제원표의 유역면적·수혜면적·저수량을 모델 입력으로 변환

단위(제원표): 유역·홍수·만수·수혜면적 = ha, 저수량 = 천㎥, 수위 = EL.m
"""
from __future__ import annotations

import copy
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

CATALOG_CSV = Path(__file__).resolve().parents[1] / "data" / "krc_reservoirs.csv"

# 지역 별칭 → 본부명 / 소재지 키워드
REGION_ALIASES = {
    "경북": ["경북", "경상북도"], "경남": ["경남", "경상남도"], "전북": ["전북", "전라북도", "전북특별자치도"],
    "전남": ["전남", "전라남도"], "충북": ["충북", "충청북도"], "충남": ["충남", "충청남도"],
    "경기": ["경기", "경기도"], "강원": ["강원", "강원도", "강원특별자치도"], "제주": ["제주"],
}
STOPWORDS = {"저수지", "댐", "지", "호", "알려줘", "보여줘", "찾아줘", "제원", "현황", "저수율", "수위", "의", "좀", "어디"}


@lru_cache(maxsize=1)
def load_catalog(path: str = str(CATALOG_CSV)) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"표준코드": str, "연락처": str})
    df["시설명"] = df["시설명"].astype(str).str.strip()
    df["_addr"] = df["소재지"].fillna("")
    return df


def _tokens(query: str) -> list[str]:
    q = re.sub(r"[^\w\s]", " ", query)
    toks = []
    for t in q.split():
        t2 = re.sub(r"(저수지|댐|호)$", "", t)  # "성주댐" → "성주"
        if t2 and t2 not in STOPWORDS:
            toks.append(t2)
    return toks


def search(query: str, limit: int = 10) -> pd.DataFrame:
    """이름·지사·소재지·본부를 함께 보고 점수를 매겨 저수지를 찾습니다.
    점수: 시설명 일치 10, 시설명 포함 6, 지사·시군 일치 3, 본부(도) 일치 2. 동점이면 총저수량 큰 순."""
    df = load_catalog()
    toks = _tokens(query)
    if not toks:
        return df.head(0)
    score = pd.Series(0.0, index=df.index)
    name_hit = pd.Series(False, index=df.index)
    for t in toks:
        exact = df["시설명"] == t
        contains = df["시설명"].str.contains(t, regex=False)
        name_hit |= contains
        score += exact * 10 + (contains & ~exact) * 6
        score += df["지사"].astype(str).str.contains(t, regex=False) * 3
        score += df["_addr"].str.contains(t, regex=False) * 3
        for region, aliases in REGION_ALIASES.items():
            if t in aliases:
                score += (df["본부"] == region) * 2
    # 이름에 걸린 게 하나라도 있으면 이름 일치 후보 위주로
    cand = df[(score > 0) & (name_hit | ~name_hit.any())].assign(_score=score)
    return cand.sort_values(["_score", "총저수량"], ascending=[False, False]).head(limit).drop(columns=["_addr"])


def get(code: str) -> pd.Series:
    df = load_catalog()
    row = df[df["표준코드"] == str(code)]
    if row.empty:
        raise KeyError(f"표준코드 {code} 없음")
    return row.iloc[0]


def summary(row: pd.Series) -> dict:
    """사람이 읽기 좋은 제원 요약"""
    f = lambda c: None if pd.isna(row.get(c)) else (float(row[c]) if isinstance(row[c], (int, float)) else row[c])
    return {
        "표준코드": row["표준코드"], "시설명": row["시설명"], "본부/지사": f"{row['본부']} / {row['지사']}",
        "소재지": row["소재지"], "준공": str(row["준공년도"])[:4],
        "유역면적(km²)": round(f("유역면적") / 100, 2) if f("유역면적") else None,
        "수혜면적(ha)": f("수혜면적"), "만수면적(ha)": f("만수면적"),
        "총저수량(천㎥)": f("총저수량"), "유효저수량(천㎥)": f("유효저수량"), "사수량(천㎥)": f("사수량"),
        "만수위(EL.m)": f("만수위"), "사수위(EL.m)": f("사수위"), "홍수위(EL.m)": f("홍수위"),
        "제체": f"{row['제체형식']} H={row['제체높이']}m L={row['제체길이']}m",
        "한발빈도(년)": f("한발빈도"), "연락처": row["연락처"],
    }


CALIB_JSON = Path(__file__).resolve().parents[1] / "data" / "calibrated_params.json"


def load_saved_calibration() -> dict:
    import json
    try:
        return json.loads(CALIB_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def save_calibration(code: str, params: dict, meta: dict | None = None) -> None:
    """웹앱·스크립트에서 보정한 매개변수를 저수지별로 저장 → 다음부터 config_for 가 자동 적용"""
    import json
    d = load_saved_calibration()
    d[str(code)] = {**{k: round(float(v), 4) for k, v in params.items()
                       if k in ("x1", "x2", "x3", "x4", "release_scale", "cap_frac")}, **(meta or {})}
    CALIB_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def estimate_irrigation_area(row: pd.Series) -> float:
    """제원표에 수혜면적이 없을 때(0·공란) 같은 본부 큰 저수지들의 '수혜면적÷유효저수량' 중앙값으로 추정 (ha)"""
    df = load_catalog()
    g = df[(df["본부"] == row["본부"]) & (df["수혜면적"] > 0) & (df["유효저수량"] > 0) & (df["총저수량"] >= 1000)]
    if len(g) < 10:
        g = df[(df["수혜면적"] > 0) & (df["유효저수량"] > 0) & (df["총저수량"] >= 1000)]
    ratio = float((g["수혜면적"] / g["유효저수량"]).median())
    eff = float(row["유효저수량"]) if pd.notna(row["유효저수량"]) and row["유효저수량"] > 0 else float(row["총저수량"]) * 0.9
    return round(eff * ratio, 0)


def config_for(code: str, base_cfg: dict) -> dict:
    """제원표 값으로 모델 설정을 만듭니다(나머지는 base_cfg 기본값 유지)."""
    row = get(code)
    cfg = copy.deepcopy(base_cfg)
    total = float(row["총저수량"])
    eff = float(row["유효저수량"]) if pd.notna(row["유효저수량"]) and row["유효저수량"] > 0 else total * 0.9
    eff = min(eff, total)
    full_area_km2 = float(row["만수면적"]) / 100 if pd.notna(row["만수면적"]) else 0.0
    r = cfg["reservoir"]
    r.update({
        "code": row["표준코드"], "name": row["시설명"], "location": row["소재지"],
        "completed": int(str(row["준공년도"])[:4]) if str(row["준공년도"])[:4].isdigit() else None,
        "total_storage": total, "effective_storage": eff, "dead_storage": total - eff,
        # 평균 수면적 ≈ 만수면적의 70% (저수위 변동 반영한 근사)
        "surface_area_km2": round(full_area_km2 * 0.7, 3),
        "full_level": row["만수위"], "dead_level": row["사수위"],
        "full_area_ha": float(row["만수면적"]) if pd.notna(row["만수면적"]) else None,
        "dam_height": float(row["제체높이"]) if pd.notna(row["제체높이"]) else None,
    })
    # 성주댐 외에는 좌표가 제원표에 없으므로 지도는 표시하지 않음
    if row["표준코드"] != base_cfg["reservoir"].get("code"):
        r["lat"] = r["lon"] = None
    cfg["catchment"]["area_km2"] = float(row["유역면적"]) / 100
    cal = base_cfg.get("calibrated", {}).get(str(row["표준코드"])) or load_saved_calibration().get(str(row["표준코드"]))
    if cal:  # 실측으로 보정한 저수지: GR4J·공급계수·운영 상한 적용
        cfg["gr4j"] = {k: float(cal[k]) for k in ("x1", "x2", "x3", "x4", "release_scale", "cap_frac") if k in cal}
        cfg.setdefault("operation", {})["ops_cap_frac"] = float(cal.get("cap_frac", 1.0))
    else:
        cfg.setdefault("operation", {})["ops_cap_frac"] = 1.0
    area = float(row["수혜면적"]) if pd.notna(row["수혜면적"]) else 0.0
    cfg["irrigation"]["area_estimated"] = area <= 0
    cfg["irrigation"]["area_ha"] = area if area > 0 else estimate_irrigation_area(row)
    return cfg
