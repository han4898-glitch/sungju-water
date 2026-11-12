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


# 제원표 오류 보정 (원본 CSV는 그대로 두고 이 파일의 값으로 덮어씀)
CORR_CSV = Path(__file__).resolve().parents[1] / "data" / "spec_corrections.csv"
CORR_COLS = ["총저수량", "유효저수량", "사수량", "수혜면적", "유역면적", "만수면적", "제체높이"]


def load_corrections() -> pd.DataFrame:
    if not CORR_CSV.exists():
        return pd.DataFrame(columns=["표준코드", *CORR_COLS, "근거"])
    return pd.read_csv(CORR_CSV, dtype={"표준코드": str, "근거": str})


@lru_cache(maxsize=1)
def load_catalog(path: str = str(CATALOG_CSV)) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"표준코드": str, "연락처": str})
    df["시설명"] = df["시설명"].astype(str).str.strip()
    df["_addr"] = df["소재지"].fillna("")
    df["_보정"] = ""
    corr = load_corrections()
    for _, c in corr.iterrows():
        m = df["표준코드"] == str(c["표준코드"])
        if not m.any():
            continue
        changed = []
        for col in CORR_COLS:
            if col in c and pd.notna(c[col]):
                old = df.loc[m, col].iloc[0]
                df.loc[m, col] = float(c[col])
                changed.append(f"{col} {old:g}→{float(c[col]):g}")
        df.loc[m, "_보정"] = ", ".join(changed) + (f" (근거: {c['근거']})" if pd.notna(c.get("근거")) else "")
    return df


def save_correction(code: str, values: dict, note: str = "") -> None:
    """제원 보정값 저장 (같은 저수지는 덮어씀) — 저장 후 카탈로그 캐시를 비움"""
    corr = load_corrections()
    corr = corr[corr["표준코드"] != str(code)]
    row = {"표준코드": str(code), **{k: values.get(k) for k in CORR_COLS}, "근거": note}
    corr = pd.concat([corr, pd.DataFrame([row])], ignore_index=True)
    tmp = CORR_CSV.with_suffix(".tmp")
    corr.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(CORR_CSV)
    load_catalog.cache_clear()


def spec_issues(row: pd.Series) -> dict:
    """제원 이상 여부 점검 → {"issues": [설명…], "x10": 저수량이 만㎥로 잘못 들어간 것으로 보이면 True}

    - 평균수심(총저수량 ÷ 만수면적)이 제체높이의 8% 미만이면 저수량이 너무 작음
      (전국 중앙값 34%, 10배 하면 정상 범위 15~80%에 들어오면 '만㎥ 단위 입력' 오류로 추정)
    - 수혜면적 1ha당 총저수량이 0.5천㎥ 미만 (전국 하위 5% 아래)
    - 유효저수량 > 총저수량
    """
    f = lambda c: float(row[c]) if c in row and pd.notna(row[c]) else 0.0
    tot, eff, area, h, irr = f("총저수량"), f("유효저수량"), f("만수면적"), f("제체높이"), f("수혜면적")
    issues, x10 = [], False
    if tot > 0 and area > 0 and h > 0:
        depth = tot / area / 10  # m (천㎥ ÷ ha ÷ 10)
        ratio = depth / h
        if ratio < 0.08:
            issues.append(f"평균수심 {depth:.2f}m가 제체높이 {h:g}m의 {ratio:.0%}로 너무 얕음 (보통 25~45%)")
            # 10배 했을 때 수심비가 정상 범위이고, 수혜면적 대비 저수량도 작을 때만 '만㎥ 입력'으로 판단
            # (고삼처럼 저수량은 맞고 제체높이가 잘못된 경우를 거르기 위함)
            x10 = 0.15 <= ratio * 10 <= 0.8 and (irr <= 0 or tot / irr < 3.0)
            if not x10 and irr > 0 and tot / irr >= 3.0:
                issues[-1] += " — 저수량은 수혜면적에 비해 정상이므로 제체높이·만수면적 오류 가능"
        elif ratio > 1.5:
            issues.append(f"평균수심 {depth:.1f}m가 제체높이 {h:g}m보다 훨씬 큼 — 만수면적이나 제체높이 오류 의심")
    if tot > 0 and irr > 0 and tot / irr < 0.5:
        issues.append(f"수혜면적 1ha당 저수량 {tot / irr:.2f}천㎥ — 전국 하위 5% 아래")
    if eff > tot * 1.001 > 0:
        issues.append(f"유효저수량 {eff:g}이 총저수량 {tot:g}보다 큼")
    if x10:
        issues.append("총·유효저수량이 천㎥가 아니라 만㎥로 입력된 것으로 보임 (10배 하면 정상 범위)")
    return {"issues": issues, "x10": x10}


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
