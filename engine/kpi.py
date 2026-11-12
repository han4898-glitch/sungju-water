"""용수공급 성능지표(KPI)"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _max_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def compute_kpis(s: pd.DataFrame, cfg: dict) -> dict:
    """
    - 용수공급 만족률(양적 신뢰도) : 공급량 / 필요수량
    - 시간 신뢰도                 : 필요수량을 95% 이상 공급한 날 / 관개일
    - 부족량, 부족일수, 최장 연속 부족일
    - 최저 유효저수율과 발생일, 기말 저수율
    - 무효방류(월류)량
    - 부족 심도지수 SI = 평균[(부족량/필요수량)²]×100  (작을수록 좋음)
      같은 부족량이라도 한꺼번에 크게 모자라는 것을 더 나쁘게 평가 → 제한급수(헤징) 효과를 드러냄
    """
    irr_days = s["demand"] > 0.01
    dem = s.loc[irr_days, "demand"].sum()
    sup = s.loc[irr_days, "supply"].sum()
    short = (s["supply"] < 0.95 * s["demand"]) & irr_days
    imin = s["storage_pct"].idxmin()
    d10 = s["deficit"].rolling(10).sum()
    q10 = s["demand"].rolling(10).sum()
    worst10 = float((d10 / q10.where(q10 > 1)).max() * 100) if q10.max() > 1 else 0.0
    ratio = (s.loc[irr_days, "deficit"] / s.loc[irr_days, "demand"]).clip(0, 1)
    shortage_index = float((ratio ** 2).mean() * 100) if irr_days.any() else 0.0
    return {
        "shortage_index": shortage_index,
        "volumetric_reliability_pct": float(sup / dem * 100) if dem > 0 else 100.0,
        "time_reliability_pct": float((1 - short.sum() / max(irr_days.sum(), 1)) * 100),
        "total_demand_k_m3": float(dem),
        "total_supply_k_m3": float(sup),
        "total_deficit_k_m3": float(s["deficit"].sum()),
        "deficit_days": int(short.sum()),
        "max_consecutive_deficit_days": _max_run(short.values),
        "worst_10day_shortage_pct": 0.0 if np.isnan(worst10) else worst10,
        "min_storage_pct": float(s["storage_pct"].min()),
        "min_storage_date": imin.strftime("%Y-%m-%d"),
        "end_storage_pct": float(s["storage_pct"].iloc[-1]),
        "spill_k_m3": float(s["spill"].sum()),
        "inflow_k_m3": float(s["inflow"].sum()),
    }


def quick_si_min(s: pd.DataFrame, dead: float, eff: float) -> tuple[float, float, float]:
    """최적화 반복용 빠른 지표: (부족 심도지수 SI, 최저 유효저수율 %, 기말 유효저수율 %) — compute_kpis와 같은 정의"""
    dem = s["demand"].to_numpy()
    sup = s["supply"].to_numpy()
    pct = (s["storage"].to_numpy() - dead) / eff * 100
    m = dem > 0.01
    si = float(((np.clip((dem[m] - sup[m]).clip(min=0) / dem[m], 0, 1)) ** 2).mean() * 100) if m.any() else 0.0
    return si, float(pct.min()), float(pct[-1])


KPI_LABELS = {
    "shortage_index": ("부족 심도지수(SI)", ""),
    "volumetric_reliability_pct": ("용수공급 만족률", "%"),
    "time_reliability_pct": ("시간 신뢰도", "%"),
    "total_deficit_k_m3": ("총 부족량", "천㎥"),
    "deficit_days": ("부족일수", "일"),
    "max_consecutive_deficit_days": ("최장 연속 부족", "일"),
    "worst_10day_shortage_pct": ("최악 10일 부족률", "%"),
    "min_storage_pct": ("최저 유효저수율", "%"),
    "end_storage_pct": ("기말 유효저수율", "%"),
    "spill_k_m3": ("월류(무효방류)", "천㎥"),
}
