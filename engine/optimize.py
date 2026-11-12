"""방류(공급) 운영규칙 최적화 — 어느 저수지든 제원·기상·유입량 모형만 있으면 같은 방식으로 실행됩니다.

찾는 것: 저수율 경계 3개(관심·주의·경계)와 구간별 공급 비율 3개 (+ 선택: 예보 연계 절감률)
목적함수: 부족 심도지수 SI = 평균[(부족량/필요수량)²] 최소화
          → 한 번에 크게 모자라는 것보다 미리 조금씩 아끼는 쪽(헤징)을 고르게 됨
제약(벌점, 선택): 기말 유효저수율 ≥ 목표 (다음 해를 위해 남겨 둘 물)
기존 '단계별 제한급수' 규칙을 탐색 시작점에 넣어, 최적안은 항상 그보다 같거나 낫습니다.
탐색: scipy differential evolution, 계산은 기준 물수지 엔진(pywr와 교차검증된 동일 규칙)으로 빠르게
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .kpi import compute_kpis, quick_si_min
from .operation import Scenario, forecast_factor, normal_series
from .reference_engine import run_reference

# x = [t1, gap12, gap23, f2, f3_ratio, f4_ratio, forecast_cut]
BOUNDS = [(35.0, 85.0), (5.0, 30.0), (5.0, 25.0), (0.6, 1.0), (0.6, 1.0), (0.5, 1.0), (0.0, 0.3)]


def decode(x, use_forecast: bool = True, forecast_days: int = 10, forecast_mm: float = 20.0) -> Scenario:
    t1 = float(x[0])
    t2 = max(t1 - float(x[1]), 10.0)
    t3 = max(t2 - float(x[2]), 5.0)
    f2 = float(x[3])
    f3 = f2 * float(x[4])
    f4 = f3 * float(x[5])
    cut = float(x[6]) if use_forecast else 0.0
    return Scenario(
        name="최적 운영규칙",
        description=(f"저수율 {t1:.0f}/{t2:.0f}/{t3:.0f}% 아래에서 공급 {f2:.0%}/{f3:.0%}/{f4:.0%}"
                     + (f" + 향후 {forecast_days}일 예보강수 {forecast_mm:g}mm 미만이면 {cut:.0%} 추가 절감" if cut > 0.005 else "")),
        curves_pct=[round(t1, 1), round(t2, 1), round(t3, 1)],
        supply_factors=[1.0, round(f2, 3), round(f3, 3), round(f4, 3)],
        forecast_days=forecast_days if cut > 0.005 else 0,
        forecast_threshold_mm=forecast_mm if cut > 0.005 else 0.0,
        forecast_cut=round(cut, 3) if cut > 0.005 else 0.0,
    )


def _run(inputs: pd.DataFrame, cfg: dict, sc: Scenario, v0: float, cap: float, demand_scale: float, inflow_scale: float):
    res = cfg["reservoir"]
    total, dead = res["total_storage"], res["dead_storage"]
    out = run_reference(inputs["inflow"] * inflow_scale, inputs["demand"] * demand_scale, inputs["evap"],
                        forecast_factor(inputs["rain"], sc), sc, total, dead, v0,
                        cfg["environment"]["river_maintenance_flow"], cap, normal_series(cfg, inputs.index))
    out["storage_pct"] = (out["storage"] - dead) / (total - dead) * 100
    out["deficit"] = (out["demand"] - out["supply"]).clip(lower=0)
    return out


def optimize_rules(inputs: pd.DataFrame, cfg: dict, start: str | None = None, end: str | None = None,
                   initial_pct: float = 70.0, demand_scale: float = 1.0, inflow_scale: float = 1.0,
                   end_target_pct: float = 0.0,
                   use_forecast: bool = True, maxiter: int = 25, seed: int = 0, progress=None) -> dict:
    """end_target_pct: 기말에 최소한 남길 유효저수율(%) — 0이면 제약 없음"""
    df = inputs.loc[start:end]
    res = cfg["reservoir"]
    total, dead = res["total_storage"], res["dead_storage"]
    cap = dead + (total - dead) * float(cfg.get("operation", {}).get("ops_cap_frac", 1.0))
    v0 = min(dead + (total - dead) * initial_pct / 100, cap)
    n_eval = [0]

    def score(k: dict) -> float:
        return k["shortage_index"] + max(0.0, end_target_pct - k["end_storage_pct"]) ** 2 * 0.05

    dead_, eff_ = res["dead_storage"], res["total_storage"] - res["dead_storage"]

    def loss(x):
        n_eval[0] += 1
        si, _, end_pct = quick_si_min(_run(df, cfg, decode(x, use_forecast), v0, cap, demand_scale, inflow_scale), dead_, eff_)
        return score({"shortage_index": si, "end_storage_pct": end_pct})

    bounds = BOUNDS if use_forecast else BOUNDS[:-1] + [(0.0, 0.0)]
    cb = None
    if progress:
        it = [0]

        def cb(xk, convergence=None):  # noqa: ARG001
            it[0] += 1
            progress(min(it[0] / maxiter, 1.0), f"최적화 {it[0]}/{maxiter}세대")

    x_hedge = [60.0, 20.0, 15.0, 0.9, 0.75 / 0.9, 0.55 / 0.75, 0.1 if use_forecast else 0.0]  # 단계별 제한급수(+예보)
    r = differential_evolution(loss, bounds, maxiter=maxiter, popsize=10, seed=seed, tol=1e-4,
                               polish=False, callback=cb, x0=x_hedge)
    best = decode(r.x, use_forecast)
    base = Scenario(name="관행 운영", curves_pct=[], supply_factors=[1.0])
    k_best = compute_kpis(_run(df, cfg, best, v0, cap, demand_scale, inflow_scale), cfg)
    k_base = compute_kpis(_run(df, cfg, base, v0, cap, demand_scale, inflow_scale), cfg)
    return {"scenario": best, "kpis": k_best, "baseline_kpis": k_base, "score": float(r.fun),
            "baseline_score": score(k_base), "evaluations": n_eval[0],
            "end_target_pct": end_target_pct}
