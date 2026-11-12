"""GR4J 매개변수 보정 — 관측 저수량(RAWRIS 일 저수율)에 맞춥니다.

농업용 저수지는 유입량을 직접 재지 않으므로, '관측 공급량'을 넣은 저수지 물수지 모의 저수량이
관측 저수량과 같아지도록 유입량 모형(GR4J) 매개변수를 찾습니다.
목적함수: 저수율 NSE 최대화 (scipy Nelder-Mead / differential evolution)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .gr4j import run_gr4j, runoff_to_volume


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    m = ~np.isnan(obs)
    sim, obs = sim[m], obs[m]
    return float(1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))


def rmse(sim: np.ndarray, obs: np.ndarray) -> float:
    m = ~np.isnan(obs)
    return float(np.sqrt(np.mean((sim[m] - obs[m]) ** 2)))


def simulate_storage_with_observed_release(
    params, rain, et0, release, evap, area_km2, total, dead, v0, flood_mask=None
) -> np.ndarray:
    """params = [x1, x2, x3, x4, (공급 배율), (평상시 운영 상한 0~1), (홍수기 상한 0~1)]
    flood_mask: 홍수기(6/21~9/20) 날짜 True 배열"""
    x1, x2, x3, x4 = params[:4]
    if len(params) > 4:  # 공급량 배율 (공급 기록이 없을 때 필요수량 추정치를 보정)
        release = release * params[4]
    n = len(rain)
    cap = np.full(n, float(total))
    if len(params) > 5:  # 운영 상한: 이 저수율을 넘는 물은 방류(월류·사전방류)
        cap[:] = dead + params[5] * (total - dead)
    if len(params) > 6 and flood_mask is not None:  # 홍수기 제한수위
        cap[np.asarray(flood_mask)] = dead + params[6] * (total - dead)
    q = runoff_to_volume(run_gr4j(rain, et0, x1, x2, x3, x4), area_km2)
    v = v0
    out = np.empty(n)
    for i in range(n):
        v = v + q[i] - release[i] - evap[i]
        v = min(max(v, dead), cap[i])
        out[i] = v
    return out


def flood_season(idx: pd.DatetimeIndex) -> np.ndarray:
    md = idx.month * 100 + idx.day
    return np.asarray((md >= 621) & (md <= 920))


def calibrate_gr4j(
    inputs: pd.DataFrame,
    observed: pd.DataFrame,
    cfg: dict,
    bounds=((50, 1500), (-5, 3), (10, 400), (0.6, 5)),
    maxiter: int = 40,
    seed: int = 0,
    fit_release_scale: bool = False,
) -> dict:
    """observed: index=date, 열 storage(천㎥) 와 release(천㎥/day, 관측 또는 추정 공급량)"""
    res = cfg["reservoir"]
    df = inputs.join(observed, how="inner")
    obs = df["storage"].values
    args = (
        df["rain"].values, df["et0"].values, df["release"].values, df["evap"].values,
        cfg["catchment"]["area_km2"], res["total_storage"], res["dead_storage"], obs[0], flood_season(df.index),
    )

    def loss(p):
        v = -nse(simulate_storage_with_observed_release(p, *args), obs)
        return v if np.isfinite(v) else 1e6

    if fit_release_scale:
        # 공급 배율, 운영 상한 (홍수기 상한을 따로 두는 7번째 매개변수는 검증 성능이 나아지지 않아 기본 미사용)
        bounds = tuple(bounds) + ((0.3, 1.5), (0.6, 1.0))
    r = differential_evolution(loss, bounds, maxiter=maxiter, seed=seed, tol=1e-6, polish=True)
    sim = simulate_storage_with_observed_release(r.x, *args)
    eff = res["total_storage"] - res["dead_storage"]
    to_pct = lambda v: (v - res["dead_storage"]) / eff * 100
    return {
        "params": dict(zip(["x1", "x2", "x3", "x4", "release_scale", "cap_frac", "cap_flood_frac"], map(float, r.x))),
        "nse": nse(sim, obs),
        "rmse_pct": rmse(to_pct(sim), to_pct(obs)),
        "series": pd.DataFrame({"obs_pct": to_pct(obs), "sim_pct": to_pct(sim)}, index=df.index),
    }


def evaluate_params(inputs: pd.DataFrame, observed: pd.DataFrame, cfg: dict, params: dict) -> dict:
    """보정된 매개변수를 다른 기간(검증기간)에 적용해 성능을 평가합니다."""
    res = cfg["reservoir"]
    df = inputs.join(observed, how="inner")
    obs = df["storage"].values
    sim = simulate_storage_with_observed_release(
        [params["x1"], params["x2"], params["x3"], params["x4"]] + (
            [params["release_scale"], params.get("cap_frac", 1.0), params.get("cap_flood_frac", params.get("cap_frac", 1.0))]
            if "release_scale" in params else []),
        df["rain"].values, df["et0"].values, df["release"].values, df["evap"].values,
        cfg["catchment"]["area_km2"], res["total_storage"], res["dead_storage"], obs[0], flood_season(df.index),
    )
    eff = res["total_storage"] - res["dead_storage"]
    to_pct = lambda v: (v - res["dead_storage"]) / eff * 100
    return {
        "nse": nse(sim, obs),
        "rmse_pct": rmse(to_pct(sim), to_pct(obs)),
        "series": pd.DataFrame({"obs_pct": to_pct(obs), "sim_pct": to_pct(sim)}, index=df.index),
    }
