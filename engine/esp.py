"""앙상블 유량 예측(ESP, Ensemble Streamflow Prediction) — 기상청 과거 관측으로 앞으로의 저수율 확률 전망

방법 (미국 NWS·K-water 장기 유량전망에 쓰는 표준 방식)
  1) 오늘까지는 실제 관측 기상으로 유출 모형(GR4J) 상태를 맞춤
  2) 앞으로 1~3일은 기상청 단기예보 사용
  3) 그 이후는 '과거 각 해의 같은 날짜 기상'을 그대로 이어 붙여 해마다 하나씩 미래 시나리오(앙상블 멤버)를 만듦
     → 2015~2025년 관측이면 11가지 '있을 법한 미래'
  4) 각 멤버로 저수지 운영을 모의 → 날짜별 저수율 분포(10/50/90%), 관리단계 진입 확률, 부족 확률
  5) 기상청 1·3개월 전망(강수 많음/비슷/적음 확률)을 넣으면, 과거 해들을 강수량 3분위로 나눠 가중치를 줌
  6) 관측 오차 보정: 과거 여러 시점에서 '그때 관측 저수율로 시작해 실제 기상으로 다시 예측'(재예측, hindcast)한
     모의값과 실제 저수율의 차이를 리드타임·모의값 구간별로 모아, 앙상블에 더해 줌 (모형 구조 오차 반영)
     → 성주·묘곡 2019~2025 역검증: 10~90% 범위 적중률 43% → 79% (이상적 80%)
운영규칙 선택: 멤버 전체의 기대 SI(부족 심도지수)가 가장 작은 규칙 = 불확실성을 고려한 최적 운영
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .kpi import compute_kpis, quick_si_min
from .operation import PRESETS, Scenario, forecast_factor, normal_series, presets_for
from .optimize import BOUNDS, decode
from .reference_engine import run_reference
from .runner import prepare_inputs

COLS = ["rain", "tmax", "tmin", "rh", "wind", "sunshine"]
STAGES = (("관심", 0), ("주의", 1), ("경계", 2))


def build_members(hist: pd.DataFrame, start: pd.Timestamp, days: int,
                  forecast: pd.DataFrame | None = None, warmup_days: int = 730) -> dict[int, pd.DataFrame]:
    """과거 해별 앙상블 기상 {해: 기상 DataFrame(워밍업 관측 + 미래)}"""
    start = pd.Timestamp(start).normalize()
    end = start + pd.Timedelta(days=days - 1)
    cols = [c for c in COLS if c in hist]
    obs = hist.loc[: start - pd.Timedelta(days=1), cols]
    obs = obs.loc[start - pd.Timedelta(days=warmup_days):]
    members = {}
    for y in sorted(set(hist.index.year)):
        shift = start.year - y
        if shift <= 0:
            continue
        s_y = start - pd.DateOffset(years=shift)
        e_y = s_y + pd.Timedelta(days=days - 1)
        seg = hist.loc[s_y:e_y, cols]
        if len(seg) < days * 0.95 or seg["rain"].isna().mean() > 0.05:
            continue  # 그해 자료가 부족하면 제외
        fut = seg.reset_index(drop=True).reindex(range(days))
        fut.index = pd.date_range(start, end, freq="D")
        if forecast is not None and len(forecast):  # 처음 며칠은 단기예보로 덮어씀
            f = forecast.reindex(fut.index).dropna(how="all")
            for c in f.columns:
                if c in fut:
                    m = f[c].notna()
                    fut.loc[f.index[m], c] = f.loc[m, c]
        w = pd.concat([obs, fut]).astype(float)
        members[int(y)] = w[~w.index.duplicated(keep="last")].asfreq("D").interpolate(limit_direction="both")
    return members


def tercile_weights(members: dict[int, pd.DataFrame], start: pd.Timestamp, days: int,
                    probs: tuple[float, float, float] | None = None, window_days: int = 90) -> pd.Series:
    """기상청 장기전망 (적음, 비슷, 많음) 확률 → 과거 해 가중치. probs 없으면 모두 같음"""
    years = list(members)
    if not probs:
        return pd.Series(1 / len(years), index=years)
    n = min(days, window_days)
    tot = pd.Series({y: float(members[y].loc[start:start + pd.Timedelta(days=n - 1), "rain"].sum()) for y in years})
    rank = tot.rank(method="first")
    k = len(years)
    cat = np.where(rank <= k / 3, 0, np.where(rank <= 2 * k / 3, 1, 2))
    p = np.asarray(probs, float) / max(sum(probs), 1e-9)
    w = np.array([p[c] / max((cat == c).sum(), 1) for c in cat])
    return pd.Series(w / w.sum(), index=years)


def prepare_members(members: dict[int, pd.DataFrame], cfg: dict, gr4j_params: dict | None) -> dict[int, pd.DataFrame]:
    return {y: prepare_inputs(w, cfg, gr4j_params) for y, w in members.items()}


def _run(inp: pd.DataFrame, cfg: dict, sc: Scenario, start, end, current_pct: float) -> pd.DataFrame:
    res = cfg["reservoir"]
    total, dead = res["total_storage"], res["dead_storage"]
    cap = dead + (total - dead) * float(cfg.get("operation", {}).get("ops_cap_frac", 1.0))
    v0 = min(dead + (total - dead) * current_pct / 100, cap)
    df = inp.loc[start:end]
    out = run_reference(df["inflow"], df["demand"], df["evap"], forecast_factor(df["rain"], sc), sc, total, dead, v0,
                        cfg["environment"]["river_maintenance_flow"], cap, normal_series(cfg, df.index))
    out["storage_pct"] = (out["storage"] - dead) / (total - dead) * 100
    out["deficit"] = (out["demand"] - out["supply"]).clip(lower=0)
    return out


def hindcast_errors(inputs: pd.DataFrame, cfg: dict, obs_rate: pd.Series, days: int,
                    before: pd.Timestamp | None = None, step: int = 15) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """과거 재예측 오차 {리드(일): (모의 저수율, 관측−모의)} — 실제 기상·관측 시작값으로 days일 모의를 step일마다 반복"""
    obs = obs_rate.dropna().clip(0, 100)
    last = min(inputs.index.max(), obs.index.max()) - pd.Timedelta(days=days)
    if before is not None:
        last = min(last, pd.Timestamp(before) - pd.Timedelta(days=days))
    first = max(inputs.index.min() + pd.Timedelta(days=365), obs.index.min() + pd.Timedelta(days=1))
    sims, errs = [], []
    sc = PRESETS["관행 운영"]
    for st in pd.date_range(first, last, freq=f"{step}D"):
        prev = obs.loc[:st - pd.Timedelta(days=1)]
        if prev.empty or (st - prev.index[-1]).days > 3:
            continue
        o = _run(inputs, cfg, sc, st, st + pd.Timedelta(days=days - 1), float(prev.iloc[-1]))["storage_pct"]
        sims.append(o.values)
        errs.append(obs.reindex(o.index).values - o.values)
    if not sims:
        return {}
    S, E = np.array(sims), np.array(errs)
    return {L: (S[:, L][~np.isnan(E[:, L])], E[:, L][~np.isnan(E[:, L])]) for L in range(days)}


def _corrected_samples(v: float, sim: np.ndarray, err: np.ndarray, tol: float = 5.0, k_min: int = 8) -> np.ndarray:
    """모의값 v 근처(±tol %p)였던 과거 재예측의 오차를 더한 표본"""
    m = np.abs(sim - v) <= tol
    if m.sum() < k_min:
        m = np.zeros(len(sim), bool)
        m[np.argsort(np.abs(sim - v))[:k_min]] = True
    return np.clip(v + err[m], 0, 100)


def run_esp(prepared: dict[int, pd.DataFrame], cfg: dict, sc: Scenario, start, days: int, current_pct: float,
            weights: pd.Series | None = None, errors: dict | None = None) -> dict:
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(days=days - 1)
    years = list(prepared)
    w = weights if weights is not None else pd.Series(1 / len(years), index=years)
    stor, kp = {}, {}
    for y in years:
        o = _run(prepared[y], cfg, sc, start, end, current_pct)
        stor[y] = o["storage_pct"]
        kp[y] = compute_kpis(o, cfg)
    S = pd.DataFrame(stor)
    K = pd.DataFrame(kp).T
    wv = w.reindex(S.columns).fillna(0).values

    def wq(row, q):
        o = np.argsort(row.values)
        c = np.cumsum(wv[o])
        return float(row.values[o][np.searchsorted(c, q * c[-1])])

    band = pd.DataFrame({f"p{int(q * 100)}": S.apply(lambda r: wq(r, q), axis=1) for q in (0.1, 0.5, 0.9)})
    band["mean"] = S.values @ wv
    curves = list(sc.curves_pct) or [60.0, 40.0, 25.0]
    risk = {}
    corrected = bool(errors) and len(errors) >= days
    if corrected:  # 관측 오차 보정: 날짜별로 (멤버 모의값 + 과거 재예측 오차) 표본의 가중 분위수
        rows, below = [], {c: 0.0 for c in curves[:3]}
        for L in range(days):
            sim_L, err_L = errors[L]
            vals, ww = [], []
            for j, v in enumerate(S.iloc[L].values):
                smp = _corrected_samples(v, sim_L, err_L) if len(sim_L) else np.array([v])
                vals.append(smp)
                ww.append(np.full(len(smp), wv[j] / len(smp)))
            vals, ww = np.concatenate(vals), np.concatenate(ww)
            o = np.argsort(vals)
            c = np.cumsum(ww[o]) / ww.sum()
            rows.append([float(vals[o][np.searchsorted(c, q)]) for q in (0.1, 0.5, 0.9)] + [float(vals @ ww / ww.sum())])
            for cv in below:
                below[cv] = max(below[cv], float(ww[vals < cv].sum() / ww.sum() * 100))
        band_raw = band
        band = pd.DataFrame(rows, index=S.index, columns=["p10", "p50", "p90", "mean"])
        for name, i in STAGES:
            if i < len(curves):
                risk[f"{name}({curves[i]:.0f}%) 이하 확률"] = below[curves[i]]
    else:
        band_raw = band
        for name, i in STAGES:
            if i < len(curves):
                risk[f"{name}({curves[i]:.0f}%) 이하 확률"] = float(((S.min() < curves[i]).values * wv).sum() * 100)
    # 계획적 제한급수(헤징)는 부족으로 치지 않고, 10일 동안 필요수량의 절반 이상 모자라는 '심각 부족'만 셈
    risk["심각 부족(10일 50%↑) 확률"] = float(((K["worst_10day_shortage_pct"].astype(float) >= 50).values * wv).sum() * 100)
    return {"band": band, "band_raw": band_raw, "corrected": corrected, "members": S, "kpis": K, "weights": w,
            "expected": {k: float(K[k].astype(float).values @ wv) for k in
                         ("shortage_index", "volumetric_reliability_pct", "worst_10day_shortage_pct",
                          "min_storage_pct", "end_storage_pct", "total_deficit_k_m3")},
            "risk": risk}


def compare_rules(prepared, cfg, start, days, current_pct, weights=None, extra: dict | None = None,
                  errors: dict | None = None) -> pd.DataFrame:
    rows = {}
    for name, sc in {**presets_for(cfg), **(extra or {})}.items():
        r = run_esp(prepared, cfg, Scenario.from_dict(sc.to_dict()), start, days, current_pct, weights, errors)
        rows[name] = {**r["expected"], **r["risk"]}
    return pd.DataFrame(rows).T


def optimize_rules_esp(prepared, cfg, start, days, current_pct, weights=None, maxiter: int = 12,
                       seed: int = 0, progress=None) -> dict:
    """앙상블 전체의 기대 SI가 최소인 운영규칙 (불확실성을 고려한 최적화)"""
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(days=days - 1)
    years = list(prepared)
    w = (weights if weights is not None else pd.Series(1 / len(years), index=years)).reindex(years).values

    dead = cfg["reservoir"]["dead_storage"]
    eff = cfg["reservoir"]["total_storage"] - dead

    def loss(x):
        sc = decode(x)
        return float(sum(wi * quick_si_min(_run(prepared[y], cfg, sc, start, end, current_pct), dead, eff)[0]
                         for y, wi in zip(years, w) if wi > 0))

    it = [0]

    def cb(xk, convergence=None):  # noqa: ARG001
        it[0] += 1
        if progress:
            progress(min(it[0] / maxiter, 1.0), f"앙상블 최적화 {it[0]}/{maxiter}세대")

    x0 = [60.0, 20.0, 15.0, 0.9, 0.75 / 0.9, 0.55 / 0.75, 0.1]
    r = differential_evolution(loss, BOUNDS, maxiter=maxiter, popsize=8, seed=seed, tol=1e-4, polish=False,
                               callback=cb, x0=x0)
    sc = decode(r.x)
    sc.name = "앙상블 최적 규칙"
    return {"scenario": sc, "expected_si": float(r.fun)}
