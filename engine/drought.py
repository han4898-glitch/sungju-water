"""가뭄 대응 운영규칙의 실제 가뭄 해 재현 검증과 과적합 확인

1) 다년 최적화 (optimize_multi)
   여러 해의 관개기(4/1~10/31)를 한꺼번에 모의해 '평균 부족 심도지수 SI'가 가장 작은 규칙 하나를 찾음.
   → 특정 한 해의 비를 미리 아는 '사후 최적'이 아니라, 어느 해에도 무난한 규칙 (실제 운영에 쓸 수 있는 형태)
2) 가뭄 해 재현 (replay)
   2017·2022년 같은 실제 가뭄 해에, 그해 4/1 관측 저수율에서 시작해 실제 기상으로
   관행 운영 / 기존 제한급수 / 최적 규칙을 모의하고 관측 저수율과 함께 비교
3) 과적합 확인 (out_of_sample)
   규칙을 찾을 때 쓰지 않은 해(검증 해)에 적용해도 관행보다 나은지 확인
   - 시간 분할: 2015~2021년으로 찾고 2022년 이후에 적용
   - 한 해 빼기: 검증할 가뭄 해를 뺀 나머지 해로 찾고 그 해에 적용
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .kpi import compute_kpis, quick_si_min
from .operation import PRESETS, Scenario, presets_for
from .optimize import BOUNDS, _run, decode

SEASON = ("04-01", "10-31")
KPI_COLS = ["shortage_index", "worst_10day_shortage_pct", "volumetric_reliability_pct",
            "min_storage_pct", "end_storage_pct", "total_deficit_k_m3"]


def season_of(year: int, season=SEASON) -> tuple[str, str]:
    return f"{year}-{season[0]}", f"{year}-{season[1]}"


def start_pct(obs_rate: pd.Series, start: str, default: float = 70.0) -> float:
    prev = obs_rate.dropna().loc[: pd.Timestamp(start) - pd.Timedelta(days=1)]
    return float(min(prev.iloc[-1], 100.0)) if len(prev) else default


def _cap(cfg: dict) -> float:
    r = cfg["reservoir"]
    return r["dead_storage"] + (r["total_storage"] - r["dead_storage"]) * float(cfg.get("operation", {}).get("ops_cap_frac", 1.0))


def _v0(cfg: dict, pct: float) -> float:
    r = cfg["reservoir"]
    return min(r["dead_storage"] + (r["total_storage"] - r["dead_storage"]) * pct / 100, _cap(cfg))


def typical_start_pct(obs_rate: pd.Series, season=SEASON) -> float:
    """관측이 없는 옛날 해(장기 기상자료)의 시작 저수율 = 관측된 해들의 4/1 저수율 중앙값"""
    v = [start_pct(obs_rate, season_of(y, season)[0], np.nan) for y in sorted(set(obs_rate.dropna().index.year))]
    v = [x for x in v if not np.isnan(x)]
    return float(np.median(v)) if v else 70.0


def weather_years(inputs: pd.DataFrame, season=SEASON) -> list[int]:
    """기상자료만 있으면 되는 해 (다년 최적화 학습용 — 장기 관측 1973~ 이면 50개 안팎)"""
    return [y for y in sorted(set(inputs.index.year))
            if pd.Timestamp(season_of(y, season)[0]) >= inputs.index.min() + pd.Timedelta(days=180)
            and pd.Timestamp(season_of(y, season)[1]) <= inputs.index.max()]


def training_years(inputs: pd.DataFrame, obs_rate: pd.Series, season=SEASON, n_dry: int = 15,
                   exclude: list[int] | None = None, upto: int | None = None) -> list[int]:
    """다년 최적화 학습 해: 관측 저수율이 있는 해 전부 + 장기 기상자료 중 관개기 강수가 가장 적은 n_dry개 해
    (50년을 다 쓰면 계산이 10배 이상 느려지므로, 규칙을 좌우하는 가뭄 해 위주로 추림)"""
    ex = set(exclude or [])
    obs_y = [y for y in years_available(inputs, obs_rate, season) if y not in ex and (upto is None or y <= upto)]
    wy = [y for y in weather_years(inputs, season) if y not in ex and y not in obs_y and (upto is None or y <= upto)]
    rain = {y: float(inputs.loc[season_of(y, season)[0]:season_of(y, season)[1], "rain"].sum()) for y in wy}
    dry = sorted(rain, key=rain.get)[:n_dry]
    return sorted(set(obs_y) | set(dry))


def years_available(inputs: pd.DataFrame, obs_rate: pd.Series, season=SEASON) -> list[int]:
    ys = []
    for y in sorted(set(inputs.index.year)):
        s, e = season_of(y, season)
        if pd.Timestamp(s) >= inputs.index.min() + pd.Timedelta(days=180) and pd.Timestamp(e) <= inputs.index.max() \
                and len(obs_rate.dropna().loc[:pd.Timestamp(s) - pd.Timedelta(days=1)]) \
                and pd.Timestamp(e) <= obs_rate.dropna().index.max():
            ys.append(y)
    return ys


def season_kpis(inputs, cfg, sc: Scenario, years, obs_rate, season=SEASON) -> pd.DataFrame:
    rows = {}
    for y in years:
        s, e = season_of(y, season)
        rows[y] = compute_kpis(_run(inputs.loc[s:e], cfg, sc, _v0(cfg, start_pct(obs_rate, s)), _cap(cfg), 1.0, 1.0), cfg)
    return pd.DataFrame(rows).T


def optimize_multi(inputs, cfg, years, obs_rate, season=SEASON, maxiter: int = 20, seed: int = 0,
                   progress=None, reserve_pct: float = 0.0) -> dict:
    """여러 해 관개기의 평균 SI 최소 규칙 (각 해는 그해 4/1 관측 저수율에서 시작)
    reserve_pct: 비상 저수율 하한 — 최저 저수율이 이보다 내려가면 벌점 (사수위 근처 취수 곤란·이듬해 대비)"""
    segs = []
    typ = typical_start_pct(obs_rate, season)
    for y in years:
        s, e = season_of(y, season)
        segs.append((inputs.loc[s:e], _v0(cfg, start_pct(obs_rate, s, typ) if pd.Timestamp(s) > obs_rate.dropna().index.min()
                                          else typ)))
    cap = _cap(cfg)
    dead = cfg["reservoir"]["dead_storage"]
    eff = cfg["reservoir"]["total_storage"] - dead

    def loss(x):
        sc = decode(x)
        tot = 0.0
        for d, v0 in segs:
            si, mn, _ = quick_si_min(_run(d, cfg, sc, v0, cap, 1.0, 1.0), dead, eff)
            tot += si + 0.05 * max(0.0, reserve_pct - mn) ** 2
        return tot / len(segs)

    it = [0]

    def cb(xk, convergence=None):  # noqa: ARG001
        it[0] += 1
        if progress:
            progress(min(it[0] / maxiter, 1.0), f"다년 최적화 {it[0]}/{maxiter}세대 ({len(years)}개 해)")

    x0 = [60.0, 20.0, 15.0, 0.9, 0.75 / 0.9, 0.55 / 0.75, 0.1]
    r = differential_evolution(loss, BOUNDS, maxiter=maxiter, popsize=10, seed=seed, tol=1e-4, polish=False,
                               callback=cb, x0=x0)
    sc = decode(r.x)
    sc.name = "다년 최적 규칙"
    sc.description = (f"{min(years)}~{max(years)}년 {len(years)}개 해 평균 기준"
                      + (f"·비상 저수율 {reserve_pct:.0f}% 유지" if reserve_pct > 0 else "") + " — " + sc.description)
    return {"scenario": sc, "mean_si": float(r.fun), "years": list(years)}


def replay(inputs, cfg, obs_rate, year: int, rules: dict[str, Scenario], season=SEASON, engine: str = "reference") -> dict:
    """그해 관측 시작 저수율·실제 기상으로 규칙별 모의 + 관측 저수율"""
    from .runner import run_scenario

    s, e = season_of(year, season)
    p0 = start_pct(obs_rate, s)
    runs = {k: run_scenario(inputs, cfg, Scenario.from_dict(v.to_dict()), s, e, p0, engine=engine) for k, v in rules.items()}
    obs = obs_rate.loc[s:e].clip(upper=100)
    tbl = pd.DataFrame({k: {c: r.kpis[c] for c in KPI_COLS} for k, r in runs.items()}).T
    return {"year": year, "start_pct": p0, "runs": runs, "observed": obs, "table": tbl,
            "observed_min_pct": float(obs.min()) if len(obs) else np.nan,
            "observed_min_date": obs.idxmin().strftime("%Y-%m-%d") if len(obs) else ""}


def out_of_sample(inputs, cfg, obs_rate, train_years, test_years, season=SEASON, maxiter: int = 20,
                  progress=None, reserve_pct: float = 0.0) -> dict:
    """train 해로 규칙을 찾아 test 해에 적용 → 관행·기존 규칙과 비교 (해별 SI)"""
    opt = optimize_multi(inputs, cfg, train_years, obs_rate, season, maxiter, progress=progress, reserve_pct=reserve_pct)
    rules = {**presets_for(cfg), "다년 최적 규칙": opt["scenario"]}
    rows = []
    for name, sc in rules.items():
        for part, ys in (("규칙 찾은 해", train_years), ("검증 해", test_years)):
            k = season_kpis(inputs, cfg, Scenario.from_dict(sc.to_dict()), ys, obs_rate, season)
            rows.append({"규칙": name, "구분": part, "해": f"{min(ys)}~{max(ys)}" if len(ys) > 1 else str(ys[0]),
                         "평균 SI": k["shortage_index"].mean(), "최악 해 SI": k["shortage_index"].max(),
                         "평균 최악10일 부족률(%)": k["worst_10day_shortage_pct"].mean(),
                         "평균 공급 만족률(%)": k["volumetric_reliability_pct"].mean(),
                         "평균 최저 저수율(%)": k["min_storage_pct"].mean()})
    return {"scenario": opt["scenario"], "table": pd.DataFrame(rows)}
