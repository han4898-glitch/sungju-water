"""모의 실행 진입점: 기상 → 유입량(GR4J) → 수요량 → 저수지 운영(pywr) → KPI"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .demand import compute_et0, paddy_irrigation_demand
from .gr4j import run_gr4j, runoff_to_volume
from .kpi import compute_kpis
from .operation import Scenario, forecast_factor, normal_series
from .pywr_engine import PYWR_AVAILABLE, run_pywr
from .reference_engine import run_reference

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config" / "sungju.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_weather(path: str | Path = ROOT / "data" / "weather_sample.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").asfreq("D")
    if "rain" in df:
        df["rain"] = df["rain"].fillna(0.0)
    return df.interpolate(limit_direction="both")  # 관측 결측 보간 (ET0·유출 계산이 NaN이 되지 않도록)


@dataclass
class RunResult:
    scenario: Scenario
    engine: str
    series: pd.DataFrame
    kpis: dict


def prepare_inputs(weather: pd.DataFrame, cfg: dict, gr4j_params: dict | None = None) -> pd.DataFrame:
    """엔진에 넣을 일별 입력(유입량·필요수량·증발량)을 만듭니다."""
    st, res = cfg["station"], cfg["reservoir"]
    p = gr4j_params or cfg["gr4j"]
    et0 = compute_et0(weather, st["lat"], st["elevation_m"])
    q_mm = run_gr4j(weather["rain"].values, et0.values, p["x1"], p["x2"], p["x3"], p["x4"])
    inflow = pd.Series(runoff_to_volume(q_mm, cfg["catchment"]["area_km2"]), index=weather.index)
    dem = paddy_irrigation_demand(weather, et0, cfg["irrigation"])
    # 실측 보정 공급계수: 이론 필요수량 대비 실제 공급 비율 (실관개면적 감소·회귀수 등 반영)
    dem["demand"] = dem["demand"] * float(p.get("release_scale", 1.0))
    # 수면증발 (천㎥/day) = 계수 × ET0(mm) × 수면적(km²)
    evap = res["evap_coeff"] * et0 * res["surface_area_km2"]
    return pd.DataFrame(
        {"rain": weather["rain"], "et0": et0, "inflow": inflow, "demand": dem["demand"],
         "ponding_mm": dem["ponding_mm"], "evap": evap}
    )


def run_scenario(
    inputs: pd.DataFrame,
    cfg: dict,
    scenario: Scenario,
    start: str | None = None,
    end: str | None = None,
    initial_pct: float = 70.0,
    demand_scale: float = 1.0,
    inflow_scale: float = 1.0,
    engine: str = "auto",
) -> RunResult:
    """한 시나리오를 실행합니다.

    initial_pct  : 시작 시점 유효저수율(%)
    demand_scale : 필요수량 배율 (재배면적·작부 변화 시나리오)
    inflow_scale : 유입량 배율 (기후변화·가뭄 강도 시나리오)
    engine       : "pywr" | "reference" | "auto"(pywr 있으면 pywr)
    """
    scenario.validate()
    df = inputs.loc[start:end]
    res = cfg["reservoir"]
    total, dead = res["total_storage"], res["dead_storage"]
    cap_frac = float(cfg.get("operation", {}).get("ops_cap_frac", 1.0))
    cap = dead + (total - dead) * cap_frac
    v0 = min(dead + (total - dead) * initial_pct / 100, cap)
    fc = forecast_factor(df["rain"], scenario)
    args = dict(
        inflow=df["inflow"] * inflow_scale,
        demand=df["demand"] * demand_scale,
        evap=df["evap"],
        fc_factor=fc,
        sc=scenario,
        total_storage=total,
        dead_storage=dead,
        initial_storage=v0,
        env_flow=cfg["environment"]["river_maintenance_flow"],
        cap_storage=cap,
        normal_pct=normal_series(cfg, df.index),
    )
    use = engine
    if engine == "auto":
        use = "pywr" if PYWR_AVAILABLE else "reference"
    out = run_pywr(**args) if use == "pywr" else run_reference(**args)
    out["storage_pct"] = (out["storage"] - dead) / (total - dead) * 100
    out["deficit"] = (out["demand"] - out["supply"]).clip(lower=0)
    out["rain"] = df["rain"]
    return RunResult(scenario, use, out, compute_kpis(out, cfg))
