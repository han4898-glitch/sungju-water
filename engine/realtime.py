"""실시간 운영 전망: 현재 저수율 + 기상청 단기예보 → 향후 N일 시나리오별 저수율·공급 전망

입력 기상 = [관측 이력(GR4J 초기화용)] + [단기예보 약 3일] + [그 이후 가정]
  그 이후 가정:  "무강우"  — 비가 오지 않는다고 보는 보수적 가정 (가뭄 대응 판단용)
                 "평년"    — 관측 이력에서 같은 날짜의 평균(기후값)
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .operation import PRESETS, Scenario, band_factor, presets_for
from .runner import prepare_inputs, run_scenario

COLS = ["rain", "tmax", "tmin", "rh", "wind", "sunshine"]


def climatology(hist: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    h = hist[[c for c in COLS if c in hist]].copy()
    h["doy"] = h.index.dayofyear
    clim = h.groupby("doy").mean(numeric_only=True)
    clim = clim.reindex(range(1, 367)).interpolate(limit_direction="both")
    out = clim.loc[np.minimum(dates.dayofyear, 366)].copy()
    out.index = dates
    return out


def build_weather(hist: pd.DataFrame, forecast: pd.DataFrame | None, start: pd.Timestamp,
                  days: int, assumption: str = "무강우") -> tuple[pd.DataFrame, pd.Series]:
    """모의용 기상 시계열과 각 날짜의 자료 출처(관측/예보/가정)"""
    end = start + pd.Timedelta(days=days - 1)
    hist = hist.loc[: start - pd.Timedelta(days=1)]
    warm_start = start - pd.Timedelta(days=730)
    full_idx = pd.date_range(min(warm_start, hist.index.min()) if len(hist) else warm_start, end, freq="D")
    clim = climatology(hist if len(hist) > 365 else hist, full_idx) if len(hist) else None
    w = pd.DataFrame(index=full_idx, columns=COLS, dtype=float)
    src = pd.Series("가정", index=full_idx)
    if len(hist):
        w.loc[hist.index.intersection(full_idx)] = hist.reindex(columns=COLS).loc[hist.index.intersection(full_idx)]
        src.loc[hist.index.intersection(full_idx)] = "관측"
    # 관측 이력과 오늘 사이 빈 날짜 → 기후값
    gap = w.index[(w.index < start) & w["rain"].isna()]
    if len(gap) and clim is not None:
        w.loc[gap] = clim.loc[gap, COLS].values
        src.loc[gap] = "기후값"
    future = w.index[w.index >= start]
    if clim is not None:
        w.loc[future] = clim.loc[future, COLS].values
    if assumption == "무강우":
        w.loc[future, "rain"] = 0.0
    if forecast is not None and len(forecast):
        f = forecast.reindex(future).dropna(how="all")
        for c in f.columns:
            if c in w:
                m = f[c].notna()
                w.loc[f.index[m], c] = f.loc[m, c]
        src.loc[f.index] = "예보"
    w = w.astype(float).interpolate(limit_direction="both")
    return w, src


def project(hist: pd.DataFrame, forecast: pd.DataFrame | None, cfg: dict, current_pct: float,
            days: int = 30, assumption: str = "무강우", start: date | None = None,
            scenarios: dict[str, Scenario] | None = None, gr4j_params: dict | None = None,
            engine: str = "auto") -> dict:
    start_ts = pd.Timestamp(start or date.today())
    w, src = build_weather(hist, forecast, start_ts, days, assumption)
    inputs = prepare_inputs(w, cfg, gr4j_params)
    s, e = start_ts.strftime("%Y-%m-%d"), (start_ts + pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")
    out = {}
    for name, sc in (scenarios or presets_for(cfg)).items():
        sc = Scenario.from_dict(sc.to_dict())
        out[name] = run_scenario(inputs, cfg, sc, s, e, current_pct, engine=engine)
    return {"runs": out, "weather": w.loc[s:e], "source": src.loc[s:e], "inputs": inputs.loc[s:e]}


def today_decision(current_pct: float, forecast_rain_mm: float, demand_today: float,
                   scenario: Scenario, normal_pct: float | None = None) -> dict:
    """오늘 적용할 공급 비율과 관리 단계 (평년 대비 규칙이면 normal_pct = 오늘의 평년 저수율)"""
    if scenario.basis == "normal" and not (normal_pct and normal_pct > 0):
        return {"factor": float("nan"), "stage": "-", "forecast_cut": False, "supply_k_m3": float("nan")}
    f = band_factor(current_pct, scenario, normal_pct)
    cut = bool(scenario.forecast_days and forecast_rain_mm < scenario.forecast_threshold_mm)
    if cut:
        f *= 1 - scenario.forecast_cut
    x = current_pct / normal_pct * 100 if scenario.basis == "normal" else current_pct
    names = ["관심", "주의", "경계", "심각"]
    stage = "정상"
    for i, c in enumerate(scenario.curves_pct):
        if x < c:
            stage = names[i] if i < len(names) else stage
    return {"factor": f, "stage": stage, "forecast_cut": cut, "supply_k_m3": demand_today * f}
