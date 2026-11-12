"""농어촌공사 저수지 용수관리 MCP 서버 (기본: 성주댐) (선택 가점 항목)

Claude 같은 AI 에이전트가 자연어로 모델 조건을 바꾸고, 실행하고, 결과를 조회하게 합니다.
  예) "경북 성주 저수지 지금 저수율 알려줘", "성주랑 경천 제원 비교해줘",
      "저수율 45%에서 다음 10일 비가 10mm 온다면 이번 주 공급량을 얼마나 줄여야 해?"

실행(로컬, stdio):   python mcp_server.py
Claude Desktop 설정 예 (claude_desktop_config.json):
  "mcpServers": {"krc-water": {"command": "python", "args": ["<경로>/mcp_server.py"]}}
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # mcp 2.x 에서 이름이 바뀜
    from mcp.server.mcpserver import MCPServer as FastMCP

import pandas as pd

from pathlib import Path

from engine import PRESETS, Scenario, catalog, presets_for, geocode, krc_api, load_config, load_weather, prepare_inputs, run_scenario

ROOT = Path(__file__).resolve().parent

mcp = FastMCP("krc-reservoir-water")
CFG = load_config()
_INPUTS: dict[str, pd.DataFrame] = {}


def _resolve(reservoir: str | None) -> tuple[str, dict]:
    """저수지 이름·지역 검색어(또는 표준코드) → (코드, 모델 설정)"""
    if not reservoir:
        code = CFG["reservoir"]["code"]
    elif reservoir.isdigit():
        code = reservoir
    else:
        hit = catalog.search(reservoir, limit=1)
        if hit.empty:
            raise ValueError(f"'{reservoir}'에 맞는 저수지가 없습니다. find_reservoir로 먼저 찾아보세요.")
        code = hit.iloc[0]["표준코드"]
    cfg = catalog.config_for(code, CFG)
    from engine import krc_level
    from engine.operation import attach_normal_curve

    if krc_level.obs_path(code).exists():  # 평년 저수율 곡선 → 평년 대비(공사 가뭄단계) 규칙 사용 가능
        attach_normal_curve(cfg, pd.read_csv(krc_level.obs_path(code), parse_dates=["date"], index_col="date")["rate"])
    return code, cfg


def _weather(code: str) -> pd.DataFrame:
    """그 저수지의 기상청 관측자료(weather_kma_auto_<코드>.csv, 1973~)가 있으면 그것, 없으면 예시 자료"""
    p = ROOT / "data" / f"weather_kma_auto_{code}.csv"
    return load_weather(p) if p.exists() else load_weather()


def _inputs(code: str, cfg: dict) -> pd.DataFrame:
    if code not in _INPUTS:
        _INPUTS[code] = prepare_inputs(_weather(code), cfg, cfg["gr4j"])
    return _INPUTS[code]


def _round(d: dict) -> dict:
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


@mcp.tool()
def find_reservoir(query: str, limit: int = 5) -> list[dict]:
    """한국농어촌공사 저수지 3,402개소에서 이름·지역으로 찾습니다. 예: "경북 성주", "문경 경천", "경산 저수지"."""
    hits = catalog.search(query, limit)
    return [{"표준코드": r["표준코드"], "시설명": r["시설명"], "본부": r["본부"], "지사": r["지사"],
             "소재지": r["소재지"], "총저수량_천m3": float(r["총저수량"])} for _, r in hits.iterrows()]


@mcp.tool()
def get_reservoir_specs(reservoir: str) -> dict:
    """저수지 제원(유역·수혜·만수면적, 총/유효저수량, 수위, 제체 등)을 돌려줍니다. reservoir = 이름 검색어 또는 표준코드."""
    code, _ = _resolve(reservoir)
    row = catalog.get(code)
    d = catalog.summary(row)
    if row["수혜면적"]:
        d["유역배율"] = round(float(row["유역면적"] / row["수혜면적"]), 2)
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}


@mcp.tool()
def compare_reservoirs(reservoirs: list[str]) -> list[dict]:
    """여러 저수지의 주요 제원을 나란히 비교합니다. 예: ["성주", "문경 경천", "영천"]"""
    return [get_reservoir_specs(r) for r in reservoirs]


@mcp.tool()
def get_reservoir_status(reservoir: str, use_cache: bool = True) -> dict:
    """저수지 수위 API(재난안전데이터 공유플랫폼, 원천 농어촌공사)로 최신 저수율·수위를 조회합니다.
    인증키(SAFETYDATA_KEY)와 config [api] 데이터셋 ID가 필요합니다. 실패하면 마지막으로 저장된 자료를 돌려줍니다."""
    code, cfg = _resolve(reservoir)
    api = CFG.get("api", {})
    cache = krc_api.cache_path(code, "raw")
    note = None
    try:
        extra = {api["code_param"]: code} if api.get("code_param") else None
        df = krc_api.normalize(krc_api.fetch_pages(api, api.get("dataset_10min") or api.get("dataset_daily"), extra, 5),
                               api.get("fields"))
        sub = krc_api.filter_reservoir(df, code, cfg["reservoir"]["name"])
        if not sub.empty:
            sub.to_csv(cache, index=False)
    except Exception as e:
        note = f"API 호출 실패: {e}"
        sub = pd.read_csv(cache, parse_dates=["time"]) if (use_cache and cache.exists()) else pd.DataFrame()
    if sub.empty:
        return {"reservoir": cfg["reservoir"]["name"], "code": code, "error": note or "해당 저수지 자료 없음"}
    last = sub.iloc[-1].to_dict()
    last["time"] = str(last.get("time"))
    out = {"reservoir": cfg["reservoir"]["name"], "code": code, "latest": last, "records": len(sub)}
    if "rate" in sub:
        day = sub.set_index("time")["rate"].resample("D").last().dropna().tail(30)
        out["rate_last_30d"] = {d.strftime("%m-%d"): round(v, 1) for d, v in day.items()}
    if note:
        out["note"] = note + " (저장된 자료로 응답)"
    return out


@mcp.tool()
def get_reservoir_info(reservoir: str = "성주") -> dict:
    """모델에 쓰는 저수지 설정(제원, 수혜면적, GR4J 매개변수)과 기상자료 기간을 돌려줍니다."""
    code, cfg = _resolve(reservoir)
    w = _inputs(code, cfg)
    return {"reservoir": cfg["reservoir"], "catchment_km2": cfg["catchment"]["area_km2"],
            "irrigation_area_ha": cfg["irrigation"]["area_ha"], "gr4j": cfg["gr4j"],
            "data_period": [str(w.index.min().date()), str(w.index.max().date())]}


@mcp.tool()
def list_scenarios() -> dict:
    """기본 운영 시나리오(관행/단계별 제한급수/예보 연계)의 규칙을 돌려줍니다."""
    return {k: v.to_dict() for k, v in PRESETS.items()}


@mcp.tool()
def run_operation(
    reservoir: str = "성주",
    scenario: str = "단계별 제한급수",
    start: str = "2025-10-01",
    end: str = "2026-10-31",
    initial_pct: float = 70.0,
    inflow_scale: float = 1.0,
    demand_scale: float = 1.0,
    curves_pct: list[float] | None = None,
    supply_factors: list[float] | None = None,
) -> dict:
    """저수지 운영 모델을 실행하고 KPI와 월별 저수율을 돌려줍니다. reservoir = 이름 검색어 또는 표준코드.
    curves_pct/supply_factors를 주면 해당 시나리오 규칙을 덮어씁니다(사용자 정의 규칙)."""
    code, cfg = _resolve(reservoir)
    sc = Scenario.from_dict(PRESETS[scenario].to_dict())
    if curves_pct is not None and supply_factors is not None:
        sc.curves_pct, sc.supply_factors, sc.name = curves_pct, supply_factors, "사용자 정의"
    r = run_scenario(_inputs(code, cfg), cfg, sc, start, end, initial_pct, demand_scale, inflow_scale)
    monthly = r.series["storage_pct"].resample("MS").last().round(1)
    return {"reservoir": cfg["reservoir"]["name"], "engine": r.engine, "scenario": sc.to_dict(), "kpis": _round(r.kpis),
            "monthly_end_storage_pct": {d.strftime("%Y-%m"): v for d, v in monthly.items()}}


@mcp.tool()
def compare_scenarios(
    reservoir: str = "성주", start: str = "2025-10-01", end: str = "2026-10-31", initial_pct: float = 70.0,
    inflow_scale: float = 1.0, demand_scale: float = 1.0,
) -> dict:
    """기본 운영규칙(관행·단계별 제한급수·예보 연계·공사 가뭄단계 연동)을 같은 조건에서 실행해 KPI를 비교합니다."""
    code, cfg = _resolve(reservoir)
    out = {}
    for k, p in presets_for(cfg).items():
        r = run_scenario(_inputs(code, cfg), cfg, Scenario.from_dict(p.to_dict()), start, end,
                         initial_pct, demand_scale, inflow_scale)
        out[k] = _round(r.kpis)
    return out


@mcp.tool()
def optimize_operation(
    reservoir: str = "성주", start: str = "2025-03-01", end: str = "2025-10-31", initial_pct: float = 70.0,
    inflow_scale: float = 1.0, demand_scale: float = 1.0, end_target_pct: float = 0.0, use_forecast: bool = True,
) -> dict:
    """어느 저수지든 방류(공급) 운영규칙을 최적화합니다: 저수율 경계 3개·구간별 공급 비율(+예보 연계 절감)을
    부족 심도지수 SI가 최소가 되도록 찾고, 관행 운영과 KPI를 비교합니다. reservoir = 이름 검색어 또는 표준코드."""
    from engine.optimize import optimize_rules

    code, cfg = _resolve(reservoir)
    r = optimize_rules(_inputs(code, cfg), cfg, start, end, initial_pct, demand_scale, inflow_scale,
                       end_target_pct=end_target_pct, use_forecast=use_forecast)
    return {"reservoir": cfg["reservoir"]["name"], "code": code, "optimal_rule": r["scenario"].to_dict(),
            "kpis_optimal": _round(r["kpis"]), "kpis_conventional": _round(r["baseline_kpis"]),
            "evaluations": r["evaluations"]}


@mcp.tool()
def esp_outlook(reservoir: str = "성주", current_pct: float | None = None, days: int = 150,
                outlook_below: float = 33, outlook_normal: float = 34, outlook_above: float = 33,
                scenario: str = "단계별 제한급수") -> dict:
    """기상청 과거 관측(2015~)으로 만든 앙상블(ESP)로 앞으로의 저수율 확률 전망과 관리단계 진입 확률을 계산합니다.
    outlook_* = 기상청 1·3개월 강수 전망 확률(적음/비슷/많음, %). current_pct 없으면 농어촌공사 최신 관측 저수율."""
    from engine import esp, krc_level

    code, cfg = _resolve(reservoir)
    wf = ROOT / "data" / f"weather_kma_auto_{code}.csv"
    hist = load_weather(wf if wf.exists() else ROOT / "data" / "weather_sample.csv")
    if current_pct is None:
        p = krc_level.obs_path(code)
        current_pct = float(pd.read_csv(p)["rate"].dropna().iloc[-1]) if p.exists() else 50.0
    today = pd.Timestamp.today().normalize()
    members = esp.build_members(hist, today, days)
    probs = None if (outlook_below, outlook_normal, outlook_above) == (33, 34, 33) else (outlook_below, outlook_normal, outlook_above)
    w = esp.tercile_weights(members, today, days, probs)
    prep = esp.prepare_members(members, cfg, cfg["gr4j"])
    errs = None
    if krc_level.obs_path(code).exists():  # 과거 재예측 오차로 관측 보정
        ob = pd.read_csv(krc_level.obs_path(code), parse_dates=["date"], index_col="date")["rate"]
        errs = esp.hindcast_errors(prepare_inputs(hist, cfg, cfg["gr4j"]), cfg, ob, days)
    r = esp.run_esp(prep, cfg, Scenario.from_dict(PRESETS[scenario].to_dict()), today, days, current_pct, w, errs)
    monthly = r["band"].resample("MS").last().round(1)
    return {"reservoir": cfg["reservoir"]["name"], "current_pct": current_pct, "members": list(members),
            "weather_source": wf.name if wf.exists() else "weather_sample.csv (가상)",
            "monthly_storage_pct": {d.strftime("%Y-%m"): row.to_dict() for d, row in monthly.iterrows()},
            "risk_pct": {k: round(v, 1) for k, v in r["risk"].items()},
            "expected": _round(r["expected"]),
            "error_corrected": r["corrected"],
            "rule_comparison": esp.compare_rules(prep, cfg, today, days, current_pct, w, errors=errs)[["shortage_index"]].round(2)
            .to_dict()["shortage_index"]}


@mcp.tool()
def drought_stage_now(reservoir: str = "성주") -> dict:
    """농어촌공사 RAWRIS 실시간 저수율·평년 대비와 농업가뭄 단계(정상/관심/주의/경계/심각)를 알려줍니다."""
    from engine import rawris

    code, cfg = _resolve(reservoir)
    df, src = rawris.snapshot()
    r = rawris.lookup(code, df)
    if not r:
        return {"reservoir": cfg["reservoir"]["name"], "code": code, "message": "RAWRIS 실시간 계측 대상이 아닙니다."}
    return {"reservoir": cfg["reservoir"]["name"], "code": code, "time": str(r["시각"]), "storage_pct": r["저수율"],
            "normal_pct": r["평년"], "pct_of_normal": r["평년대비"], "stage": r["단계"], "source": src,
            "stage_rule": "평년 대비 70% 초과 정상, 60~70 관심, 50~60 주의, 40~50 경계, 40 이하 심각 (ADMS 기준)"}


@mcp.tool()
def drought_supply_plan(reservoir: str = "성주", days: int | None = None, outlook_below: float = 33,
                        outlook_normal: float = 34, outlook_above: float = 33, reserve_pct: float = 20,
                        optimize: bool = True) -> dict:
    """[한 번에] 가뭄을 예측해 공급량을 설정합니다. 예) "성주댐 가뭄 예측해서 공급량 설정해줘"
    현재 저수율(RAWRIS 실시간) → 기상청 단기예보 + 1973년~ 과거 기상 앙상블로 저수율 확률 전망(관측 오차 보정)
    → 기본 규칙·앙상블 최적 규칙 비교 → 권장 규칙, 오늘 공급 비율, 월별 공급계획(천㎥).
    days 없으면 다음 관개기 끝(9/30)까지. outlook_* = 기상청 1·3개월 강수 전망(적음/비슷/많음 %).
    reserve_pct = 지키고 싶은 비상 유효저수율(%). 1~2분 걸릴 수 있습니다."""
    from engine import kma, krc_level, plan, rawris

    code, cfg = _resolve(reservoir)
    weather = _weather(code)
    obs = None
    if krc_level.obs_path(code).exists():
        obs = pd.read_csv(krc_level.obs_path(code), parse_dates=["date"], index_col="date")["rate"]
    raw = None
    try:
        raw = rawris.lookup(code, rawris.snapshot()[0])
    except Exception:  # noqa: BLE001
        pass
    fc = None
    try:
        c = geocode.get_coords(code, cfg["reservoir"]["name"], cfg["reservoir"]["location"])
        if c:
            fc, _ = kma.fetch_forecast(c["lat"], c["lon"])
    except Exception:  # noqa: BLE001
        pass
    probs = None if (outlook_below, outlook_normal, outlook_above) == (33, 34, 33) else (outlook_below, outlook_normal, outlook_above)
    p = plan.make_plan(code, cfg, weather, _inputs(code, cfg), cfg["gr4j"], obs, raw, fc, days, probs, reserve_pct, optimize)
    m = p["monthly"].round(1)
    return {"reservoir": cfg["reservoir"]["name"], "code": code, "summary": p["summary"],
            "current": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in p["now"].items()},
            "recommended_rule": p["rule_name"], "rule": p["rule"].to_dict(), "today_supply_factor": round(p["today_factor"], 3),
            "risk_pct": {k: round(v, 1) for k, v in p["risk"].items()},
            "monthly_plan": {i: {k: (None if pd.isna(v) else v) for k, v in r.items()} for i, r in m.iterrows()},
            "rule_comparison": p["compare"][["shortage_index", "worst_10day_shortage_pct", "min_storage_pct"]]
            .astype(float).round(2).to_dict(orient="index"),
            "ensemble_years": f"{min(p['members'])}~{max(p['members'])} ({len(p['members'])}개 해)",
            "error_corrected": p["corrected"], "horizon_days": p["days"]}


@mcp.tool()
def supply_factor_now(current_pct: float, forecast_rain_mm: float, scenario: str = "예보 연계 운영") -> dict:
    """현재 유효저수율과 예보 강수 합계로 오늘 적용할 공급 비율(필요수량 대비)을 계산합니다."""
    from engine.operation import band_factor

    sc = PRESETS[scenario]
    f = band_factor(current_pct, sc)
    if sc.forecast_days and forecast_rain_mm < sc.forecast_threshold_mm:
        f *= 1 - sc.forecast_cut
    stage = "정상"
    names = ["관심", "주의", "경계", "심각"]
    for i, c in enumerate(sc.curves_pct):
        if current_pct < c:
            stage = names[i] if i < 4 else stage
    return {"supply_factor": round(f, 3), "stage": stage, "rule": sc.to_dict()}


@mcp.tool()
def get_weather_forecast(reservoir: str = "성주") -> dict:
    """저수지 위치의 기상청 단기예보(약 3일)를 일 단위로 돌려줍니다 (강수 mm, 최고·최저기온).
    환경변수 DATA_GO_KR_KEY(공공데이터포털 인증키) 필요."""
    from engine import kma

    code, cfg = _resolve(reservoir)
    c = geocode.get_coords(code, cfg["reservoir"]["name"], cfg["reservoir"]["location"])
    if not c:
        return {"error": "저수지 좌표를 찾지 못했습니다."}
    f, info = kma.fetch_forecast(c["lat"], c["lon"])
    return {"reservoir": cfg["reservoir"]["name"], **info, "rain_total_mm": round(float(f["rain"].sum()), 1),
            "daily": {d.strftime("%Y-%m-%d"): {k: (None if pd.isna(v) else round(float(v), 1)) for k, v in r.items()}
                      for d, r in f.iterrows()}}


@mcp.tool()
def project_operation(reservoir: str = "성주", current_pct: float = 50.0, days: int = 30,
                      assumption: str = "무강우", use_forecast: bool = True) -> dict:
    """현재 유효저수율에서 출발해 기상청 예보(+이후 무강우/평년 가정)로 향후 N일을 시나리오별로 전망하고,
    오늘 적용할 공급비율을 돌려줍니다. 예) "성주 저수율 45%인데 앞으로 한 달 버틸 수 있어?" """
    from datetime import date

    from engine import kma
    from engine.realtime import project, today_decision

    code, cfg = _resolve(reservoir)
    c = geocode.get_coords(code, cfg["reservoir"]["name"], cfg["reservoir"]["location"])
    hist_path = ROOT / "data" / f"weather_kma_auto_{code}.csv"
    hist = load_weather(hist_path) if hist_path.exists() else load_weather()
    fc = None
    if use_forecast and c and kma.get_key():
        fc, _ = kma.fetch_forecast(c["lat"], c["lon"])
        fc = fc.loc[pd.Timestamp(date.today()):]
    pr = project(hist, fc, cfg, current_pct, days, assumption)
    rain = float(fc["rain"].sum()) if fc is not None and len(fc) else 0.0
    dem = float(pr["inputs"]["demand"].iloc[0])
    return {
        "reservoir": cfg["reservoir"]["name"], "weather_history": hist_path.name if hist_path.exists() else "sample",
        "forecast_rain_mm": round(rain, 1), "demand_today_k_m3": round(dem, 1),
        "scenarios": {k: {**{kk: today_decision(current_pct, rain, dem, r.scenario)[kk] for kk in ("factor", "stage")},
                          "end_storage_pct": round(r.kpis["end_storage_pct"], 1),
                          "min_storage_pct": round(r.kpis["min_storage_pct"], 1),
                          "deficit_k_m3": round(r.kpis["total_deficit_k_m3"], 1)}
                      for k, r in pr["runs"].items()},
    }


if __name__ == "__main__":
    mcp.run()
