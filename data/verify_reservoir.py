"""다른 저수지에서도 전 과정이 되는지 점검 → verify_result.txt

  python data/verify_reservoir.py 영덕 묘곡
순서: 카탈로그 검색 → 제원→모델 설정 → 좌표 → 기상청 ASOS 관측 → 농어촌공사 일별 저수율
      → GR4J 유입량·필요수량 → 3개 시나리오(pywr) → 운영규칙 최적화 (가뭄 조건 포함)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from engine import catalog, geocode, kma, krc_level  # noqa: E402
from engine.operation import PRESETS  # noqa: E402
from engine.optimize import optimize_rules  # noqa: E402
from engine.calibrate import calibrate_gr4j, evaluate_params  # noqa: E402
from engine.runner import load_config, load_weather, prepare_inputs, run_scenario  # noqa: E402

q = " ".join(sys.argv[1:]) or "영덕 묘곡"
row = catalog.search(q).iloc[0]
code = row["표준코드"]
cfg = catalog.config_for(code, load_config())
r = cfg["reservoir"]
print(f"[1] 검색 '{q}' → {row['시설명']} ({code}) {row['소재지']}")
print(f"    총저수량 {r['total_storage']:,.0f}천㎥ · 유역 {cfg['catchment']['area_km2']:.1f}km² · "
      f"수혜면적 {cfg['irrigation']['area_ha']:,.0f}ha" + (" (추정)" if cfg['irrigation'].get('area_estimated') else ""))
c = geocode.get_coords(code, row["시설명"], row["소재지"])
r["lat"], r["lon"] = c["lat"], c["lon"]
print(f"[2] 좌표 {c['lat']:.4f}, {c['lon']:.4f} ({c['source']})")
t = time.time()
wpath, wi = kma.update_weather(code, r["lat"], r["lon"])
print(f"[3] 기상청 ASOS {', '.join(s['name'] for s in wi['stations'])} → {wpath.name} (~{wi['last']}, {time.time()-t:.0f}초)")
try:
    obs, oi = krc_level.update(code, cfg)
    print(f"[4] 일별 저수율 {oi['first']}~{oi['last']} · 최신 {obs['rate'].dropna().iloc[-1]:.1f}%")
    cur = float(obs["rate"].dropna().iloc[-1])
except LookupError as e:
    print(f"[4] {e}")
    cur = 60.0
w = load_weather(wpath)
yr = w.index.max().year - 1
s_, e_ = f"{yr}-03-01", f"{yr}-10-31"
obs_s = krc_level.obs_path(code)
if obs_s.exists():  # 관측 저수율로 GR4J·공급계수·운영상한 보정 (보정 2015~2021, 검증 2022~)
    so = pd.read_csv(ROOT / "data" / f"observed_storage_{code}.csv", parse_dates=["date"], index_col="date")
    base_g = {k: cfg["gr4j"][k] for k in ("x1", "x2", "x3", "x4")}
    inp0 = prepare_inputs(w, cfg, base_g)
    so["release"] = (inp0["demand"] + cfg["environment"]["river_maintenance_flow"]).reindex(so.index)
    so = so.loc[so.index.intersection(w.index)].dropna()
    t = time.time()
    rc = calibrate_gr4j(inp0.loc[:"2021-12-31"], so, cfg, maxiter=20, fit_release_scale=True)
    rv = evaluate_params(inp0.loc["2022-01-01":], so, cfg, rc["params"])
    catalog.save_calibration(code, rc["params"], {"nse_cal": round(rc["nse"], 3), "nse_val": round(rv["nse"], 3)})
    cfg = catalog.config_for(code, load_config())
    print(f"[5] 보정 NSE {rc['nse']:.3f} / 검증 NSE {rv['nse']:.3f} ({time.time()-t:.0f}초) → "
          + ", ".join(f"{k}={v:.3g}" for k, v in rc["params"].items() if k != "cap_flood_frac"))
    ob = pd.read_csv(obs_s, parse_dates=["date"], index_col="date")["rate"].dropna()
    cur = float(ob.loc[:s_].iloc[-1])
cfg["station"]["lat"] = r["lat"]
cfg["reservoir"]["lat"], cfg["reservoir"]["lon"] = r["lat"], r["lon"]
inp = prepare_inputs(w, cfg, cfg["gr4j"])
print(f"[6] 유입량 {inp.loc[s_:e_, 'inflow'].sum():,.0f}천㎥ · 필요수량 {inp.loc[s_:e_, 'demand'].sum():,.0f}천㎥ ({s_}~{e_})")
for label, kw in [("실제 기상", dict(initial_pct=cur, inflow_scale=1.0, demand_scale=1.0)),
                  ("가뭄(유입×0.5, 수요×1.2)", dict(initial_pct=min(cur, 50), inflow_scale=0.5, demand_scale=1.2))]:
    print(f"[7] 시나리오 비교 — {label}, 시작 저수율 {kw['initial_pct']:.0f}%")
    rows = {}
    for k, p in PRESETS.items():
        rows[k] = run_scenario(inp, cfg, p, s_, e_, kw["initial_pct"], kw["demand_scale"], kw["inflow_scale"]).kpis
    o = optimize_rules(inp, cfg, s_, e_, **kw)
    rows["최적 운영규칙"] = o["kpis"]
    tb = pd.DataFrame(rows).T[["shortage_index", "volumetric_reliability_pct", "worst_10day_shortage_pct",
                               "min_storage_pct", "end_storage_pct"]].astype(float).round(1)
    tb.columns = ["SI", "공급만족률%", "최악10일부족%", "최저저수율%", "기말저수율%"]
    print(tb.to_string())
    print(f"    최적 규칙: {o['scenario'].description}")
print("완료 — 이 저수지도 성주와 같은 과정으로 방류(공급)량 최적화가 됩니다.")
