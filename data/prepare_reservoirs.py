"""여러 저수지 자료를 한 번에 준비 → 웹앱(클라우드)에서 버튼 없이 바로 모의·예측되도록

  python data/prepare_reservoirs.py                      # 기본: 달창 보문 사곡 수성 오어
  python data/prepare_reservoirs.py 달창 오어 "경주 보문"

저수지마다
  1) 기상청 ASOS 일자료 1973-01-01 ~ 어제   → data/weather_kma_auto_<코드>.csv
  2) 농어촌공사 일별 저수율 2015 ~ 어제      → data/obs_rate_<코드>.csv, observed_storage_<코드>.csv
  3) 저수율이 있으면 GR4J·공급계수·운영상한 보정 (보정 2015~2021 / 검증 2022~) → data/calibrated_params.json
결과 요약: prepare_result.txt
"""
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from engine import catalog, geocode, kma, krc_level  # noqa: E402
from engine.calibrate import calibrate_gr4j, evaluate_params  # noqa: E402
from engine.runner import load_config, load_weather, prepare_inputs  # noqa: E402

DEFAULT = ["경북 달창", "경주 보문", "의성 사곡", "달성 수성", "포항 오어"]
names = sys.argv[1:] or DEFAULT  # 인자 하나가 저수지 하나 (예: 달창 오어 "경주 보문")

if not hasattr(catalog, "spec_issues"):
    sys.exit("engine/catalog.py가 예전 버전입니다. 제원보정_업데이트.zip을 먼저 이 폴더에 덮어쓰세요 (달창 저수량 보정 필요).")

lines = []


def log(msg: str) -> None:
    print(msg, flush=True)
    lines.append(msg)


for q in names:
    t0 = time.time()
    try:
        row = catalog.search(q).iloc[0]
        code = str(row["표준코드"])
        cfg = catalog.config_for(code, load_config())
        r = cfg["reservoir"]
        log(f"\n=== {row['시설명']} ({row['본부']}·{row['지사']}, {code}) 총저수량 {r['total_storage']:,.0f}천㎥ ===")
        if str(row.get("_보정", "") or ""):
            log(f"  제원 보정 적용: {row['_보정']}")
        c = geocode.get_coords(code, row["시설명"], row["소재지"])
        log(f"  좌표 {c['lat']:.4f}, {c['lon']:.4f} ({c['source']})")

        p, wi = kma.update_weather(code, c["lat"], c["lon"], start="1973-01-01",
                                   progress=lambda f, t: print("   ", t, flush=True))
        w = load_weather(p)
        log(f"  [기상] {wi['first']} ~ {wi['last']} ({len(w):,}일) · 관측소 {', '.join(s['name'] for s in wi['stations'])}")

        try:
            obs, oi = krc_level.update(code, cfg)
            log(f"  [저수율] {oi['first']} ~ {oi['last']} · 최신 {obs['rate'].dropna().iloc[-1]:.1f}%")
        except LookupError as e:
            log(f"  [저수율] 없음 — {e}")

        so_p = ROOT / "data" / f"observed_storage_{code}.csv"
        if krc_level.obs_path(code).exists() and so_p.exists():
            so = pd.read_csv(so_p, parse_dates=["date"], index_col="date")
            inp0 = prepare_inputs(w, cfg, {k: cfg["gr4j"][k] for k in ("x1", "x2", "x3", "x4")})
            so["release"] = (inp0["demand"] + cfg["environment"]["river_maintenance_flow"]).reindex(so.index)
            so = so.loc[so.index.intersection(w.index)].dropna()
            if len(so.loc[:"2021-12-31"]) > 365 and len(so.loc["2022-01-01":]) > 180:
                rc = calibrate_gr4j(inp0.loc[:"2021-12-31"], so, cfg, maxiter=20, fit_release_scale=True)
                rv = evaluate_params(inp0.loc["2022-01-01":], so, cfg, rc["params"])
                catalog.save_calibration(code, rc["params"], {"nse_cal": round(rc["nse"], 3), "nse_val": round(rv["nse"], 3)})
                log(f"  [보정] NSE 보정 {rc['nse']:.3f} / 검증 {rv['nse']:.3f}")
            else:
                log("  [보정] 저수율 기간이 짧아 건너뜀")
        log(f"  완료 ({time.time() - t0:.0f}초)")
    except Exception as e:  # 한 저수지가 실패해도 나머지는 계속
        log(f"\n=== '{q}' 실패: {e}")
        traceback.print_exc()

log("\nGitHub에 올릴 파일: data/weather_kma_auto_*.csv, data/obs_rate_*.csv, data/observed_storage_*.csv, "
    "data/calibrated_params.json (새로 생기거나 바뀐 것)")
(ROOT / "prepare_result.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n결과를 prepare_result.txt에 저장했습니다.")
