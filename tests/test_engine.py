"""엔진 검증 테스트:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import PRESETS, Scenario, load_config, load_weather, prepare_inputs, run_scenario  # noqa: E402
from engine.gr4j import run_gr4j  # noqa: E402
from engine.pywr_engine import PYWR_AVAILABLE  # noqa: E402

CFG = load_config()
# 평년 대비 규칙(공사 가뭄단계 연동) 검증용 가상 평년 저수율 곡선: 봄 높고 여름 낮음
CFG.setdefault("operation", {})["normal_curve_pct"] = [
    float(65 + 15 * np.cos(2 * np.pi * (d - 90) / 366)) for d in range(1, 367)]
INPUTS = prepare_inputs(load_weather(), CFG)


def test_gr4j_mass_balance_long_run():
    """장기간 유출 ≤ 강수 (x2=0이면 외부 유입이 없으므로)"""
    p, e = INPUTS["rain"].values, INPUTS["et0"].values
    q = run_gr4j(p, e, 350, 0.0, 90, 1.7)
    assert (q >= 0).all()
    assert q.sum() < p.sum()


@pytest.mark.parametrize("name", list(PRESETS))
def test_reservoir_water_balance(name):
    r = run_scenario(INPUTS, CFG, Scenario.from_dict(PRESETS[name].to_dict()), "2020-01-01", "2026-10-31", engine="reference")
    s = r.series
    dv = s["storage"].diff().iloc[1:]
    flux = (s["inflow"] - s["supply"] - s["env_flow"] - s["evap"] - s["spill"]).iloc[1:]
    assert np.allclose(dv, flux, atol=1e-6)
    res = CFG["reservoir"]
    assert s["storage"].min() >= res["dead_storage"] - 1e-6
    assert s["storage"].max() <= res["total_storage"] + 1e-6
    assert (s["supply"] <= s["demand"] + 1e-9).all()


def test_hedging_reduces_shortage_severity_in_drought():
    kw = dict(start="2025-10-01", end="2026-10-31", initial_pct=70, engine="reference")
    base = run_scenario(INPUTS, CFG, Scenario.from_dict(PRESETS["관행 운영"].to_dict()), **kw).kpis
    hedge = run_scenario(INPUTS, CFG, Scenario.from_dict(PRESETS["단계별 제한급수"].to_dict()), **kw).kpis
    assert hedge["shortage_index"] < base["shortage_index"]


@pytest.mark.skipif(not PYWR_AVAILABLE, reason="pywr 미설치")
@pytest.mark.parametrize("name", list(PRESETS))
def test_pywr_matches_reference(name):
    sc = Scenario.from_dict(PRESETS[name].to_dict())
    kw = dict(start="2025-01-01", end="2026-10-31", initial_pct=70)
    ref = run_scenario(INPUTS, CFG, sc, engine="reference", **kw).series
    pw = run_scenario(INPUTS, CFG, sc, engine="pywr", **kw).series
    # pywr의 저수율 구간 판정 방식 차이로 경계 근처에서 하루 어긋날 수 있어 0.5% 허용
    eff = CFG["reservoir"]["effective_storage"]
    assert (pw["storage"] - ref["storage"]).abs().max() < 0.005 * eff


@pytest.mark.skipif(not PYWR_AVAILABLE, reason="pywr 미설치")
def test_pywr_safe_across_threads():
    """웹앱처럼 여러 스레드에서 번갈아 pywr를 돌려도 프로세스가 죽지 않아야 함 (GLPK 스레드 문제 회귀 테스트)"""
    import threading

    sc = Scenario.from_dict(PRESETS["단계별 제한급수"].to_dict())
    errors = []

    def work():
        try:
            for _ in range(3):
                run_scenario(INPUTS, CFG, sc, "2026-01-01", "2026-06-30", 60, engine="pywr")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors


def test_optimizer_beats_presets_for_other_reservoir():
    """성주가 아닌 저수지(문경 경천)도 같은 방식으로 운영규칙 최적화 — 가뭄 조건에서 기존 규칙보다 SI가 작아야 함"""
    from engine import catalog
    from engine.operation import presets_for
    from engine.optimize import optimize_rules
    base = load_config()
    cfg = catalog.config_for("4728010036", base)
    inp = prepare_inputs(load_weather(), cfg, cfg["gr4j"])
    a = dict(start="2025-03-01", end="2025-10-31", initial_pct=45, demand_scale=1.2, inflow_scale=0.45)
    r = optimize_rules(inp, cfg, maxiter=12, **a)
    si = {k: run_scenario(inp, cfg, p, a["start"], a["end"], a["initial_pct"], a["demand_scale"], a["inflow_scale"],
                          engine="reference").kpis["shortage_index"] for k, p in presets_for(cfg).items()}
    assert r["kpis"]["shortage_index"] <= min(si.values()) + 1e-9
    r["scenario"].validate()


def test_normal_basis_rule_and_official_stage():
    """평년 대비 규칙: 같은 저수율이라도 평년이 높은 날엔 더 절감 / 공사 가뭄단계 경계값"""
    from engine.operation import band_factor, official_stage, presets_for

    sc = PRESETS["공사 가뭄단계 연동"]
    assert band_factor(45, sc, 50) == 1.0          # 평년 대비 90% → 정상
    assert band_factor(45, sc, 80) == 0.85         # 평년 대비 56% → 주의
    assert official_stage(74.5) == "정상" and official_stage(69.2) == "관심" and official_stage(59.3) == "주의"
    assert official_stage(45) == "경계" and official_stage(40) == "심각"
    assert "공사 가뭄단계 연동" not in presets_for({"operation": {}}) and "공사 가뭄단계 연동" in presets_for(CFG)
    with pytest.raises(ValueError):
        band_factor(45, sc, None)
