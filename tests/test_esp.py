"""앙상블 확률 전망(ESP) 테스트"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import catalog, esp  # noqa: E402
from engine.operation import PRESETS  # noqa: E402
from engine.runner import load_config, load_weather  # noqa: E402

START, DAYS = pd.Timestamp("2025-04-01"), 120


@pytest.fixture(scope="module")
def setup():
    cfg = catalog.config_for("4728010036", load_config())  # 성주 아닌 저수지
    h = load_weather()
    m = esp.build_members(h, START, DAYS)
    return cfg, h, m, esp.prepare_members(m, cfg, cfg["gr4j"])


def test_members_are_past_years_shifted(setup):
    _, h, m, _ = setup
    assert len(m) >= 8 and all(y < START.year for y in m)
    y = min(m)
    fut = m[y].loc[START:START + pd.Timedelta(days=9), "rain"].values
    src = h.loc[START - pd.DateOffset(years=START.year - y):, "rain"].values[:10]
    assert (fut == src).all()  # 미래 구간 = 그해 같은 날짜 관측
    assert (m[y].loc[:START - pd.Timedelta(days=1), "rain"] == h.loc[m[y].index.min():START - pd.Timedelta(days=1), "rain"]).all()


def test_forecast_overrides_first_days(setup):
    _, h, _, _ = setup
    fc = pd.DataFrame({"rain": [77.0, 66.0]}, index=pd.date_range(START, periods=2))
    m = esp.build_members(h, START, DAYS, fc)
    assert all(w.loc[START, "rain"] == 77.0 for w in m.values())


def test_weights_follow_outlook(setup):
    _, _, m, _ = setup
    w = esp.tercile_weights(m, START, DAYS, (60, 30, 10))
    tot = pd.Series({y: m[y].loc[START:START + pd.Timedelta(days=89), "rain"].sum() for y in m})
    assert w.sum() == pytest.approx(1) and w[tot.idxmin()] > w[tot.idxmax()]


def test_esp_band_and_rule_comparison(setup):
    cfg, _, _, pp = setup
    r = esp.run_esp(pp, cfg, PRESETS["단계별 제한급수"], START, DAYS, 50)
    b = r["band"]
    assert len(b) == DAYS and (b["p10"] <= b["p50"] + 1e-9).all() and (b["p50"] <= b["p90"] + 1e-9).all()
    assert b["p50"].iloc[0] == pytest.approx(50, abs=3)
    cmp = esp.compare_rules(pp, cfg, START, DAYS, 50)
    from engine.operation import presets_for
    assert set(presets_for(cfg)) <= set(cmp.index)


def test_esp_optimizer_not_worse_than_presets(setup):
    cfg, _, _, pp = setup
    o = esp.optimize_rules_esp(pp, cfg, START, DAYS, 35, maxiter=5)
    cmp = esp.compare_rules(pp, cfg, START, DAYS, 35)
    assert o["expected_si"] <= cmp["shortage_index"].min() + 1e-6


def test_error_correction_widens_band(setup):
    """관측 오차 보정: 가짜 관측(모의 + 체계적 +8%p)으로 재예측 오차를 만들면 보정 후 중앙값이 그만큼 올라가야 함"""
    cfg, h, _, pp = setup
    from engine.runner import prepare_inputs
    inp = prepare_inputs(h, cfg, cfg["gr4j"])
    base = esp._run(inp, cfg, PRESETS["관행 운영"], "2016-01-01", "2024-12-31", 60)["storage_pct"]
    fake_obs = (base + 8).clip(0, 100)
    errs = esp.hindcast_errors(inp, cfg, fake_obs, 60, before=START, step=30)
    assert len(errs) == 60
    raw = esp.run_esp(pp, cfg, PRESETS["관행 운영"], START, 60, 50)
    cor = esp.run_esp(pp, cfg, PRESETS["관행 운영"], START, 60, 50, errors=errs)
    assert cor["corrected"] and (cor["band"]["p50"] - raw["band"]["p50"]).iloc[-1] > 2


def test_one_shot_supply_plan(setup):
    """가뭄 예측 → 공급계획 한 번에 (성주 아닌 저수지, 가상 관측)"""
    from engine import plan
    from engine.operation import attach_normal_curve
    from engine.runner import prepare_inputs

    cfg, h, _, _ = setup
    obs = pd.Series(55.0 + 20 * np.sin(np.arange(len(h)) * 2 * np.pi / 365), index=h.index)
    attach_normal_curve(cfg, obs)
    inp = prepare_inputs(h, cfg, cfg["gr4j"])
    p = plan.make_plan("4728010036", cfg, h, inp, cfg["gr4j"], obs.loc[:"2025-03-31"], None, None, None,
                       None, 20, True, today=pd.Timestamp("2025-04-01"), maxiter=3)
    assert p["days"] == 183 and p["rule_name"] in p["compare"].index
    assert 0 < p["today_factor"] <= 1 and "공급계획" in p["summary"]
    assert set(p["monthly"].columns) >= {"필요수량(천㎥)", "공급계획(천㎥)", "공급 비율(%)"}
    assert (p["monthly"]["공급계획(천㎥)"] <= p["monthly"]["필요수량(천㎥)"] + 1e-6).all()
