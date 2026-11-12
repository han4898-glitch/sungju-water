"""가뭄 대응 검증(재현·과적합)과 SPI 테스트 — 가상자료로 알고리즘 확인"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import catalog, drought, spi  # noqa: E402
from engine.operation import PRESETS  # noqa: E402
from engine.runner import load_config, load_weather, prepare_inputs  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    cfg = catalog.config_for("4728010036", load_config())
    w = load_weather()
    inp = prepare_inputs(w, cfg, cfg["gr4j"])
    obs = pd.Series(60.0, index=w.index)  # 가상 관측 저수율 60%
    return cfg, w, inp, obs


def test_spi_is_standardized_and_detects_dry_spell(setup):
    _, w, _, _ = setup
    r = w["rain"].copy()
    s3 = spi.spi(r, 3).dropna()
    assert abs(s3.mean()) < 0.3 and 0.7 < s3.std() < 1.3
    dry = r.copy()
    dry.loc["2022-03-01":"2022-07-31"] = 0.0  # 5개월 무강우
    assert spi.spi(dry, 3).loc["2022-07-01"] < -1.0  # 자료 12년이면 무강수 확률 때문에 −1.4 안팎이 하한
    assert spi.classify(-2.1) == "극한가뭄" and spi.classify(0.2) == "정상"
    assert len(spi.current(r)) == 4


def test_replay_and_multi_year_optimizer(setup):
    cfg, _, inp, obs = setup
    ys = drought.years_available(inp, obs)
    assert len(ys) >= 5
    o = drought.optimize_multi(inp, cfg, ys[:-1], obs, maxiter=4, reserve_pct=20)
    o["scenario"].validate()
    rp = drought.replay(inp, cfg, obs, ys[-1], {"관행 운영": PRESETS["관행 운영"], "최적": o["scenario"]})
    assert rp["start_pct"] == pytest.approx(60) and set(rp["table"].index) == {"관행 운영", "최적"}
    # 다년 최적 규칙의 '규칙 찾은 해' 평균 SI는 기존 규칙들보다 크지 않아야 함 (단계별 제한급수가 탐색 시작점)
    base = drought.season_kpis(inp, cfg, PRESETS["단계별 제한급수"], ys[:-1], obs)["shortage_index"].mean()
    best = drought.season_kpis(inp, cfg, o["scenario"], ys[:-1], obs)
    pen = 0.05 * np.clip(20 - best["min_storage_pct"], 0, None) ** 2
    base_k = drought.season_kpis(inp, cfg, PRESETS["단계별 제한급수"], ys[:-1], obs)
    base_pen = 0.05 * np.clip(20 - base_k["min_storage_pct"], 0, None) ** 2
    assert (best["shortage_index"] + pen).mean() <= (base_k["shortage_index"] + base_pen).mean() + 1e-6
    assert base >= 0


def test_out_of_sample_table(setup):
    cfg, _, inp, obs = setup
    ys = drought.years_available(inp, obs)
    r = drought.out_of_sample(inp, cfg, obs, ys[:4], ys[4:], maxiter=3)
    t = r["table"]
    assert set(t["구분"]) == {"규칙 찾은 해", "검증 해"} and "다년 최적 규칙" in set(t["규칙"])
