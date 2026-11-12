"""실시간 운영 전망 테스트"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import PRESETS, load_config, load_weather  # noqa: E402
from engine.realtime import build_weather, project, today_decision  # noqa: E402

CFG, HIST = load_config(), load_weather()
FC = pd.DataFrame({"rain": [0.0, 30.0, 5.0], "tmax": [25, 22, 24], "tmin": [15, 16, 14]},
                  index=pd.date_range("2026-06-02", periods=3, name="date"))


def test_build_weather_sources():
    w, src = build_weather(HIST, FC, pd.Timestamp("2026-06-01"), 10, "무강우")
    assert src.loc["2026-06-02"] == "예보" and src.loc["2026-06-05"] == "가정"
    assert w.loc["2026-06-03", "rain"] == 30.0 and w.loc["2026-06-06", "rain"] == 0.0
    assert not w.isna().any().any()


def test_forecast_rain_raises_storage():
    dry = project(HIST, None, CFG, 40, 20, "무강우", pd.Timestamp("2026-06-01").date())
    wet = project(HIST, FC.assign(rain=[60.0, 80.0, 40.0]), CFG, 40, 20, "무강우", pd.Timestamp("2026-06-01").date())
    k = "관행 운영"
    assert wet["runs"][k].kpis["end_storage_pct"] > dry["runs"][k].kpis["end_storage_pct"]


def test_today_decision_stages():
    sc = PRESETS["예보 연계 운영"]
    d = today_decision(30, 5.0, 100.0, sc)
    assert d["stage"] == "주의" and d["forecast_cut"] and d["factor"] == pytest.approx(0.75 * 0.9)
    assert today_decision(80, 50.0, 100.0, sc)["factor"] == 1.0
