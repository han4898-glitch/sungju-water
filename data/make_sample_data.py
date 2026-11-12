"""데모용 '가상' 기상자료와 '가상 관측' 저수량을 만듭니다.

※ 실제 관측자료가 아닙니다. 웹앱이 자료 없이도 돌아가도록 하는 예시입니다.
   과제 제출 전 fetch_kma.py(기상청 ASOS)와 RAWRIS 저수율 자료로 교체하세요.

- 2015-01-01 ~ 2026-10-31 일자료
- 경북 내륙 기후를 흉내 낸 계절 패턴(장마·태풍기 집중호우)
- 2026년 6~8월은 강수를 크게 줄여 올해 경북 가뭄 상황을 재현
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 월별 강수일 확률, 강수일 평균 강수량(mm) — 경북 내륙 연 1,100mm 내외를 목표로 설정
WET_P = [0.12, 0.15, 0.22, 0.25, 0.27, 0.35, 0.50, 0.42, 0.32, 0.18, 0.18, 0.12]
WET_MEAN = [4, 5.5, 7, 8.5, 9, 12.5, 15.5, 15, 12, 7, 6, 4]
T_MEAN = [-0.5, 2, 7.5, 13.5, 18.5, 22.5, 25.5, 26, 21, 14.5, 7.5, 1.5]
T_RANGE = [10, 11, 12, 13, 12, 9.5, 7.5, 8, 9, 11.5, 11, 10]


def make_weather(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", "2026-10-31", freq="D")
    rows = []
    wet_prev = False
    for d in idx:
        m = d.month - 1
        p = WET_P[m] * (1.35 if wet_prev else 0.85)
        drought = d.year == 2026 and 5 <= d.month <= 8
        if drought:
            p *= 0.75
        wet = rng.random() < p
        rain = rng.gamma(0.75, WET_MEAN[m] / 0.75) if wet else 0.0
        if drought:
            rain *= 0.85
        wet_prev = wet
        tm = T_MEAN[m] + rng.normal(0, 2.0) + (1.2 if drought else 0.0)
        tr = T_RANGE[m] * (0.65 if wet else 1.1) + rng.normal(0, 1)
        rh = np.clip((80 if wet else 62) + rng.normal(0, 8) + (8 if 6 <= d.month <= 8 else 0), 25, 99)
        wind = np.clip(rng.gamma(4, 0.45), 0.3, 8)
        sunshine = np.clip((2.0 if wet else 8.0) + rng.normal(0, 1.8), 0, 13)
        rows.append((d, round(rain, 1), round(tm + tr / 2, 1), round(tm - tr / 2, 1),
                     round(rh, 0), round(wind, 1), round(sunshine, 1)))
    return pd.DataFrame(rows, columns=["date", "rain", "tmax", "tmin", "rh", "wind", "sunshine"]).set_index("date")


def make_observed(weather: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """'참' 매개변수로 관행 운영을 돌려 만든 가상 관측 저수량(+측정 잡음).
    보정 기능이 참 매개변수를 되찾는지 확인하는 데 씁니다."""
    from engine import load_config, prepare_inputs, run_scenario, PRESETS

    cfg = load_config()
    true_params = {"x1": 520.0, "x2": -0.6, "x3": 140.0, "x4": 2.2}
    inputs = prepare_inputs(weather, cfg, true_params)
    r = run_scenario(inputs, cfg, PRESETS["관행 운영"], start="2019-01-01", end="2026-10-31",
                     initial_pct=75, engine="reference")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 120, len(r.series))  # ±120천㎥ 수준 계측 오차
    obs = pd.DataFrame(
        {"storage": (r.series["storage"] + noise).clip(cfg["reservoir"]["dead_storage"], cfg["reservoir"]["total_storage"]),
         "release": r.series["supply"] + r.series["env_flow"] + r.series["spill"]},
        index=r.series.index,
    )
    obs.index.name = "date"
    return obs.round(1), true_params


if __name__ == "__main__":
    w = make_weather()
    w.to_csv(ROOT / "data" / "weather_sample.csv")
    obs, tp = make_observed(w)
    obs.to_csv(ROOT / "data" / "observed_storage_sample.csv")
    yearly = w["rain"].resample("YE").sum().round(0)
    print("연강수량(가상):\n", yearly.to_string())
    print("가상관측 생성에 쓴 참 매개변수:", tp)
