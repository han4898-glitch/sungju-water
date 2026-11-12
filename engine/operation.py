"""저수지 운영 시나리오(운영규칙) 정의.

각 시나리오는 '저수율 구간별 공급 비율' + '예보 연계 절감'으로 표현됩니다.
두 계산 엔진(pywr, reference)이 같은 규칙을 그대로 읽어 씁니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    name: str
    description: str = ""
    # 유효저수량 대비 저수율(%) 경계. 내림차순. 예: [60, 40, 25] → 관심/주의/경계/심각 4구간
    curves_pct: list[float] = field(default_factory=list)
    # 각 구간의 공급 비율. len = len(curves_pct) + 1, 위 구간부터
    supply_factors: list[float] = field(default_factory=lambda: [1.0])
    # 예보 연계: 향후 N일 강수 합이 기준보다 적으면 추가 절감
    forecast_days: int = 0
    forecast_threshold_mm: float = 0.0
    forecast_cut: float = 0.0  # 추가 절감 비율 (0.1 = 10%)
    supply_env_flow: bool = True
    # 경계 기준: "absolute" = 유효저수율(%) 그대로, "normal" = 평년 대비 저수율(%) = 저수율 ÷ 같은 날 평년 저수율 × 100
    # (한국농어촌공사 농업가뭄 단계: 평년 대비 70% 초과 정상, 60~70 관심, 50~60 주의, 40~50 경계, 40 이하 심각)
    basis: str = "absolute"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> None:
        if len(self.supply_factors) != len(self.curves_pct) + 1:
            raise ValueError("supply_factors 길이는 curves_pct 길이 + 1 이어야 합니다.")
        if any(a <= b for a, b in zip(self.curves_pct, self.curves_pct[1:])):
            raise ValueError("curves_pct는 내림차순이어야 합니다.")
        if not all(0 <= f <= 1.5 for f in self.supply_factors):
            raise ValueError("supply_factors는 0~1.5 범위여야 합니다.")
        if self.basis not in ("absolute", "normal"):
            raise ValueError("basis는 'absolute' 또는 'normal' 이어야 합니다.")


PRESETS: dict[str, Scenario] = {
    "관행 운영": Scenario(
        name="관행 운영",
        description="저수량이 남아 있는 한 필요수량을 그대로 공급",
        curves_pct=[],
        supply_factors=[1.0],
    ),
    "단계별 제한급수": Scenario(
        name="단계별 제한급수",
        description="저수율 60/40/25% 아래로 내려가면 공급을 90/75/55%로 단계 절감",
        curves_pct=[60, 40, 25],
        supply_factors=[1.0, 0.9, 0.75, 0.55],
    ),
    "예보 연계 운영": Scenario(
        name="예보 연계 운영",
        description="단계별 제한급수 + 향후 10일 예보 강수가 20mm 미만이면 10% 추가 절감",
        curves_pct=[60, 40, 25],
        supply_factors=[1.0, 0.9, 0.75, 0.55],
        forecast_days=10,
        forecast_threshold_mm=20.0,
        forecast_cut=0.10,
    ),
    "공사 가뭄단계 연동": Scenario(
        name="공사 가뭄단계 연동",
        description="한국농어촌공사 농업가뭄 단계(평년 대비 저수율 70/60/50/40%)에 맞춰 공급을 95/85/70/55%로 절감 "
                    "(단계 기준은 ADMS 농업가뭄관리시스템, 단계별 절감률은 예시값)",
        curves_pct=[70, 60, 50, 40],
        supply_factors=[1.0, 0.95, 0.85, 0.7, 0.55],
        basis="normal",
    ),
}

OFFICIAL_STAGES = [(70, "정상"), (60, "관심"), (50, "주의"), (40, "경계"), (-1, "심각")]


def official_stage(ratio_to_normal_pct: float | None) -> str:
    """한국농어촌공사 농업가뭄 단계 (평년 대비 저수율 %, ADMS 저수율지도 기준: 70 초과 정상 … 40 이하 심각)"""
    if ratio_to_normal_pct is None or np.isnan(ratio_to_normal_pct):
        return "-"
    for th, name in OFFICIAL_STAGES:
        if ratio_to_normal_pct > th:
            return name
    return "심각"


def normal_series(cfg: dict, index: pd.DatetimeIndex) -> pd.Series | None:
    """cfg["operation"]["normal_curve_pct"] (1~366일 평년 저수율) → 날짜별 시계열. 없으면 None"""
    curve = cfg.get("operation", {}).get("normal_curve_pct")
    if not curve:
        return None
    arr = np.asarray(curve, float)
    doy = np.minimum(index.dayofyear.values, len(arr)) - 1
    return pd.Series(arr[doy], index=index)


def needs_normal(sc: Scenario) -> bool:
    return sc.basis == "normal" and bool(sc.curves_pct)


def presets_for(cfg: dict | None) -> dict[str, Scenario]:
    """이 저수지에서 쓸 수 있는 기본 규칙 (평년 저수율 곡선이 없으면 평년 대비 규칙은 뺌)"""
    has_normal = bool(cfg and cfg.get("operation", {}).get("normal_curve_pct"))
    return {k: v for k, v in PRESETS.items() if has_normal or not needs_normal(v)}


def attach_normal_curve(cfg: dict, obs_rate: pd.Series | None) -> dict:
    """관측 일별 저수율(전체 기간)로 1~366일 평년 저수율 곡선을 만들어 cfg에 넣음 (평년 대비 규칙·가뭄단계용)"""
    if obs_rate is None:
        return cfg
    o = obs_rate.dropna().clip(0, 100)
    if len(o) < 365 * 2:
        return cfg
    g = o.groupby(o.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    cfg.setdefault("operation", {})["normal_curve_pct"] = [round(float(v), 2) for v in g.values]
    return cfg


def forecast_factor(rain: pd.Series, sc: Scenario) -> pd.Series:
    """예보 연계 절감 계수 시계열.

    과거 모의에서는 '완전 예보'(실제 미래 강수)를 예보로 간주합니다.
    실시간 운영에서는 기상청 단기·중기예보 강수로 이 값을 대체합니다(mcp_server 참조).
    """
    if sc.forecast_days <= 0 or sc.forecast_cut <= 0:
        return pd.Series(1.0, index=rain.index)
    fut = rain[::-1].rolling(sc.forecast_days, min_periods=1).sum()[::-1].shift(-1).fillna(0)
    return pd.Series(np.where(fut < sc.forecast_threshold_mm, 1 - sc.forecast_cut, 1.0), index=rain.index)


def band_factor(storage_pct_effective: float, sc: Scenario, normal_pct: float | None = None) -> float:
    """현재 유효저수율(%)이 속한 구간의 공급 비율 (pywr EffectiveStorageBandParameter와 같은 규칙).
    sc.basis == "normal" 이면 평년 대비 저수율(= 저수율 ÷ 평년 × 100)로 구간을 판단"""
    x = storage_pct_effective
    if sc.basis == "normal":
        if normal_pct is None or not normal_pct > 0:
            raise ValueError("평년 대비 규칙에는 평년 저수율 곡선이 필요합니다 (농어촌공사 일별 저수율을 받아 주세요).")
        x = storage_pct_effective / normal_pct * 100
    for curve, factor in zip(sc.curves_pct, sc.supply_factors):
        if x >= curve:
            return factor
    return sc.supply_factors[len(sc.curves_pct)]
