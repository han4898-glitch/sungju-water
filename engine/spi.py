"""표준강수지수 SPI (Standardized Precipitation Index, McKee et al. 1993) — 기상청 강수로 기상가뭄 판단

SPI-n: 최근 n개월 누적강수를 같은 달(종료월)의 과거 분포(감마분포)에 맞춘 뒤 표준정규값으로 바꾼 것
  0 이하 = 평년보다 건조,  -1.0 이하 보통가뭄, -1.5 이하 심한가뭄, -2.0 이하 극한가뭄 (기상청 가뭄 단계 구분과 같은 값)
보통 SPI-3 = 농업(토양수분·저수지 유입), SPI-6 = 저수지·수문 가뭄의 선행지표로 씀
자료 기간이 짧으면(30년 미만) 분포 추정이 불안정하므로 참고용으로 해석
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

CLASSES = [(-np.inf, -2.0, "극한가뭄"), (-2.0, -1.5, "심한가뭄"), (-1.5, -1.0, "보통가뭄"),
           (-1.0, 1.0, "정상"), (1.0, 1.5, "약간습윤"), (1.5, np.inf, "습윤")]


def classify(v: float) -> str:
    if v is None or np.isnan(v):
        return "-"
    for lo, hi, name in CLASSES:
        if lo < v <= hi:
            return name
    return "-"


def monthly_rain(rain: pd.Series, min_days: int = 25) -> pd.Series:
    """일강수 → 월강수 (관측일 25일 미만인 달은 결측)"""
    r = rain.astype(float)
    m = r.resample("MS").sum(min_count=1)
    n = r.resample("MS").count()
    return m.where(n >= min_days)


def spi(rain_daily: pd.Series, scale: int = 3) -> pd.Series:
    """월별 SPI-scale (index = 월 시작일, 종료월 기준)"""
    acc = monthly_rain(rain_daily).rolling(scale, min_periods=scale).sum()
    out = pd.Series(np.nan, index=acc.index)
    for mth in range(1, 13):
        x = acc[acc.index.month == mth].dropna()
        if len(x) < 5:
            continue
        pos = x[x > 0]
        q = 1 - len(pos) / len(x)  # 무강수 확률
        if len(pos) >= 4 and pos.std() > 0:
            a, _, b = stats.gamma.fit(pos.values, floc=0)
            cdf = q + (1 - q) * stats.gamma.cdf(x.values, a, scale=b)
        else:  # 표본이 너무 적으면 경험분포
            cdf = (x.rank().values - 0.44) / (len(x) + 0.12)
        out.loc[x.index] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    return out


def current(rain_daily: pd.Series, scales=(1, 3, 6, 12)) -> pd.DataFrame:
    """가장 최근 완결된 달 기준 SPI 표 (이번 달이 25일 미만이면 지난달 기준)"""
    rows = []
    for s in scales:
        v = spi(rain_daily, s).dropna()
        if len(v):
            rows.append({"지수": f"SPI-{s}", "기준월": v.index[-1].strftime("%Y-%m"), "값": float(v.iloc[-1]),
                         "판정": classify(float(v.iloc[-1]))})
    return pd.DataFrame(rows)
