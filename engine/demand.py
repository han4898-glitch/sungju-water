"""기준증발산량(ET0)과 논 관개 필요수량 산정.

- ET0: FAO-56 Penman-Monteith (일사·풍속·습도 자료가 있을 때),
       없으면 Hargreaves 식으로 대체
- 논 필요수량: 담수심 물수지(써레질 → 담수 유지 → 관개 종료) 방식.
  국내 농업용수 수요량 산정에서 쓰는 일반적인 논 담수심 모의 구조를 따릅니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- ET0 ----
def _extraterrestrial_radiation(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    """Ra (MJ/m²/day), FAO-56 식 21"""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )


def et0_hargreaves(df: pd.DataFrame, lat: float) -> pd.Series:
    doy = df.index.dayofyear.values
    ra = _extraterrestrial_radiation(doy, lat)
    tmean = (df["tmax"] + df["tmin"]) / 2
    td = (df["tmax"] - df["tmin"]).clip(lower=0)
    et0 = 0.0023 * (tmean + 17.8) * np.sqrt(td) * ra * 0.408
    return et0.clip(lower=0)


def et0_penman_monteith(df: pd.DataFrame, lat: float, elev: float) -> pd.Series:
    """필요 열: tmax, tmin, rh(%), wind(m/s, 10m), sunshine(hr)"""
    doy = df.index.dayofyear.values
    tmax, tmin = df["tmax"].values, df["tmin"].values
    tmean = (tmax + tmin) / 2
    u2 = df["wind"].values * 4.87 / np.log(67.8 * 10 - 5.42)  # 10m → 2m
    p = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
    gamma = 0.000665 * p
    es_ = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))
    es = (es_(tmax) + es_(tmin)) / 2
    ea = df["rh"].values / 100 * es
    delta = 4098 * es_(tmean) / (tmean + 237.3) ** 2

    ra = _extraterrestrial_radiation(doy, lat)
    phi = np.deg2rad(lat)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1, 1))
    nmax = 24 / np.pi * ws
    rs = (0.25 + 0.5 * df["sunshine"].values / nmax) * ra
    rso = (0.75 + 2e-5 * elev) * ra
    rns = 0.77 * rs
    sigma = 4.903e-9
    rnl = (
        sigma * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
        * (0.34 - 0.14 * np.sqrt(np.clip(ea, 0, None)))
        * (1.35 * np.clip(rs / rso, 0, 1) - 0.35)
    )
    rn = rns - rnl
    et0 = (0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)) / (
        delta + gamma * (1 + 0.34 * u2)
    )
    return pd.Series(np.clip(et0, 0, None), index=df.index)


def compute_et0(df: pd.DataFrame, lat: float, elev: float) -> pd.Series:
    """자료가 다 있는 날은 Penman-Monteith, 습도·풍속·일조가 빠진 날만 Hargreaves (장기 자료 앞부분 대비)"""
    cols = ["tmax", "tmin", "rh", "wind", "sunshine"]
    hg = et0_hargreaves(df, lat)
    if not set(cols).issubset(df.columns):
        return hg
    ok = df[cols].notna().all(axis=1)
    if not ok.any():
        return hg
    pm = et0_penman_monteith(df[cols].fillna(df[cols].mean()), lat, elev)
    return pm.where(ok, hg)


# ------------------------------------------------------- 논 필요수량 ----
def _md(date: pd.Timestamp) -> str:
    return date.strftime("%m-%d")


def _kc(date: pd.Timestamp, kc_dates, kc_values) -> float:
    year = date.year
    xs = [pd.Timestamp(f"{year}-{d}").dayofyear for d in kc_dates]
    return float(np.interp(date.dayofyear, xs, kc_values))


def paddy_irrigation_demand(
    weather: pd.DataFrame, et0: pd.Series, irr: dict
) -> pd.DataFrame:
    """일별 논 관개 필요수량을 계산합니다.

    반환 열:
      net_mm    : 논에서 필요한 순용수량 (mm)
      gross_mm  : 송수손실 포함 조용수량 (mm)
      demand    : 저수지에서 내보내야 할 양 (천㎥/day)
      ponding_mm: 담수심 (mm)
    """
    area = irr["area_ha"]
    eff = irr["conveyance_efficiency"]
    h = 0.0
    rows = []
    for date in weather.index:
        md = _md(date)
        rain = float(weather.at[date, "rain"])
        net = 0.0
        in_season = irr["season_start"] <= md <= irr["season_end"]
        puddling = irr["season_start"] <= md <= irr["puddling_end"]
        if not in_season:
            h = 0.0
        elif puddling:
            start = pd.Timestamp(f"{date.year}-{irr['season_start']}")
            end = pd.Timestamp(f"{date.year}-{irr['puddling_end']}")
            ndays = (end - start).days + 1
            # 써레질 용수를 기간에 나누어 공급, 강우는 그만큼 차감
            net = max(0.0, irr["puddling_depth_mm"] / ndays - rain)
            h = min(irr["ponding_target_mm"], h + 2.5)  # 이앙 직후 담수 형성
        else:
            kc = _kc(date, irr["kc_dates"], irr["kc_values"])
            etc = kc * float(et0.at[date])
            h = h + rain - etc - irr["percolation_mm"]
            if h > irr["ponding_max_mm"]:
                h = irr["ponding_max_mm"]  # 물꼬 초과분 배수 (유효우량 아님)
            # 관개구역 전체를 평균한 담수심을 목표 수심으로 매일 보충 (용수로 일 공급능력 상한)
            if h < irr["ponding_target_mm"]:
                net = min(irr["ponding_target_mm"] - h, irr["canal_capacity_mm"])
                h += net
        gross = net / eff
        rows.append((net, gross, gross * area * 0.01, h))
    out = pd.DataFrame(rows, index=weather.index, columns=["net_mm", "gross_mm", "demand", "ponding_mm"])
    return out
