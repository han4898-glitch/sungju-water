"""기준(reference) 저수지 물수지 엔진 — pywr 결과 교차검증용.

pywr 네트워크와 동일한 우선순위로 일 단위 물수지를 계산합니다.
  1) 수면증발  2) 관개용수(운영규칙 적용)  3) 하천유지유량  4) 만수위 초과분 월류
pywr가 설치되지 않은 환경에서는 대체 엔진으로도 쓰입니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .operation import Scenario, band_factor


def run_reference(
    inflow: pd.Series,
    demand: pd.Series,
    evap: pd.Series,
    fc_factor: pd.Series,
    sc: Scenario,
    total_storage: float,
    dead_storage: float,
    initial_storage: float,
    env_flow: float,
    cap_storage: float | None = None,
    normal_pct: pd.Series | None = None,
) -> pd.DataFrame:
    """cap_storage: 운영 상한 저수량 (이를 넘는 물은 방류). 없으면 총저수량
    normal_pct: 날짜별 평년 저수율(%) — 평년 대비 규칙(sc.basis == "normal")에 필요"""
    eff = total_storage - dead_storage
    cap = total_storage if cap_storage is None else min(cap_storage, total_storage)
    # pandas 값 하나씩 읽기(.at)는 느려서 배열로 바꿔 계산 (결과는 같음)
    inf_a = np.asarray(inflow, float)
    dem_a = np.asarray(demand.reindex(inflow.index), float)
    eva_a = np.asarray(evap.reindex(inflow.index), float)
    fc_a = np.asarray(fc_factor.reindex(inflow.index), float)
    nr_a = np.asarray(normal_pct.reindex(inflow.index), float) if normal_pct is not None else None
    curves, factors = list(sc.curves_pct), list(sc.supply_factors)
    normal_basis = sc.basis == "normal" and bool(curves)
    if normal_basis and nr_a is None:
        band_factor(50.0, sc, None)  # 평년 곡선이 없으면 같은 오류 메시지로 중단
    env_target = env_flow if sc.supply_env_flow else 0.0
    n = len(inf_a)
    rec = np.empty((n, 9))
    v = float(initial_storage)
    for i in range(n):
        pct = (v - dead_storage) / eff * 100
        x = pct / nr_a[i] * 100 if normal_basis else pct
        bf = factors[len(curves)]
        for c, f in zip(curves, factors):
            if x >= c:
                bf = f
                break
        factor = bf * fc_a[i]
        target = dem_a[i] * factor
        avail = v + inf_a[i] - dead_storage
        e = min(eva_a[i], max(avail, 0.0)); avail -= e
        supply = min(target, max(avail, 0.0)); avail -= supply
        env = min(env_target, max(avail, 0.0)); avail -= env
        v_new = dead_storage + avail
        spill = max(0.0, v_new - cap)
        v_new -= spill
        rec[i] = (v_new, inf_a[i], dem_a[i], target, supply, env, e, spill, factor)
        v = v_new
    return pd.DataFrame(
        rec,
        index=inflow.index,
        columns=["storage", "inflow", "demand", "target", "supply", "env_flow", "evap", "spill", "factor"],
    )
