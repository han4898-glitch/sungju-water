"""pywr 기반 저수지 운영 모델.

pywr(https://github.com/pywr/pywr)는 선형계획법으로 매 시간 단계마다 물 배분을 푸는
오픈소스 수자원 시스템 모델입니다. 이 모듈은 성주댐 네트워크를 pywr JSON으로 만들어 실행합니다.

네트워크:
    catchment(유입) ──► reservoir(성주댐) ──┬─► evaporation (cost -1000, 최우선)
                                            ├─► irrigation  (cost  -500, 관개)
                                            ├─► env_flow    (cost  -200, 하천유지)
                                            └─► spill       (cost     0, 월류)
    reservoir cost = -10  → 물을 저류하는 것이 월류보다 유리, 관개보다는 불리
관개 상한 = 필요수량 × 유효저수율 구간 계수(EffectiveStorageBandParameter) × 예보 계수
"""
from __future__ import annotations

import concurrent.futures
import gc
import os
import tempfile

import pandas as pd

from .operation import Scenario

try:
    from pywr.model import Model  # type: ignore
    from pywr.parameters import Parameter, load_parameter  # type: ignore

    PYWR_AVAILABLE = True
except Exception:  # pragma: no cover - 설치 여부에 따라
    PYWR_AVAILABLE = False


if PYWR_AVAILABLE:

    class EffectiveStorageBandParameter(Parameter):
        """유효저수율(사수량 제외) 기준 저수율 구간별 공급비율.

        pywr 기본 ControlCurveParameter는 총저수량 대비 저수율로 구간을 판단하므로,
        사수량이 큰 저수지(성주댐: 총저수량의 26%)에서는 운영규칙 기준(유효저수율)과 달라집니다.
        이 매개변수는 각 시간 단계 시작 시점의 저수량으로 유효저수율을 직접 계산합니다.
        """

        def __init__(self, model, storage_node, curves_pct, values, dead_storage, effective_storage,
                     normal_param=None, **kwargs):
            super().__init__(model, **kwargs)
            self.storage_node = storage_node
            self.curves_pct = list(curves_pct)
            self.values = list(values)
            self.dead_storage = float(dead_storage)
            self.effective_storage = float(effective_storage)
            self.normal_param = normal_param  # 평년 저수율(%) 시계열 — 평년 대비 규칙일 때
            if normal_param is not None:
                self.children.add(normal_param)

        def value(self, ts, scenario_index):
            v = self.storage_node.volume[scenario_index.global_id]
            pct = (v - self.dead_storage) / self.effective_storage * 100.0
            if self.normal_param is not None:
                pct = pct / max(self.normal_param.get_value(scenario_index), 1e-6) * 100.0
            for curve, factor in zip(self.curves_pct, self.values):
                if pct >= curve:
                    return factor
            return self.values[len(self.curves_pct)]

        @classmethod
        def load(cls, model, data):
            ref = data.pop("storage_node")
            node = model.nodes[ref]
            nref = data.pop("normal", None)
            normal = load_parameter(model, nref) if nref is not None else None
            return cls(model, node, data.pop("curves_pct"), data.pop("values"),
                       data.pop("dead_storage"), data.pop("effective_storage"), normal, **data)

    EffectiveStorageBandParameter.register()


def build_model_dict(
    ts_csv: str,
    start: str,
    end: str,
    sc: Scenario,
    total_storage: float,
    dead_storage: float,
    initial_storage: float,
    env_flow: float,
    cap_storage: float | None = None,
) -> dict:
    def ts(col):
        return {"type": "dataframe", "url": ts_csv, "column": col, "index_col": "date", "parse_dates": True}

    if sc.curves_pct:
        band = {
            "type": "effectivestorageband",
            "storage_node": "reservoir",
            "curves_pct": list(sc.curves_pct),
            "values": list(sc.supply_factors),
            "dead_storage": dead_storage,
            "effective_storage": total_storage - dead_storage,
        }
        if sc.basis == "normal":
            band["normal"] = "normal_ts"
    else:
        band = {"type": "constant", "value": sc.supply_factors[0]}

    return {
        "metadata": {"title": f"성주댐 용수공급 - {sc.name}", "minimum_version": "1.0"},
        "timestepper": {"start": start, "end": end, "timestep": 1},
        "parameters": {
            "inflow_ts": ts("inflow"),
            "demand_ts": ts("demand"),
            "evap_ts": ts("evap"),
            "forecast_factor": ts("fc_factor"),
            "normal_ts": ts("normal_pct"),
            "band_factor": band,
            "irrigation_max": {
                "type": "aggregated",
                "agg_func": "product",
                "parameters": ["demand_ts", "band_factor", "forecast_factor"],
            },
        },
        "nodes": [
            {"name": "catchment", "type": "catchment", "flow": "inflow_ts"},
            {
                "name": "reservoir",
                "type": "storage",
                "max_volume": total_storage if cap_storage is None else min(cap_storage, total_storage),
                "min_volume": dead_storage,
                "initial_volume": initial_storage,
                "cost": -10,
            },
            {"name": "evaporation", "type": "output", "max_flow": "evap_ts", "cost": -1000},
            {"name": "irrigation", "type": "output", "max_flow": "irrigation_max", "cost": -500},
            {"name": "env_flow", "type": "output", "max_flow": env_flow if sc.supply_env_flow else 0.0, "cost": -200},
            {"name": "spill", "type": "output", "cost": 0},
        ],
        "edges": [
            ["catchment", "reservoir"],
            ["reservoir", "evaporation"],
            ["reservoir", "irrigation"],
            ["reservoir", "env_flow"],
            ["reservoir", "spill"],
        ],
        "recorders": {
            "rec_storage": {"type": "numpyarraystoragerecorder", "node": "reservoir"},
            "rec_irrigation": {"type": "numpyarraynoderecorder", "node": "irrigation"},
            "rec_env": {"type": "numpyarraynoderecorder", "node": "env_flow"},
            "rec_evap": {"type": "numpyarraynoderecorder", "node": "evaporation"},
            "rec_spill": {"type": "numpyarraynoderecorder", "node": "spill"},
            "rec_target": {"type": "numpyarrayparameterrecorder", "parameter": "irrigation_max"},
        },
    }


# GLPK(pywr의 선형계획 해법기)는 메모리를 스레드별로 관리합니다. 웹앱(Streamlit)은 실행할 때마다 다른
# 스레드를 쓰므로, 한 스레드에서 만든 모델이 다른 스레드에서 해제되면 'glp_free: memory allocation error'로
# 프로세스 전체가 종료됩니다. → 모든 pywr 계산을 전용 스레드 하나에서 만들고·풀고·해제합니다.
_PYWR_THREAD = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pywr")


def run_pywr(*args, **kwargs) -> pd.DataFrame:
    if not PYWR_AVAILABLE:
        raise RuntimeError("pywr가 설치되어 있지 않습니다. `pip install pywr`")
    return _PYWR_THREAD.submit(_run_pywr_impl, *args, **kwargs).result()


def _run_pywr_impl(
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
    if not PYWR_AVAILABLE:
        raise RuntimeError("pywr가 설치되어 있지 않습니다. `pip install pywr`")
    if sc.basis == "normal" and sc.curves_pct and normal_pct is None:
        raise ValueError("평년 대비 규칙에는 평년 저수율 곡선이 필요합니다 (농어촌공사 일별 저수율을 받아 주세요).")

    ts = pd.DataFrame({"inflow": inflow, "demand": demand, "evap": evap, "fc_factor": fc_factor,
                       "normal_pct": normal_pct.values if normal_pct is not None else 100.0})
    ts.index.name = "date"
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "timeseries.csv")
        ts.to_csv(csv)
        data = build_model_dict(
            csv,
            inflow.index[0].strftime("%Y-%m-%d"),
            inflow.index[-1].strftime("%Y-%m-%d"),
            sc,
            total_storage,
            dead_storage,
            initial_storage,
            env_flow,
            cap_storage,
        )
        model = Model.load(data)
        model.run()

        def col(name):
            return model.recorders[name].to_dataframe().iloc[:, 0].values

        out = pd.DataFrame(
            {
                "storage": col("rec_storage").copy(),
                "inflow": inflow.values,
                "demand": demand.values,
                "target": col("rec_target"),
                "supply": col("rec_irrigation"),
                "env_flow": col("rec_env"),
                "evap": col("rec_evap"),
                "spill": col("rec_spill"),
            },
            index=inflow.index,
        )
        del model  # 같은 스레드에서 해제
        gc.collect()
    out["factor"] = (out["target"] / out["demand"].where(out["demand"] > 0)).fillna(1.0)
    return out
