"""GR4J 일 강우-유출 모형 (Perrin, Michel & Andréassian, 2003).

프랑스 IRSTEA(현 INRAE)가 개발해 공개한 4매개변수 개념적 모형으로,
airGR(R) 등 여러 오픈소스 구현이 있습니다. 여기서는 원 논문의 식을 그대로 옮겼습니다.

입력: 일강수 P (mm), 잠재증발산 E (mm)
출력: 유역 유출고 Q (mm/day)
"""
from __future__ import annotations

import numpy as np


def _s_curve1(t: float, x4: float) -> float:
    if t <= 0:
        return 0.0
    if t < x4:
        return (t / x4) ** 2.5
    return 1.0


def _s_curve2(t: float, x4: float) -> float:
    if t <= 0:
        return 0.0
    if t < x4:
        return 0.5 * (t / x4) ** 2.5
    if t < 2 * x4:
        return 1.0 - 0.5 * (2.0 - t / x4) ** 2.5
    return 1.0


def _unit_hydrographs(x4: float):
    n1 = int(np.ceil(x4))
    n2 = int(np.ceil(2 * x4))
    uh1 = np.array([_s_curve1(i, x4) - _s_curve1(i - 1, x4) for i in range(1, n1 + 1)])
    uh2 = np.array([_s_curve2(i, x4) - _s_curve2(i - 1, x4) for i in range(1, n2 + 1)])
    return uh1, uh2


def run_gr4j(
    precip: np.ndarray,
    pet: np.ndarray,
    x1: float,
    x2: float,
    x3: float,
    x4: float,
    s0_frac: float = 0.5,
    r0_frac: float = 0.4,
) -> np.ndarray:
    """GR4J를 일 단위로 실행해 유출고(mm/day) 배열을 돌려줍니다."""
    precip = np.asarray(precip, dtype=float)
    pet = np.asarray(pet, dtype=float)
    n = len(precip)
    x4 = max(x4, 0.5)
    uh1, uh2 = _unit_hydrographs(x4)
    q9_store = np.zeros(len(uh1))
    q1_store = np.zeros(len(uh2))

    s = s0_frac * x1  # 생산저류고
    r = r0_frac * x3  # 추적저류고
    q_out = np.zeros(n)

    for t in range(n):
        p, e = precip[t], pet[t]
        # --- 생산 함수 ---
        if p >= e:
            pn, en = p - e, 0.0
            tw = np.tanh(min(pn / x1, 13.0))
            ps = x1 * (1 - (s / x1) ** 2) * tw / (1 + s / x1 * tw)
            es = 0.0
        else:
            pn, en = 0.0, e - p
            tw = np.tanh(min(en / x1, 13.0))
            es = s * (2 - s / x1) * tw / (1 + (1 - s / x1) * tw)
            ps = 0.0
        s = s - es + ps
        # 침루
        perc = s * (1 - (1 + (4.0 / 9.0 * s / x1) ** 4) ** -0.25)
        s -= perc
        pr = perc + (pn - ps)

        # --- 추적 함수 ---
        q9_store = np.roll(q9_store, -1)
        q9_store[-1] = 0.0
        q9_store += 0.9 * pr * uh1
        q1_store = np.roll(q1_store, -1)
        q1_store[-1] = 0.0
        q1_store += 0.1 * pr * uh2
        q9 = q9_store[0] if len(uh1) else 0.0
        q1 = q1_store[0] if len(uh2) else 0.0
        # np.roll 이후 0번이 현재 시간 단계 값 (아래에서 다시 소비 처리)
        q9_store[0] = 0.0
        q1_store[0] = 0.0

        f = x2 * (r / x3) ** 3.5  # 지하수 교환
        r = max(0.0, r + q9 + f)
        qr = r * (1 - (1 + (r / x3) ** 4) ** -0.25)
        r -= qr
        qd = max(0.0, q1 + f)
        q_out[t] = qr + qd

    return q_out


def runoff_to_volume(q_mm: np.ndarray, area_km2: float) -> np.ndarray:
    """유출고(mm/day) → 유입량(천㎥/day). 1 mm × 1 km² = 1,000 ㎥ = 1 천㎥"""
    return np.asarray(q_mm) * area_km2
