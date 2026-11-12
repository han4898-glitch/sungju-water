"""가뭄 예측 기반 공급계획 — "성주댐 가뭄 예측해서 공급량 설정해줘" 한 번에

1) 현재 상태: RAWRIS 실시간 저수율(없으면 농어촌공사 일별 저수율), 평년 대비, 공식 농업가뭄 단계
2) 가뭄 예측: 기상청 단기예보 + 과거 각 해 기상(1973~) 앙상블(ESP) + 관측 오차 보정 → 저수율 확률 전망
3) 규칙 선택: 기본 규칙(관행·단계별·예보 연계·공사 가뭄단계)과 앙상블 최적 규칙을 같은 앙상블로 비교,
   기대 부족 심도지수(SI) + 비상 저수율 하한 위반 벌점이 가장 작은 규칙 선택
4) 공급계획: 선택 규칙으로 오늘 공급 비율과 앞으로 월별 공급량(중앙값 기상 기준)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import esp
from .operation import Scenario, band_factor, normal_series, official_stage, presets_for


def current_state(code: str, cfg: dict, obs_rate: pd.Series | None, raw: dict | None) -> dict:
    """현재 저수율 (RAWRIS 실시간이 하루 이내면 우선, 아니면 일별 관측)"""
    if raw and pd.notna(raw.get("저수율")) and pd.Timestamp.now() - pd.Timestamp(raw["시각"]) < pd.Timedelta(days=1):
        return {"pct": float(min(raw["저수율"], 100.0)), "time": str(raw["시각"]), "source": "RAWRIS 실시간",
                "normal_pct": raw.get("평년"), "pct_of_normal": raw.get("평년대비"), "stage": raw.get("단계")}
    if obs_rate is not None and obs_rate.dropna().size:
        o = obs_rate.dropna()
        pct = float(min(o.iloc[-1], 100.0))
        nr = normal_series(cfg, pd.DatetimeIndex([o.index[-1]]))
        nrv = float(nr.iloc[0]) if nr is not None else None
        ratio = pct / nrv * 100 if nrv else None
        return {"pct": pct, "time": str(o.index[-1].date()), "source": "농어촌공사 일별 저수율",
                "normal_pct": nrv, "pct_of_normal": ratio, "stage": official_stage(ratio) if ratio else "-"}
    return {"pct": 60.0, "time": "-", "source": "가정값(관측 없음)", "normal_pct": None, "pct_of_normal": None,
            "stage": "-"}


def make_plan(code: str, cfg: dict, weather: pd.DataFrame, inputs: pd.DataFrame, gr4j: dict,
              obs_rate: pd.Series | None = None, raw: dict | None = None, forecast: pd.DataFrame | None = None,
              days: int | None = None, outlook: tuple[float, float, float] | None = None, reserve_pct: float = 20.0,
              optimize: bool = True, today: pd.Timestamp | None = None, maxiter: int = 8, progress=None) -> dict:
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    if days is None:  # 기본: 다음(또는 이번) 관개기 끝(9/30)까지 — 11월에 물어도 내년 봄·여름 공급계획이 나오도록
        sep30 = pd.Timestamp(year=today.year if today.month <= 9 else today.year + 1, month=9, day=30)
        days = int((sep30 - today).days) + 1
        days = max(60, min(days, 330))
    now = current_state(code, cfg, obs_rate, raw)
    members = esp.build_members(weather, today, days, forecast)
    if len(members) < 3:
        raise ValueError("과거 기상 해가 3개 미만이라 가뭄 예측을 할 수 없습니다 (기상청 관측자료를 받아 주세요).")
    prep = esp.prepare_members(members, cfg, gr4j)
    w = esp.tercile_weights(members, today, days, outlook)
    errs = esp.hindcast_errors(inputs, cfg, obs_rate, days) if obs_rate is not None and obs_rate.dropna().size > 730 else None
    rules = dict(presets_for(cfg))
    if optimize:
        if progress:
            progress(0.05, "앙상블 최적 규칙 찾는 중…")
        o = esp.optimize_rules_esp(prep, cfg, today, days, now["pct"], w, maxiter=maxiter, progress=progress)
        rules["앙상블 최적 규칙"] = o["scenario"]
    cmp = esp.compare_rules(prep, cfg, today, days, now["pct"], w, {k: v for k, v in rules.items() if k not in presets_for(cfg)}, errs)
    cmp["점수"] = cmp["shortage_index"] + 0.05 * np.clip(reserve_pct - cmp["min_storage_pct"].astype(float), 0, None) ** 2
    best = str(cmp["점수"].astype(float).idxmin())
    sc = Scenario.from_dict(rules[best].to_dict())
    r = esp.run_esp(prep, cfg, sc, today, days, now["pct"], w, errs)

    # 월별 공급계획 (멤버별 모의의 가중 중앙값 기준)
    end = today + pd.Timedelta(days=days - 1)
    sup = pd.DataFrame({y: esp._run(prep[y], cfg, sc, today, end, now["pct"])["supply"] for y in prep})
    dem = pd.DataFrame({y: esp._run(prep[y], cfg, sc, today, end, now["pct"])["demand"] for y in prep})
    wv = w.reindex(sup.columns).fillna(0).values
    monthly = pd.DataFrame({
        "필요수량(천㎥)": (dem.resample("MS").sum() * wv).sum(axis=1),
        "공급계획(천㎥)": (sup.resample("MS").sum() * wv).sum(axis=1),
        "월말 저수율 중앙(%)": r["band"]["p50"].resample("MS").last(),
        "월말 저수율 하위10%(%)": r["band"]["p10"].resample("MS").last(),
    })
    monthly["공급 비율(%)"] = np.where(monthly["필요수량(천㎥)"] > 1,
                                   monthly["공급계획(천㎥)"] / monthly["필요수량(천㎥)"] * 100, np.nan)
    monthly.index = monthly.index.strftime("%Y-%m")

    nr_today = normal_series(cfg, pd.DatetimeIndex([today]))
    f_today = band_factor(now["pct"], sc, float(nr_today.iloc[0]) if nr_today is not None else None)
    fc_rain = float(forecast["rain"].sum()) if forecast is not None and "rain" in forecast else np.nan
    if sc.forecast_days and not np.isnan(fc_rain) and fc_rain < sc.forecast_threshold_mm:
        f_today *= 1 - sc.forecast_cut
    return {"now": now, "rule_name": best, "rule": sc, "today_factor": float(f_today), "forecast_rain_mm": fc_rain,
            "risk": r["risk"], "band": r["band"], "compare": cmp, "monthly": monthly, "members": list(members),
            "weights": w, "corrected": r["corrected"], "days": days, "reserve_pct": reserve_pct,
            "summary": summary_text(now, best, sc, f_today, r, monthly, days, len(members))}


def summary_text(now, best, sc, f_today, r, monthly, days, n) -> str:
    b = r["band"]
    end = b.index[-1]
    lines = [f"현재 저수율 {now['pct']:.1f}% ({now['source']} {now['time']})"
             + (f", 평년 대비 {now['pct_of_normal']:.0f}% → 농업가뭄 단계 '{now['stage']}'" if now.get("pct_of_normal") else ""),
             f"가뭄 예측({n}개 해 기상 앙상블, {days}일): {end:%m월 %d일} 저수율 중앙 {b['p50'].iloc[-1]:.0f}% "
             f"(하위 10% {b['p10'].iloc[-1]:.0f}% ~ 상위 10% {b['p90'].iloc[-1]:.0f}%)",
             "위험(권장 규칙의 단계 경계·유효저수율 기준): " + ", ".join(f"{k} {v:.0f}%" for k, v in r["risk"].items()),
             f"권장 운영규칙: {best} — {sc.description}",
             f"오늘 공급 비율: 필요수량의 {f_today:.0%}"]
    irr = monthly[monthly["필요수량(천㎥)"] > 1]
    if len(irr):
        lines.append("월별 공급계획: " + ", ".join(f"{m} {row['공급계획(천㎥)']:,.0f}천㎥({row['공급 비율(%)']:.0f}%)"
                                               for m, row in irr.iterrows()))
    else:
        lines.append("전망 기간 대부분이 비관개기라 관개 공급량은 0 — 하천유지유량만 방류, 이듬해 봄 저수량 확보가 목표")
    return "\n".join(lines)
