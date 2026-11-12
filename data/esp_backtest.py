"""ESP 확률 전망 역검증 — 과거 시점(2019~2025년 4/1·10/1)에서 '그때까지의 자료만으로' 전망하고 실제와 비교

  python data/esp_backtest.py 성주 [전망일수=120]
출력: 시점별 10/50/90% 전망과 실제 저수율, 10~90% 범위 적중률(이상적 ≈80%), 중앙값 평균오차
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine import catalog, esp, krc_level  # noqa: E402
from engine.operation import PRESETS  # noqa: E402
from engine.runner import load_config, load_weather, prepare_inputs  # noqa: E402


def backtest(code: str, days: int = 120, data_dir: Path = ROOT / "data") -> pd.DataFrame:
    cfg = catalog.config_for(code, load_config())
    h = load_weather(data_dir / f"weather_kma_auto_{code}.csv")
    obs = pd.read_csv(data_dir / f"obs_rate_{code}.csv", parse_dates=["date"], index_col="date")["rate"].clip(0, 100)
    inp = prepare_inputs(h, cfg, cfg["gr4j"])
    rows = []
    for y in range(2019, pd.Timestamp.today().year):
        for md in ("04-01", "10-01"):
            st = pd.Timestamp(f"{y}-{md}")
            d = st + pd.Timedelta(days=days - 1)
            if d not in obs.index or obs.loc[:st].dropna().empty:
                continue
            cur = float(obs.loc[:st - pd.Timedelta(days=1)].dropna().iloc[-1])
            m = {k: v for k, v in esp.build_members(h, st, days).items() if k < y}
            if len(m) < 3:
                continue
            pp = esp.prepare_members(m, cfg, cfg["gr4j"])
            errs = esp.hindcast_errors(inp, cfg, obs.loc[:st - pd.Timedelta(days=1)], days, before=st)
            raw = esp.run_esp(pp, cfg, PRESETS["관행 운영"], st, days, cur)["band"].iloc[-1]
            cor = esp.run_esp(pp, cfg, PRESETS["관행 운영"], st, days, cur, errors=errs)["band"].iloc[-1]
            rows.append({"시작": st.date(), "시작 저수율": cur, "모형 10%": raw.p10, "모형 50%": raw.p50, "모형 90%": raw.p90,
                         "보정 10%": cor.p10, "보정 50%": cor.p50, "보정 90%": cor.p90, "실제": float(obs[d])})
    df = pd.DataFrame(rows).round(1)
    df["모형 적중"] = (df["실제"] >= df["모형 10%"] - 1) & (df["실제"] <= df["모형 90%"] + 1)
    df["보정 적중"] = (df["실제"] >= df["보정 10%"] - 1) & (df["실제"] <= df["보정 90%"] + 1)
    return df


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "성주"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    code = catalog.search(q).iloc[0]["표준코드"]
    df = backtest(code, days)
    print(df.to_string(index=False))
    print(f"\n10~90% 범위 적중률: 모형만 {df['모형 적중'].mean():.0%} → 관측 오차 보정 {df['보정 적중'].mean():.0%} (이상적 80%)")
    print(f"중앙값 평균오차: 모형만 {np.abs(df['모형 50%'] - df['실제']).mean():.1f}%p → 보정 {np.abs(df['보정 50%'] - df['실제']).mean():.1f}%p")
