"""기상청 ASOS 장기 관측(기본 1973년~)으로 저수지 기상자료를 과거로 늘립니다.
앙상블 확률 전망(ESP)의 과거 해가 11개 → 50개 안팎으로 늘고, 1994·2001년 같은 극한 가뭄이 포함되며 SPI도 안정됩니다.

  python data/extend_weather.py                 # 이미 받아 둔 저수지 전부
  python data/extend_weather.py 성주 묘곡 --start 1973-01-01
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from engine import catalog, geocode, kma  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*")
ap.add_argument("--start", default="1973-01-01")
a = ap.parse_args()
if a.names:
    codes = [catalog.search(n).iloc[0]["표준코드"] for n in a.names]
else:
    codes = [p.stem.replace("weather_kma_auto_", "") for p in (ROOT / "data").glob("weather_kma_auto_*.csv")]
coords = geocode.load_coords().set_index("표준코드")
for code in codes:
    row = catalog.get(code)
    if code in coords.index:
        lat, lon = float(coords.loc[code, "lat"]), float(coords.loc[code, "lon"])
    else:
        c = geocode.get_coords(code, row["시설명"], row["소재지"])
        lat, lon = c["lat"], c["lon"]
    print(f"{row['시설명']}({code}) {a.start}~ 받는 중…", flush=True)
    p, info = kma.update_weather(code, lat, lon, start=a.start, progress=lambda f, t: print("  ", t, flush=True))
    w = pd.read_csv(p, parse_dates=["date"], index_col="date")
    print(f"  → {info['first']} ~ {info['last']} ({len(w):,}일, 새로 {info['new_days']:,}일) · "
          f"관측소 {', '.join(s['name'] for s in info['stations'])} · 결측 강수 {w['rain'].isna().mean():.1%}", flush=True)
print("완료")
