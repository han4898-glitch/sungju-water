"""저수지 좌표 일괄 보정 (저수지 이름으로 다시 찾기)

  python data/refine_coords.py                 # 저장된 저수지 전체
  python data/refine_coords.py --region 경북 --min-storage 1000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import catalog, geocode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--region")
ap.add_argument("--min-storage", type=float, default=0.0)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
kw = {"region": a.region, "min_storage": a.min_storage} if a.region else {"codes": list(geocode.load_coords()["표준코드"])}
res = geocode.refine_all(catalog.load_catalog(), force=a.force, progress=lambda f, t: print(t), **kw)
print(res.to_string(index=False))
