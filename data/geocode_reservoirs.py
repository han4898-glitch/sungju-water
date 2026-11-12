"""저수지 좌표 일괄 변환 → data/reservoir_coords.csv

  python data/geocode_reservoirs.py --region 경북            # 경북본부 전체
  python data/geocode_reservoirs.py --region 경북 --min-storage 1000   # 총저수량 100만㎥ 이상만
  (키가 있으면 사용: set VWORLD_KEY=... 또는 set KAKAO_REST_KEY=... / 없으면 Nominatim, 초당 1건)

한 번 만들어 두면 웹앱 배포본은 지오코딩 키 없이 지도를 보여줄 수 있습니다.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import catalog, geocode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="경북", help="본부명 (경북, 경남, 전남 …) 또는 all")
    ap.add_argument("--min-storage", type=float, default=0.0, help="총저수량 하한 (천㎥)")
    a = ap.parse_args()
    cat = catalog.load_catalog()
    if a.region != "all":
        cat = cat[cat["본부"] == a.region]
    cat = cat[cat["총저수량"] >= a.min_storage].sort_values("총저수량", ascending=False)
    done = set(geocode.load_coords()["표준코드"])
    todo = cat[~cat["표준코드"].isin(done)]
    print(f"대상 {len(cat)}개 중 {len(todo)}개 변환")
    ok = 0
    for i, (_, r) in enumerate(todo.iterrows(), 1):
        c = geocode.get_coords(r["표준코드"], r["시설명"], r["소재지"])
        ok += c is not None
        print(f"[{i}/{len(todo)}] {r['시설명']:<8} {r['소재지']:<28} → "
              + (f"{c['lat']:.4f}, {c['lon']:.4f} ({c['source']})" if c else "실패"))
    print(f"완료: {ok}/{len(todo)}")


if __name__ == "__main__":
    main()
