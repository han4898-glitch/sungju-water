"""저수지 수위 API 수집 도구

  # 1) 응답 필드 확인 (처음 한 번)
  export SAFETYDATA_KEY="인증키"
  python data/fetch_reservoir.py inspect --dataset DSSP-IF-xxxxx

  # 2) 이름으로 저수지를 찾아 자료 저장 → 모델 검증용 관측자료 생성
  python data/fetch_reservoir.py fetch --name "경북 성주" --dataset DSSP-IF-xxxxx

결과:
  data/cache_raw_<표준코드>.csv            : API 원자료(표준 열로 변환)
  data/observed_storage_<표준코드>.csv     : 일 저수량·방류량 (웹앱 '모델 검증' 탭에서 선택)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import catalog, load_config  # noqa: E402
from engine.krc_api import cache_path, fetch_pages, filter_reservoir, normalize, to_observed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "fetch"])
    ap.add_argument("--dataset", help="데이터셋 ID (기본: config [api] dataset_10min)")
    ap.add_argument("--name", default="경북 성주", help="저수지 검색어")
    ap.add_argument("--pages", type=int, default=20)
    a = ap.parse_args()

    cfg = load_config()
    api = cfg["api"]
    ds = a.dataset or api.get("dataset_10min") or api.get("dataset_daily")

    if a.cmd == "inspect":
        recs = fetch_pages(api, ds, max_pages=1, num_rows=5)
        print(json.dumps(recs[:2], ensure_ascii=False, indent=2))
        from engine.krc_api import detect_fields
        if recs:
            print("\n자동 인식된 필드:", detect_fields(recs[0], api.get("fields")))
            print("틀린 것이 있으면 config/sungju.toml 의 [api.fields] 에 지정하세요.")
        return

    hit = catalog.search(a.name, limit=1)
    if hit.empty:
        raise SystemExit(f"'{a.name}'에 맞는 저수지가 없습니다.")
    row = hit.iloc[0]
    code, name = row["표준코드"], row["시설명"]
    print(f"대상: {name} ({row['본부']}/{row['지사']}, {code})")

    extra = {api["code_param"]: code} if api.get("code_param") else None
    df = normalize(fetch_pages(api, ds, extra, max_pages=a.pages), api.get("fields"))
    sub = filter_reservoir(df, code, name)
    if sub.empty:
        raise SystemExit("응답에서 해당 저수지를 찾지 못했습니다. inspect로 코드/이름 필드를 확인하세요.")
    sub.to_csv(cache_path(code, "raw"), index=False)
    rcfg = catalog.config_for(code, cfg)
    obs = to_observed(sub, rcfg)
    out = ROOT / "data" / f"observed_storage_{code}.csv"
    obs.to_csv(out)
    print(f"{len(sub)}건 → {out.name} ({len(obs)}일)")


if __name__ == "__main__":
    main()
