"""기상청 ASOS 일자료 수집 (공공데이터포털 '기상청_지상(종관, ASOS) 일자료 조회서비스').

사용법:
    1) https://www.data.go.kr 에서 해당 서비스 활용신청 → 일반 인증키(Decoding) 발급
    2) export DATA_GO_KR_KEY="발급받은키"
    3) python data/fetch_kma.py --stn 279 --start 2015-01-01 --end 2026-10-31

출력: data/weather_kma_<지점>.csv  (열: date, rain, tmax, tmin, rh, wind, sunshine)
앱 사이드바에서 이 파일을 선택하면 가상자료 대신 실측 기상으로 모의합니다.

※ 응답 필드명(sumRn, maxTa, ...)은 서비스 문서 기준입니다. 서비스 개편 시 FIELD_MAP만 고치면 됩니다.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
FIELD_MAP = {"tm": "date", "sumRn": "rain", "maxTa": "tmax", "minTa": "tmin",
             "avgRhm": "rh", "avgWs": "wind", "sumSsHr": "sunshine"}


def fetch_year(key: str, stn: int, start: str, end: str) -> list[dict]:
    q = {
        "serviceKey": key, "pageNo": 1, "numOfRows": 400, "dataType": "JSON",
        "dataCd": "ASOS", "dateCd": "DAY", "startDt": start.replace("-", ""),
        "endDt": end.replace("-", ""), "stnIds": stn,
    }
    with urllib.request.urlopen(URL + "?" + urllib.parse.urlencode(q), timeout=30) as r:
        body = json.loads(r.read().decode("utf-8"))
    items = body["response"]["body"]["items"]["item"]
    return items if isinstance(items, list) else [items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stn", type=int, default=279)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-10-31")
    a = ap.parse_args()
    key = os.environ.get("DATA_GO_KR_KEY")
    if not key:
        raise SystemExit("환경변수 DATA_GO_KR_KEY 에 공공데이터포털 인증키를 넣어주세요.")

    rows = []
    s, e = pd.Timestamp(a.start), pd.Timestamp(a.end)
    for y in range(s.year, e.year + 1):
        ys = max(s, pd.Timestamp(f"{y}-01-01")).strftime("%Y-%m-%d")
        ye = min(e, pd.Timestamp(f"{y}-12-31")).strftime("%Y-%m-%d")
        rows += fetch_year(key, a.stn, ys, ye)
        print(f"{y}년 수집 완료")
        time.sleep(0.3)

    df = pd.DataFrame(rows)[list(FIELD_MAP)].rename(columns=FIELD_MAP)
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c].replace("", None), errors="coerce")
    df["rain"] = df["rain"].fillna(0.0)  # 무강수일은 빈칸으로 옴
    df = df.set_index("date").asfreq("D").interpolate(limit=3)
    out = Path(__file__).parent / f"weather_kma_{a.stn}.csv"
    df.to_csv(out)
    print("저장:", out, len(df), "일")


if __name__ == "__main__":
    main()
