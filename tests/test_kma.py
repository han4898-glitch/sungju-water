"""기상청 연계 테스트 (오프라인 파싱 + 키가 있는 PC에서만 실접속)"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import kma  # noqa: E402


def test_grid_conversion_known_points():
    assert kma.latlon_to_grid(37.5665, 126.9780) == (60, 127)   # 서울시청
    x, y = kma.latlon_to_grid(35.8714, 128.6014)                 # 대구 (격자표 89, 90 부근)
    assert abs(x - 89) <= 1 and abs(y - 90) <= 1


def test_nearest_stations_for_sungju():
    st = kma.nearest_stations(35.858, 128.168, 3)
    names = {s["name"] for s in st}
    assert len(st) == 3 and abs(sum(s["weight"] for s in st) - 1) < 1e-9
    assert names & {"합천", "거창", "구미", "대구"}


def test_parse_pcp():
    assert kma.parse_pcp("강수없음") == 0
    assert kma.parse_pcp("1mm 미만") == 0.5
    assert kma.parse_pcp("2.0mm") == 2.0
    assert kma.parse_pcp("30.0~50.0mm") == 40.0
    assert kma.parse_pcp("50.0mm 이상") == 50.0


def test_latest_base():
    assert kma.latest_base(datetime(2026, 11, 8, 15, 42)) == ("20261108", "1400")
    assert kma.latest_base(datetime(2026, 11, 8, 1, 0)) == ("20261107", "2300")


def test_parse_asos_and_combine():
    a = kma.parse_asos([{"tm": "2026-11-01", "sumRn": "", "maxTa": "15.2", "minTa": "3.1", "avgRhm": "60",
                         "avgWs": "1.5", "sumSsHr": "8.0"},
                        {"tm": "2026-11-02", "sumRn": "12.5", "maxTa": "12.0", "minTa": "5.0", "avgRhm": "85",
                         "avgWs": "2.5", "sumSsHr": "1.0"}])
    b = a.copy()
    b["rain"] = [0.0, 2.5]
    b.loc["2026-11-02", "tmax"] = np.nan
    c = kma.combine_stations({1: a, 2: b}, {1: 0.75, 2: 0.25})
    assert c.loc["2026-11-01", "rain"] == 0.0
    assert c.loc["2026-11-02", "rain"] == pytest.approx(0.75 * 12.5 + 0.25 * 2.5)
    assert c.loc["2026-11-02", "tmax"] == pytest.approx(12.0)  # 결측 지점은 빠짐


def test_parse_forecast():
    items = []
    for h, pcp in (("0900", "강수없음"), ("1000", "2.0mm"), ("1100", "1mm 미만")):
        items += [{"fcstDate": "20261109", "fcstTime": h, "category": "PCP", "fcstValue": pcp},
                  {"fcstDate": "20261109", "fcstTime": h, "category": "TMP", "fcstValue": "10"},
                  {"fcstDate": "20261109", "fcstTime": h, "category": "REH", "fcstValue": "70"}]
    items.append({"fcstDate": "20261109", "fcstTime": "1500", "category": "TMX", "fcstValue": "14.0"})
    f = kma.parse_forecast(items)
    assert f.loc["2026-11-09", "rain"] == pytest.approx(2.5)
    assert f.loc["2026-11-09", "tmax"] == 14.0 and f.loc["2026-11-09", "tmin"] == 10.0


def test_live_asos_and_forecast():
    """DATA_GO_KR_KEY가 설정된 PC에서만 실제 호출"""
    key = kma.get_key()
    if not key:
        pytest.skip("기상청 키 없음")
    d = kma.fetch_asos(key, 279, "2026-10-01", "2026-10-07")
    assert len(d) == 7 and d["tmax"].notna().all()
    f, info = kma.fetch_forecast(35.858, 128.168, key)
    assert len(f) >= 2 and "rain" in f


def test_cumulative_rain():
    idx = pd.date_range("2020-01-01", "2026-06-30", freq="D")
    w = pd.DataFrame({"rain": 1.0}, index=idx)
    w.loc["2026-06-01":, "rain"] = 0.5
    c = kma.cumulative_rain(w, pd.Timestamp("2026-06-30"), windows=(30,))
    assert c.loc["최근 30일", "누적강수(mm)"] == pytest.approx(15.0)
    assert c.loc["최근 30일", "평년 대비(%)"] == pytest.approx(50.0)
    assert c.loc["최근 30일", "비교 연수"] == 6


def test_key_source_priority(monkeypatch):
    env = {"DATA_GO_KR_KEY": "abc%2Bdef"}
    monkeypatch.setattr(kma, "_env", lambda n: env.get(n))  # PC에 저장된 실제 키와 무관하게 확인
    assert kma.get_key() == ("abc+def", "datago")
    env["KMA_APIHUB_KEY"] = "hubkey"
    assert kma.get_key() == ("hubkey", "apihub")


def test_parse_sfcdd3_text():
    cols = kma.SFCDD3_COLS
    def row(tm, rn, tmax, tmin, hm):
        v = {c: "-9" for c in cols}
        v.update(TM=tm, STN="279", WS_AVG="1.5", TA_MAX=tmax, TA_MIN=tmin, HM_AVG=hm, SS_DAY="7.2", RN_DAY=rn)
        return ",".join(v[c] for c in cols)
    text = "# START7777\n# YYMMDD STN ...\n" + row("20261001", "-9.0", "24.1", "13.0", "65") + "\n" + \
        row("20261002", "12.5", "20.0", "15.2", "90") + "\n#7777END\n"
    d = kma.parse_sfcdd3(text)
    assert len(d) == 2 and d.loc["2026-10-01", "rain"] == 0.0 and d.loc["2026-10-02", "rain"] == 12.5
    assert d.loc["2026-10-02", "tmin"] == 15.2 and d.loc["2026-10-01", "sunshine"] == 7.2
