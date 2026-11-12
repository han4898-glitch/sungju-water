"""제원 카탈로그 검색과 저수지 수위 API 응답 정규화 테스트"""
import sys

import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import catalog, load_config  # noqa: E402
from engine.krc_api import extract_body, filter_reservoir, normalize, to_observed  # noqa: E402


def test_search_finds_sungju_and_gyeongcheon():
    assert catalog.search("경북 성주").iloc[0]["표준코드"] == "4784010100"
    assert catalog.search("성주댐").iloc[0]["시설명"] == "성주"
    top = catalog.search("문경 경천", 3)
    assert "4728010036" in top["표준코드"].values


def test_config_for_uses_catalog_units():
    cfg = catalog.config_for("4784010100", load_config())
    assert cfg["catchment"]["area_km2"] == 149.6          # 14,960 ha → km²
    assert cfg["reservoir"]["dead_storage"] == 38240 - 28150
    assert cfg["irrigation"]["area_ha"] == 3217


MOCK = {"header": {"resultCode": "00"}, "body": [
    {"FCLT_CD": "4784010100", "FCLT_NM": "성주", "MSRM_DT": "202609010000", "RSVR_WTLV": "180.1",
     "STRG_QTY": "25,000", "STRG_RT": "52.3", "INFLW": "0.2", "SPLY": "1.5"},
    {"FCLT_CD": "4784010100", "FCLT_NM": "성주", "MSRM_DT": "202609020000", "RSVR_WTLV": "180.0",
     "STRG_QTY": "24800", "STRG_RT": "51.6", "INFLW": "0.1", "SPLY": "1.6"},
    {"FCLT_CD": "4728010036", "FCLT_NM": "경천", "MSRM_DT": "202609020000", "RSVR_WTLV": "220",
     "STRG_QTY": "20000", "STRG_RT": "70", "INFLW": "0", "SPLY": "1"},
]}


def test_normalize_and_observed():
    df = normalize(extract_body(MOCK))
    sub = filter_reservoir(df, "4784010100", "성주")
    assert len(sub) == 2 and sub["rate"].iloc[-1] == 51.6
    obs = to_observed(sub, catalog.config_for("4784010100", load_config()))
    assert list(obs["storage"]) == [25000, 24800]
    assert abs(obs["release"].iloc[0] - 1.5 * 86.4) < 1e-9   # ㎥/s → 천㎥/일


def test_extract_body_nested():
    nested = {"response": {"body": {"items": {"item": MOCK["body"]}}}}
    assert len(extract_body(nested)) == 3


def test_poi_parsers():
    from engine import geocode as G

    assert G.poi_names("사곡") == ["사곡저수지", "사곡지", "사곡댐", "사곡못", "사곡호"]
    data = {"response": {"status": "OK", "result": {"items": [
        {"title": "사곡저수지", "address": {"parcel": "경상북도 청송군 ..."}, "point": {"x": "129.0", "y": "36.4"}},
        {"title": "사곡저수지", "address": {"parcel": "경상북도 의성군 사곡면 ..."}, "point": {"x": "128.85", "y": "36.29"}},
        {"title": "사곡초등학교", "address": {"parcel": "경상북도 의성군"}, "point": {"x": "128.9", "y": "36.3"}},
    ]}}}
    assert G.parse_vworld_place(data, G.poi_names("사곡"), "의성군") == (36.29, 128.85)
    osm = {"elements": [
        {"type": "way", "center": {"lat": 36.29, "lon": 128.86}, "bounds": {"minlat": 36.28, "maxlat": 36.30,
                                                                              "minlon": 128.84, "maxlon": 128.88}, "tags": {}},
        {"type": "way", "center": {"lat": 36.30, "lon": 128.87}, "bounds": {"minlat": 36.299, "maxlat": 36.301,
                                                                              "minlon": 128.869, "maxlon": 128.871},
         "tags": {"name": "작은못"}},
    ]}
    assert G.parse_overpass(osm, 36.29, 128.87, G.poi_names("사곡")) == (36.29, 128.86)  # 이름 없으면 가장 큰 수면


def test_missing_irrigation_area_is_estimated():
    from engine import catalog as C
    from engine.runner import load_config
    cfg = C.config_for("4782010480", load_config())  # 청도 성곡: 제원표 수혜면적 0
    assert cfg["irrigation"]["area_estimated"] and 300 < cfg["irrigation"]["area_ha"] < 2000
    cfg2 = C.config_for("4784010100", load_config())
    assert not cfg2["irrigation"]["area_estimated"] and cfg2["irrigation"]["area_ha"] == 3217


def test_rawris_parse_and_stage():
    from engine import rawris
    items = [{"fac_code": "4784010100", "equip_name": "성주", "bonbu_name": "경북", "jisa_name": "성주", "lat": 35.914,
              "lon": 128.138, "cr_date": 1794318000000, "cr_ratio": 45.15, "nr_ratio": 65.58, "per_rate": 68.85,
              "cr_level": 170.1, "comm_delay_2": 0.01},
             {"fac_code": "1", "equip_name": "끊김", "lat": 37.0, "lon": 127.0, "cr_date": 1651035000000, "cr_ratio": 50,
              "nr_ratio": 60, "per_rate": 83, "comm_delay_2": 1658}]
    df = rawris.to_frame(items)
    r = rawris.lookup("4784010100", df)
    assert r["단계"] == "관심" and r["저수율"] == 45.15 and r["시각"].year == 2026
    assert pd.isna(rawris.lookup("1", df)["저수율"])  # 통신 두절 계측기는 실시간 값 제외


def test_rawris_live():
    """PC(인터넷)에서만: RAWRIS 실시간 저수율 받기 — 전국 1,000곳 이상, 성주 포함"""
    import urllib.error

    from engine import rawris
    try:
        items = rawris.fetch(timeout=20)
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"RAWRIS 접속 불가: {e}")
    df = rawris.to_frame(items)
    assert len(df) > 1000
    r = rawris.lookup("4784010100", df)
    assert r and 35.8 < r["lat"] < 36.0 and 0 <= r["저수율"] <= 110
