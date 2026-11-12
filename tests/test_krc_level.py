"""농어촌공사 일별 저수율 API 테스트 (오프라인 파싱 + 키 있는 PC에서 실접속)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import krc_level as K, load_config  # noqa: E402

XML = """<?xml version="1.0" encoding="UTF-8"?>
<response><header><returnReasonCode>00</returnReasonCode><returnAuthMsg>NORMAL SERVICE.</returnAuthMsg></header>
<body><items>
<item><fac_code>4784010100</fac_code><fac_name>성주</fac_name><county>경상북도 성주군</county>
<check_date>20261001</check_date><water_level>178.5</water_level><rate>55.3</rate></item>
<item><fac_code>4784010100</fac_code><fac_name>성주</fac_name><county>경상북도 성주군</county>
<check_date>20261002</check_date><water_level>178.4</water_level><rate>54.9</rate></item>
</items><numOfRows>10</numOfRows><pageNo>1</pageNo><totalCount>2</totalCount></body></response>"""


def test_parse_xml_and_frame():
    recs, total = K.parse_xml(XML)
    assert total == 2 and recs[0]["rate"] == "55.3"
    df = K.to_frame(recs)
    assert df.loc["2026-10-02", "rate"] == 54.9 and df.loc["2026-10-01", "level"] == 178.5


def test_parse_error_response():
    err = "<OpenAPI_ServiceResponse><cmmMsgHeader><returnReasonCode>30</returnReasonCode>" \
          "<returnAuthMsg>SERVICE_KEY_IS_NOT_REGISTERED_ERROR</returnAuthMsg></cmmMsgHeader></OpenAPI_ServiceResponse>"
    with pytest.raises(RuntimeError, match="30"):
        K.parse_xml(err)


def test_normal_curve_and_status():
    idx = pd.date_range("2020-01-01", "2026-10-31", freq="D")
    rate = 60 + 20 * np.sin(2 * np.pi * idx.dayofyear / 365)
    obs = pd.DataFrame({"level": 180.0, "rate": rate}, index=idx)
    obs.loc["2026-01-01":, "rate"] -= 15  # 올해 가뭄
    st = K.status(obs)
    assert st["pct_of_normal"] < 90 and st["rate"] == pytest.approx(obs["rate"].iloc[-1])
    nc = K.normal_curve(obs, 2026)
    assert nc["years"].max() == 6


def test_observed_storage_conversion(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    cfg = load_config()
    obs = pd.DataFrame({"rate": [0.0, 50.0, 100.0]}, index=pd.date_range("2026-01-01", periods=3, name="date"))
    p = K.write_observed_storage("x", obs, cfg)
    s = pd.read_csv(p, index_col="date")["storage"].tolist()
    r = cfg["reservoir"]
    assert s[0] == r["dead_storage"] and s[2] == r["total_storage"]


def test_live_rate():
    key = K.get_key()
    if not key:
        pytest.skip("DATA_GO_KR_KEY 없음")
    d = K.fetch_rate(key, "4784010100", "2026-09-01", "2026-09-10")
    assert len(d) >= 5 and d["rate"].between(0, 120).all()


def test_no_data_code_99_is_empty():
    from engine import krc_level as K
    xml = "<response><header><resultCode>99</resultCode><resultMsg>NO_DATA</resultMsg></header></response>"
    assert K.parse_xml(xml) == ([], 0)
