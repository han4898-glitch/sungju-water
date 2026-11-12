"""MCP 서버 도구를 Claude가 부르는 것과 같은 방식으로 직접 호출 (PC: mcp 패키지·관측자료가 있을 때)"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def srv():
    pytest.importorskip("mcp")
    import mcp_server
    return mcp_server


def test_tools_registered(srv):
    import asyncio
    names = {t.name for t in asyncio.run(srv.mcp.list_tools())}
    assert {"drought_supply_plan", "drought_stage_now", "esp_outlook", "find_reservoir"} <= names


def test_find_and_specs(srv):
    assert srv.find_reservoir("경북 성주")[0]["표준코드"] == "4784010100"


def test_drought_supply_plan_sungju(srv):
    """'성주댐 가뭄 예측해서 공급량 설정해줘' 와 같은 호출"""
    if not (ROOT / "data" / "weather_kma_auto_4784010100.csv").exists():
        pytest.skip("성주 기상청 관측자료 없음")
    r = srv.drought_supply_plan("성주댐")
    assert r["code"] == "4784010100" and r["recommended_rule"]
    assert 0 < r["today_supply_factor"] <= 1 and "공급" in r["summary"]
    assert r["monthly_plan"] and r["horizon_days"] >= 60
    print("\n" + r["summary"])


def test_drought_stage_now(srv):
    import urllib.error
    try:
        r = srv.drought_stage_now("성주")
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"RAWRIS 접속 불가: {e}")
    assert r.get("stage") in {"정상", "관심", "주의", "경계", "심각"} or "message" in r
