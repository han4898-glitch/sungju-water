"""웹앱 전체 실행 점검 (Streamlit AppTest — 브라우저 없이 화면 코드를 끝까지 실행)"""
from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run(query: str | None = None):
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(APP, default_timeout=300)
    at.run()
    if query is not None:
        at.sidebar.text_input[0].set_value(query).run()
    return at


def test_app_runs_default():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_run_scenario_button():
    at = _run()
    run_btn = [b for b in at.button if "모델 실행" in b.label][0]
    run_btn.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("완료" in s.value for s in at.success)


def test_app_other_reservoir_without_coords_cache():
    """좌표 캐시가 없는 저수지(문경 경천)도 오류 없이 화면이 나와야 함 (지오코딩 실패해도)"""
    at = _run("문경 경천")
    assert not at.exception, [e.value for e in at.exception]


def test_app_with_daily_rate_file():
    """일별 저수율 파일이 있을 때 현황 그래프·실시간 운영 기본값이 오류 없이 나와야 함"""
    import numpy as np
    import pandas as pd

    p = Path(APP).parent / "data" / "obs_rate_4784010100.csv"
    created = not p.exists()
    if created:
        idx = pd.date_range("2019-01-01", "2026-11-07", freq="D", name="date")
        pd.DataFrame({"level": 178.0, "rate": 55 + 20 * np.sin(idx.dayofyear / 58.0)}, index=idx).to_csv(p)
    try:
        at = _run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("평년 대비" in m.label for m in at.metric)
    finally:
        if created:
            p.unlink()


def test_app_other_reservoir_full_flow():
    """성주가 아닌 저수지(영덕 묘곡)를 입력 → 자료 자동 준비 → 모델 실행 → 운영규칙 최적화까지 화면에서 끝까지"""
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(APP, default_timeout=900)
    at.run()
    at.sidebar.text_input[0].set_value("영덕 묘곡").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("묘곡" in t.value for t in at.title)
    [b for b in at.button if "모델 실행" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("완료" in s.value for s in at.success)
    # ③ 시나리오 비교 탭의 최적화 (라벨이 정확히 같은 버튼)
    [b for b in at.button if b.label == "최적 운영규칙 찾기"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(s.value.startswith("**최적 운영규칙**") for s in at.success)
    # 실시간 운영 탭의 앙상블(ESP) 최적화
    [b for b in at.button if "앙상블 최적 운영규칙" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("앙상블 최적 규칙" in s.value for s in at.success)


def test_app_drought_tab():
    """⑤ 가뭄 대응 검증 탭: SPI 표시 → 가뭄 해 재현 → 과적합 확인 (성주, PC의 관측자료 사용)"""
    testing = pytest.importorskip("streamlit.testing.v1")
    if not (Path(APP).parent / "data" / "obs_rate_4784010100.csv").exists():
        pytest.skip("성주 일별 저수율 파일 없음")
    at = testing.AppTest.from_file(APP, default_timeout=900)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label.startswith("SPI-3") for m in at.metric)
    if not any(b.label.startswith("가뭄 해 재현 실행") for b in at.button):
        pytest.skip("관측 기상자료가 선택되지 않음")
    [b for b in at.button if b.label.startswith("가뭄 해 재현 실행")][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("재현" in s.value for s in at.success)
    [b for b in at.button if b.label.startswith("과적합 확인 실행")][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("검증 해" in str(d.value["구분"].tolist()) for d in at.dataframe if "구분" in getattr(d.value, "columns", []))


def test_app_one_shot_plan_button():
    """실시간 운영 탭: '가뭄 예측해서 공급량 설정' 버튼 → 권장 규칙·월별 공급계획 표시"""
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    btn = [b for b in at.button if b.label.startswith("가뭄 예측해서 공급량 설정")]
    if not btn:
        pytest.skip("저수지 좌표가 없어 실시간 운영 탭이 그려지지 않음")
    btn[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("권장 운영규칙" in s.value for s in at.success)
