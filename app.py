"""성주댐 용수공급 운영 시나리오 분석 웹앱 (Streamlit)

실행: streamlit run app.py
흐름: ① 모의조건 설정 → ② 모델 실행(GR4J + 논 수요량 + pywr) → ③ 대시보드 → ④ 시나리오 비교 → ⑤ 검증
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import PRESETS, Scenario, load_config, load_weather, prepare_inputs, presets_for, run_scenario
from engine import catalog
from engine.calibrate import calibrate_gr4j, evaluate_params
from engine import krc_api
from engine.kpi import KPI_LABELS
from engine.pywr_engine import PYWR_AVAILABLE
from engine import dem as demlib
from engine import geocode
from engine import vworld
from engine import kma
from engine import krc_level
from engine.realtime import project, today_decision
from engine.optimize import optimize_rules
from engine import esp, drought, spi, rawris, plan
from engine.operation import attach_normal_curve, normal_series, official_stage
from engine.dem_plot import map_figure, surface_figure

ROOT = Path(__file__).parent
DATA = ROOT / "data"
COLORS = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626", "#0891b2"]

st.set_page_config(page_title="저수지 용수관리", page_icon="💧", layout="wide")

# ------------------------------------------------------------------ 상태 --
BASE_CFG = load_config()
cfg = BASE_CFG  # 사이드바에서 저수지를 고르면 제원표 기반 설정으로 바뀜
if "runs" not in st.session_state:
    st.session_state.runs = {}  # 이름 → RunResult
st.session_state.setdefault("opt_by_code", {})  # 저수지별 최적 운영규칙
st.session_state.setdefault("gr4j_by_code", {})  # 저수지별 GR4J·공급계수·운영상한


@st.cache_data(show_spinner=False)
def cached_weather(path: str, _mtime: float = 0.0) -> pd.DataFrame:
    return load_weather(path)


@st.cache_data(show_spinner="유입량·필요수량 계산 중…")
def cached_inputs(path: str, code: str, x1: float, x2: float, x3: float, x4: float, mtime: float = 0.0,
                  release_scale: float = 1.0, lat: float | None = None) -> pd.DataFrame:
    c = catalog.config_for(code, BASE_CFG)
    if lat is not None:  # 증발산량(ET0) 계산 위도·표고를 선택한 저수지 기준으로
        c["station"] = {**c["station"], "lat": float(lat)}
    if c["reservoir"].get("full_level"):
        c["station"]["elevation_m"] = float(c["reservoir"]["full_level"])
    return prepare_inputs(cached_weather(path, mtime), c,
                          {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "release_scale": release_scale})


@st.cache_data(show_spinner="운영 전망 모의 중…", max_entries=16)
def cached_project(weather_, fc_, cfg_, cur_, days_, assume_, day_, scs_, g_, engine_):
    return project(weather_, fc_, cfg_, cur_, days_, assume_, day_, scenarios=scs_, gr4j_params=g_, engine=engine_)


@st.cache_data(show_spinner="과거 기상 앙상블 준비 중 (처음 한 번)…", max_entries=6)
def cached_esp_members(weather_: pd.DataFrame, fc_: pd.DataFrame | None, start_: pd.Timestamp, days_: int,
                       cfg_: dict, g_: dict):
    m_ = esp.build_members(weather_, start_, days_, fc_)
    return m_, esp.prepare_members(m_, cfg_, g_)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_esp_compare(_prep, key_: tuple, cfg_: dict, start_, days_: int, cur_: float, wts_, extra_: dict, _errs):
    return esp.compare_rules(_prep, cfg_, start_, days_, cur_, wts_, extra_, _errs)


@st.cache_data(show_spinner=False, max_entries=2)
def cached_rawris(_slot: int):
    """RAWRIS 전국 계측 저수지 실시간 저수율 (10분 단위 캐시). 실패하면 (None, 이유)"""
    try:
        return rawris.snapshot()
    except Exception as e:  # noqa: BLE001
        return None, f"RAWRIS 받기 실패: {e}"


@st.cache_data(show_spinner="과거 재예측 오차 계산 중 (저수지·기간마다 처음 한 번)…", max_entries=8)
def cached_hindcast(inputs_: pd.DataFrame, cfg_: dict, obs_rate_: pd.Series, days: int) -> dict:
    return esp.hindcast_errors(inputs_, cfg_, obs_rate_, days)


@st.cache_resource(show_spinner="DEM 지형 분석 중 (저수지마다 처음 한 번, 수십 초)…", max_entries=8)
def cached_dem_analysis(files_key: tuple, code: str, lat: float, lon: float, target_km2: float,
                        full_level: float | None, dam_height: float | None):
    dcfg = BASE_CFG.get("dem", {})
    grid = demlib.load_dem(crs_override=dcfg.get("crs") or None, max_cells=int(dcfg.get("max_cells", 600000)),
                           center_lonlat=(lon, lat), radius_m=demlib.window_radius_m(target_km2))
    if grid is None:
        return None
    approx = demlib.lonlat_to_xy(lon, lat, grid.crs) if grid.crs else None
    return demlib.analyze(grid, approx, target_km2, float(dcfg.get("search_radius_m", 6000)),
                          full_level, dam_height)


def get_dem_analysis(cfg_: dict):
    """data/dem/ 의 국토지리정보원 DEM(전국 90m 등)에서 현재 저수지 주변을 잘라 분석"""
    files = demlib.find_dem_files()
    r_ = cfg_["reservoir"]
    if not files:
        return None, "no_file"
    if r_.get("lat") is None:
        return None, "no_coord"
    key = tuple((f.name, f.stat().st_mtime) for f in files)
    try:
        return cached_dem_analysis(key, r_.get("code", ""), r_["lat"], r_["lon"], cfg_["catchment"]["area_km2"],
                                   r_.get("full_level"), r_.get("dam_height")), "ok"
    except Exception as e:  # 좌표계·범위 오류 등
        return None, f"error: {e}"


@st.cache_data(show_spinner="저수지 위치(주소 → 좌표) 찾는 중…")
def cached_coords(code: str, name: str, address: str, _ver: float):
    return geocode.get_coords(code, name, address)


def vworld_texture_fn():
    """브이월드 키가 있으면 3D 지형에 입힐 항공영상 함수, 없으면 None"""
    key = vworld.get_key()
    if not key:
        return None
    ref = BASE_CFG.get("vworld", {}).get("referer")
    return lambda lon, lat: vworld.texture_for_grid(key, lon, lat, ref)


def show_3d(dem_a, ext_, lv, below_, title, exag=2.0, height=560, max_px=180, key="t"):
    tex = vworld_texture_fn()
    use = st.checkbox("항공사진 입히기 (브이월드)", value=tex is not None, disabled=tex is None, key=f"tex_{key}",
                      help=None if tex else "VWORLD_KEY(브이월드 인증키)를 설정하면 사용할 수 있습니다.")
    try:
        fig = surface_figure(dem_a, ext_, lv, below_, title, exag, height, max_px, tex if use else None)
    except Exception as e:  # 키·도메인 오류, 네트워크 차단 등
        st.warning(f"브이월드 영상을 불러오지 못해 표고 색으로 표시합니다 — {e}")
        fig = surface_figure(dem_a, ext_, lv, below_, title, exag, height, max_px, None)
    st.plotly_chart(fig, width="stretch")
    if use and tex is not None:
        st.caption("배경 영상: 브이월드(국토교통부) 위성영상 · 지형: 국토지리정보원 수치표고모델")


DEM_HELP = """**국토지리정보원 DEM 넣는 법** (국토지리정보원 DEM은 실시간 API가 아니라 파일로 제공됩니다)
1. [국토정보플랫폼](http://map.ngii.go.kr) 로그인 → 국토정보맵 → 공개 '수치표고모델(DEM)'
2. **90m DEM(전국)** 을 내려받으면 어느 저수지를 검색해도 분석됩니다 (저수지 주변만 잘라 읽음).
   - 5m DEM(도엽별)을 쓰려면 분석할 유역 전체를 덮는 도엽을 모두 받아야 합니다.
3. 받은 파일(.img/.tif 또는 x y z 텍스트)을 프로젝트의 `data/dem/` 폴더에 넣고 새로고침
4. 좌표계 정보가 없는 파일이면 `config/sungju.toml` 의 `[dem] crs` 에 좌표계(예: EPSG:5186)를 적습니다."""


# --------------------------------------------------------------- 사이드바 --
with st.sidebar:
    st.header("저수지 선택")
    q = st.text_input("이름·지역으로 검색", "경북 성주", help="예: 경북 성주, 문경 경천, 경산 문천, 영천 저수지")
    hits = catalog.search(q, limit=15)
    if hits.empty:
        st.warning("검색 결과가 없습니다. 성주댐으로 진행합니다.")
        code = BASE_CFG["reservoir"]["code"]
    else:
        labels = {r_["표준코드"]: f"{r_['시설명']} ({r_['본부']}·{r_['지사']}) {r_['총저수량']:,.0f}천㎥" for _, r_ in hits.iterrows()}
        code = st.selectbox("검색 결과", list(labels), format_func=labels.get)
    cfg = catalog.config_for(code, BASE_CFG)
    RES_NAME = cfg["reservoir"]["name"]
    RAW_DF, RAW_SRC = cached_rawris(int(time.time() // 600))
    if RAW_DF is not None and not st.session_state.get("rawris_coords_done"):
        st.session_state["rawris_coords_done"] = True
        try:
            if getattr(rawris, "update_coords", None) and rawris.update_coords(RAW_DF):
                cached_coords.clear()  # 좌표 캐시만 비움 (모형·전망 계산 캐시는 유지)
        except Exception:
            pass
    RAW_NOW = rawris.lookup(code, RAW_DF) if RAW_DF is not None else None
    # 평년 저수율 곡선 (농어촌공사 일별 저수율 전체 기간) → 평년 대비 규칙·공식 가뭄단계
    if krc_level.obs_path(code).exists():
        attach_normal_curve(cfg, pd.read_csv(krc_level.obs_path(code), parse_dates=["date"], index_col="date")["rate"])
    # 위치: 좌표 캐시(data/reservoir_coords.csv) → 없으면 소재지 주소를 지오코딩해 저장
    _ver = geocode.COORDS_CSV.stat().st_mtime if geocode.COORDS_CSV.exists() else 0.0
    coords = cached_coords(code, RES_NAME, cfg["reservoir"]["location"], _ver)
    if coords:
        cfg["reservoir"]["lat"], cfg["reservoir"]["lon"] = coords["lat"], coords["lon"]
    # 주소 중심 좌표면 저수지 이름(POI)으로 한 번 더 찾아 바로잡음
    _refined = getattr(geocode, "REFINED_SOURCES", None)  # 서버 재시작 전(구버전 모듈)에도 화면이 깨지지 않도록
    if _refined and coords and not str(coords.get("source", "")).startswith(_refined) \
            and "POI없음" not in str(coords.get("source", "")):
        with st.spinner("저수지 위치를 이름으로 다시 찾는 중…"):
            geocode.refine_coords(code, RES_NAME, cfg["reservoir"]["location"])  # 성공·실패 모두 기록됨
        st.rerun()
    # DEM이 있으면 실제 저수지(만수위 수면) 중심으로 좌표를 바로잡아 저장 (직접 입력한 좌표는 그대로 둠)
    if coords and coords.get("source") != "manual" and not str(coords.get("source", "")).startswith("DEM"):
        _da, _ds = get_dem_analysis(cfg)
        if (_da is not None and hasattr(_da, "lake_center_lonlat")
                and abs(_da.basin_area_km2 / _da.target_area_km2 - 1) <= 0.15):
            _lc = _da.lake_center_lonlat(cfg["reservoir"].get("full_level"))
            if _lc:
                geocode.save_coord(code, RES_NAME, _lc[1], _lc[0], "DEM(만수위 수면 중심)")
                st.rerun()
    with st.expander("위치 " + (f"({coords['lat']:.4f}, {coords['lon']:.4f} · {coords['source']})" if coords else "(찾지 못함)")):
        st.caption("주소의 리(里) 중심이라 댐과 수 km 다를 수 있습니다. DEM 분석이 유역면적·표고로 댐 지점을 다시 찾습니다. "
                   "직접 고치려면 지도앱에서 댐 위치를 확인해 입력하세요.")
        m1_, m2_ = st.columns(2)
        new_lat = m1_.number_input("위도", 33.0, 39.0, float(coords["lat"]) if coords else 36.0, 0.001, format="%.4f")
        new_lon = m2_.number_input("경도", 124.0, 132.0, float(coords["lon"]) if coords else 128.5, 0.001, format="%.4f")
        if st.button("이 좌표로 저장", width="stretch"):
            geocode.save_coord(code, RES_NAME, new_lat, new_lon, "manual")
            st.rerun()

    st.header("① 모의조건")
    auto_name = f"weather_kma_auto_{code}.csv"
    # 처음 고른 저수지는 기상청 관측·농어촌공사 일별 저수율을 자동으로 받아 바로 분석되게 함 (저수지당 한 번 시도)
    _tried = st.session_state.setdefault("auto_prep_tried", set())
    if code not in _tried:
        _tried.add(code)
        if kma.get_key() and cfg["reservoir"].get("lat") is not None and not (DATA / auto_name).exists():
            with st.spinner(f"{RES_NAME}: 인근 기상청 ASOS 관측자료 받는 중… (처음 한 번, 1분 안팎)"):
                try:
                    _, st.session_state["kma_info"] = kma.update_weather(code, cfg["reservoir"]["lat"], cfg["reservoir"]["lon"])
                    st.session_state["wfile"] = auto_name
                except Exception as e:
                    st.warning(f"기상청 자료 자동 받기 실패 — {e}")
        if krc_level.get_key() and not krc_level.obs_path(code).exists():
            with st.spinner(f"{RES_NAME}: 농어촌공사 일별 저수율 받는 중… (처음 한 번)"):
                try:
                    krc_level.update(code, cfg)
                except LookupError:
                    pass  # 수위 관측 없는 저수지 — 저수지 현황 탭에서 안내
                except Exception as e:
                    st.warning(f"일별 저수율 자동 받기 실패 — {e}")
    if (DATA / auto_name).exists() and st.session_state.get("wfile_code") != code:
        st.session_state["wfile"] = auto_name  # 저수지를 바꾸면 그 저수지의 관측 기상으로
    st.session_state["wfile_code"] = code
    if kma.get_key() and cfg["reservoir"].get("lat") is not None:
        if st.button("☁ 기상청 관측자료 갱신 (인근 ASOS 3개 지점)", width="stretch"):
            bar = st.progress(0.0, "기상청 자료 요청 중…")
            try:
                _, info_k = kma.update_weather(code, cfg["reservoir"]["lat"], cfg["reservoir"]["lon"],
                                               progress=lambda f, t: bar.progress(f, t))
                bar.empty()
                st.session_state["kma_info"] = info_k
                st.session_state["wfile"] = auto_name
                st.rerun()
            except Exception as e:
                bar.empty()
                st.error(f"기상청 자료를 받지 못했습니다 — {e}")
        if (DATA / auto_name).exists():
            _w0 = pd.read_csv(DATA / auto_name, usecols=["date"], nrows=1)["date"].iloc[0]
            if str(_w0) > "1980":
                if st.button("📚 장기 관측(1973년~) 받기 — 앙상블·SPI 확대 (5~10분)", width="stretch"):
                    bar = st.progress(0.0, "기상청 장기 자료 요청 중…")
                    try:
                        _, info_k = kma.update_weather(code, cfg["reservoir"]["lat"], cfg["reservoir"]["lon"],
                                                       start="1973-01-01", progress=lambda f, t: bar.progress(f, t))
                        bar.empty()
                        st.session_state["kma_info"] = info_k
                        st.rerun()
                    except Exception as e:
                        bar.empty()
                        st.error(f"장기 자료를 받지 못했습니다 — {e}")
            else:
                st.caption(f"기상 관측 {str(_w0)[:4]}년~ (장기 자료)")
    files = sorted(p.name for p in DATA.glob("weather_*.csv")
                   if not p.name.startswith("weather_kma_auto_") or p.name == auto_name)
    if st.session_state.get("wfile") not in files:
        st.session_state["wfile"] = auto_name if auto_name in files else files[0]
    wfile = st.selectbox("기상자료", files, key="wfile",
                         help="weather_sample.csv = 가상자료 · weather_kma_auto_* = 기상청 ASOS 관측(인근 3개 지점 거리가중)")
    if wfile == "weather_sample.csv":
        st.caption("⚠️ 데모용 가상 기상자료입니다." + ("" if kma.get_key() else
                   " 기상청 인증키를 넣으면 기상청 관측자료로 바꿀 수 있습니다."))
    elif wfile == auto_name and "kma_info" in st.session_state:
        ki = st.session_state["kma_info"]
        st.caption("기상청 ASOS: " + ", ".join(f"{s['name']} {s['km']:.0f}km(가중 {s['weight']:.0%})"
                                             for s in ki["stations"]) + f" · ~{ki['last']}")
    W_MTIME = (DATA / wfile).stat().st_mtime
    weather = cached_weather(str(DATA / wfile), W_MTIME)
    dmin, dmax = weather.index.min().date(), weather.index.max().date()
    default_start = max(dmin, pd.Timestamp("2025-10-01").date())
    period = st.date_input("모의기간", (default_start, dmax), min_value=dmin, max_value=dmax)
    if len(period) != 2:
        st.stop()
    start, end = (pd.Timestamp(d).strftime("%Y-%m-%d") for d in period)

    initial_pct = st.slider("시작 유효저수율 (%)", 0, 100, 70)
    inflow_scale = st.slider("유입량 배율", 0.3, 1.5, 1.0, 0.05, help="가뭄 강도·기후변화 민감도")
    demand_scale = st.slider("필요수량 배율", 0.5, 1.5, 1.0, 0.05, help="재배면적·작부 변화")
    engine_opts = ["pywr", "reference"] if PYWR_AVAILABLE else ["reference"]
    engine = st.radio("계산 엔진", engine_opts, horizontal=True,
                      help="pywr = 선형계획 기반 수자원 시스템 모델, reference = 교차검증용 물수지")
    if not PYWR_AVAILABLE:
        st.warning("pywr 미설치 → reference 엔진으로 실행합니다.")

    g = st.session_state.gr4j_by_code.setdefault(code, dict(cfg["gr4j"]))
    calibrated_ = str(code) in BASE_CFG.get("calibrated", {}) or str(code) in getattr(catalog, "load_saved_calibration", dict)()
    with st.expander("유입량 모형(GR4J)·운영 매개변수" + (" — 실측 보정값" if calibrated_ else "")):
        if calibrated_:
            st.caption("농어촌공사 일별 저수율과 기상청 ASOS로 보정한 값입니다 (config [calibrated] 또는 data/calibrated_params.json).")
        else:
            st.caption("⚠️ 아직 보정 전 기본값입니다. '④ 모델 검증'에서 보정 후 '적용'을 누르면 이 저수지 값으로 저장됩니다.")
        g["x1"] = st.number_input("X1 생산저류고 (mm)", 10.0, 2000.0, float(g["x1"]), key=f"x1_{code}")
        g["x2"] = st.number_input("X2 지하수 교환 (mm/d)", -10.0, 5.0, float(g["x2"]), key=f"x2_{code}")
        g["x3"] = st.number_input("X3 추적저류고 (mm)", 5.0, 500.0, float(g["x3"]), key=f"x3_{code}")
        g["x4"] = st.number_input("X4 단위도 기저시간 (d)", 0.5, 6.0, float(g["x4"]), key=f"x4_{code}")
        g["release_scale"] = st.number_input("공급계수 (실제 공급 / 이론 필요수량)", 0.2, 1.5,
                                             float(g.get("release_scale", 1.0)), 0.01, key=f"rs_{code}")
        g["cap_frac"] = st.number_input("운영 상한 저수율 (0~1, 초과분 방류)", 0.5, 1.0,
                                        float(g.get("cap_frac", 1.0)), 0.01, key=f"cap_{code}")
    cfg.setdefault("operation", {})["ops_cap_frac"] = float(g.get("cap_frac", 1.0))

    st.divider()
    r = cfg["reservoir"]
    st.caption(f"**{r['name']}** · {r['location']}  \n총저수량 {r['total_storage']:,.0f}천㎥ · "
               f"유효 {r['effective_storage']:,.0f}천㎥ · 유역 {cfg['catchment']['area_km2']:,.1f}km² · "
               f"수혜면적 {cfg['irrigation']['area_ha']:,.0f}ha  \n제원: 한국농어촌공사 저수지 제원 정보(2026-09-07)")

inputs = cached_inputs(str(DATA / wfile), code, g["x1"], g["x2"], g["x3"], g["x4"], W_MTIME,
                       float(g.get("release_scale", 1.0)), cfg["reservoir"].get("lat"))

st.title(f"💧 {RES_NAME} 저수지 용수공급 운영 시나리오 분석")
st.caption("GR4J 유입량 모형 + 논 담수심 수요량 모형 + pywr 저수지 운영 모델 · 경북 농업용 저수지 가뭄 대응")

tab_status, tab_live, tab_run, tab_cmp, tab_val, tab_dry, tab_info = st.tabs(
    ["저수지 현황", "실시간 운영", "② 모의 실행·대시보드", "③ 시나리오 비교", "④ 모델 검증", "⑤ 가뭄 대응 검증", "모델 개요"])


# ---------------------------------------------------------------- 도우미 --
def kpi_row(k: dict, base: dict | None = None):
    keys = ["volumetric_reliability_pct", "shortage_index", "worst_10day_shortage_pct", "total_deficit_k_m3",
            "min_storage_pct", "end_storage_pct", "max_consecutive_deficit_days", "spill_k_m3"]
    lower_better = {"shortage_index", "worst_10day_shortage_pct", "total_deficit_k_m3", "max_consecutive_deficit_days", "spill_k_m3"}
    cols = st.columns(4)
    for i, key in enumerate(keys):
        label, unit = KPI_LABELS[key]
        v = k[key]
        delta = None
        if base is not None:
            d = v - base[key]
            delta = f"{d:+,.1f}"
        cols[i % 4].metric(f"{label}", f"{v:,.1f} {unit}", delta,
                           delta_color="inverse" if key in lower_better else "normal")


def storage_chart(runs: dict, sc_for_curves: Scenario | None, height=380):
    fig = go.Figure()
    if sc_for_curves and sc_for_curves.curves_pct:
        names = ["관심", "주의", "경계", "심각"]
        for i, c in enumerate(sc_for_curves.curves_pct):
            fig.add_hline(y=c, line_dash="dot", line_color="gray",
                          annotation_text=f"{names[i] if i < 4 else ''} {c:.0f}%", annotation_position="top left")
    for i, (name, rr) in enumerate(runs.items()):
        s = rr.series
        fig.add_trace(go.Scatter(x=s.index, y=s["storage_pct"], name=name, mode="lines",
                                 line=dict(color=COLORS[i % len(COLORS)], width=2)))
    fig.update_layout(height=height, yaxis_title="유효저수율 (%)", yaxis_range=[0, 105],
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08), hovermode="x unified")
    return fig


# ============================================================== 저수지 현황 ==
with tab_status:
    row = catalog.get(code)
    info = catalog.summary(row)
    st.subheader(f"{info['시설명']} 저수지 · {info['본부/지사']}")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("총저수량", f"{info['총저수량(천㎥)']:,.0f} 천㎥")
    a2.metric("유효저수량", f"{info['유효저수량(천㎥)']:,.0f} 천㎥")
    a3.metric("유역면적", f"{info['유역면적(km²)'] or 0:,.1f} km²")
    irr_ha_ = float(cfg["irrigation"]["area_ha"])
    irr_est_ = bool(cfg["irrigation"].get("area_estimated"))
    a4.metric("수혜면적" + (" (추정)" if irr_est_ else ""), f"{irr_ha_:,.0f} ha")
    ratio = (row["유역면적"] / irr_ha_) if irr_ha_ > 0 else float("nan")
    st.caption(f"유역배율(유역면적÷수혜면적) = {ratio:.1f} — 관개 저수지는 3 이상이면 용수공급 여건이 양호한 편으로 봅니다.")
    if irr_est_:
        st.warning(f"제원표에 이 저수지의 수혜면적이 0(미기재)입니다. 같은 본부 저수지들의 '수혜면적÷유효저수량' 중앙값으로 "
                   f"**{irr_ha_:,.0f} ha**로 추정해 필요수량을 계산합니다. 관측 저수율이 있으면 "
                   f"'모델 검증' 탭의 공급계수 보정으로 실제 공급량에 맞출 수 있습니다.")
    with st.expander("전체 제원"):
        st.table(pd.Series({k: v for k, v in info.items()}, name="값").astype(str))

    st.markdown("#### 위치")
    if cfg["reservoir"].get("lat") is not None:
        pts_ = pd.DataFrame({"lat": [cfg["reservoir"]["lat"]], "lon": [cfg["reservoir"]["lon"]],
                             "size": [250], "color": ["#2563eb"]})
        da_loc, _ = get_dem_analysis(cfg)
        dam_txt = ""
        if da_loc is not None and da_loc.outlet_lonlat:
            pts_ = pd.concat([pts_, pd.DataFrame({"lat": [da_loc.outlet_lonlat[1]], "lon": [da_loc.outlet_lonlat[0]],
                                                  "size": [120], "color": ["#dc2626"]})], ignore_index=True)
            dam_txt = f" · 빨강 = 댐(DEM 유역 출구 {da_loc.outlet_lonlat[1]:.4f}, {da_loc.outlet_lonlat[0]:.4f})"
        st.map(pts_, latitude="lat", longitude="lon", size="size", color="color", zoom=12)
        st.caption(f"파랑 = {RES_NAME} 저수지 ({cfg['reservoir']['lat']:.4f}, {cfg['reservoir']['lon']:.4f} · "
                   f"{coords['source'] if coords else ''}){dam_txt} · 소재지: {info['소재지']}")
    else:
        st.warning("이 저수지의 좌표를 찾지 못했습니다. 사이드바 '위치'에서 직접 입력하세요.")
    with st.expander("저수지 위치 일괄 보정 (저수지 이름으로 다시 찾기)"):
        st.caption("주소(리·읍면 중심)로 잡힌 좌표를 브이월드 장소검색 → OpenStreetMap 수면 순서로 저수지 자체 위치로 바꿉니다. "
                   "직접 입력(manual)·DEM으로 잡은 좌표는 유지합니다.")
        cc1_, cc2_ = st.columns(2)
        run_saved = cc1_.button("저장된 저수지 전체 다시 찾기", width="stretch")
        run_region = cc2_.button(f"{row['본부']} 본부 주요 저수지 미리 찾기 (총저수량 100만㎥ 이상)", width="stretch")
        if run_saved or run_region:
            bar_ = st.progress(0.0, "시작…")
            kw_ = ({"codes": list(geocode.load_coords()["표준코드"])} if run_saved
                   else {"region": row["본부"], "min_storage": 1000.0})
            res_ = geocode.refine_all(catalog.load_catalog(), force=False,
                                      progress=lambda f, t: bar_.progress(min(f, 1.0), t), **kw_)
            bar_.empty()
            st.dataframe(res_, width="stretch", hide_index=True)
            st.caption("반영하려면 페이지를 새로고침하세요.")

    st.markdown("#### 실시간 저수율 · 농업가뭄 단계 (농어촌공사 RAWRIS)")
    if RAW_NOW and pd.notna(RAW_NOW.get("저수율")):
        rw1_, rw2_, rw3_, rw4_ = st.columns(4)
        rw1_.metric("현재 저수율", f"{RAW_NOW['저수율']:.1f} %", f"{RAW_NOW['시각']:%m-%d %H:%M} 기준", delta_color="off")
        rw2_.metric("평년 저수율", f"{RAW_NOW['평년']:.1f} %")
        rw3_.metric("평년 대비", f"{RAW_NOW['평년대비']:.1f} %")
        _stg = RAW_NOW["단계"]
        rw4_.metric("농업가뭄 단계", {"정상": "🟢 정상", "관심": "🟡 관심", "주의": "🟠 주의", "경계": "🔴 경계",
                                  "심각": "⚫ 심각"}.get(_stg, _stg))
        st.caption("단계 기준(평년 대비 저수율): 70% 초과 정상 · 60~70 관심 · 50~60 주의 · 40~50 경계 · 40 이하 심각 "
                   "(ADMS 농업가뭄관리시스템 저수율지도와 같은 기준) · 자료: " + RAW_SRC
                   + " · 이 단계 기준은 ③의 '공사 가뭄단계 연동' 운영규칙에도 쓰입니다.")
    elif RAW_DF is None:
        st.caption(f"RAWRIS 실시간 자료를 받지 못했습니다 ({RAW_SRC}). 아래 일별 저수율을 사용합니다.")
    else:
        st.caption("이 저수지는 RAWRIS 실시간 계측 대상이 아니거나 통신이 끊겨 있습니다.")

    st.markdown("#### 일별 저수율 (한국농어촌공사 저수지 수위정보)")
    obs_p = krc_level.obs_path(code)
    b1_, b2_ = st.columns([1, 3])
    if b1_.button("일별 저수율 받기·갱신", disabled=krc_level.get_key() is None,
                  help=None if krc_level.get_key() else "공공데이터포털 인증키(DATA_GO_KR_KEY)가 필요합니다."):
        try:
            with st.spinner("농어촌공사 저수지 수위정보 받는 중… (처음엔 2015년부터 받아 1분 정도 걸립니다)"):
                _, inf_ = krc_level.update(code, cfg)
            st.success(f"{inf_['first']} ~ {inf_['last']} 자료 (새로 {inf_['new_days']}일)")
        except LookupError as e:
            st.info(str(e))
        except Exception as e:
            st.error(f"받지 못했습니다 — {e}")
    if krc_level.get_key() is None:
        b2_.caption("공공데이터포털에서 「한국농어촌공사_농촌용수 저수지 수위정보 조회」를 활용신청하고 "
                    "set_kma_key.bat 으로 일반 인증키(DATA_GO_KR_KEY)를 저장하세요.")
    if obs_p.exists():
        obs_ = pd.read_csv(obs_p, parse_dates=["date"], index_col="date")
        stt_ = krc_level.status(obs_)
        if stt_:
            m1_, m2_, m3_, m4_ = st.columns(4)
            m1_.metric("최신 저수율", f"{stt_['rate']:.1f} %", f"{stt_['date']:%Y-%m-%d}", delta_color="off")
            m2_.metric("평년 대비", "—" if pd.isna(stt_["pct_of_normal"]) else f"{stt_['pct_of_normal']:.0f} %",
                       None if pd.isna(stt_["normal"]) else f"평년 {stt_['normal']:.1f}%", delta_color="off")
            m3_.metric("7일 변화", "—" if pd.isna(stt_["chg7"]) else f"{stt_['chg7']:+.1f} %p")
            m4_.metric("30일 변화", "—" if pd.isna(stt_["chg30"]) else f"{stt_['chg30']:+.1f} %p")
            yr_ = stt_["date"].year
            nc_ = krc_level.normal_curve(obs_, yr_)
            days_x = pd.date_range(f"{yr_}-01-01", f"{yr_}-12-31")
            fig = go.Figure()
            if len(nc_):
                d_ = np.minimum(days_x.dayofyear, 366)
                fig.add_trace(go.Scatter(x=days_x, y=nc_.loc[d_, "max"], line=dict(width=0), showlegend=False,
                                         hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=days_x, y=nc_.loc[d_, "min"], fill="tonexty", line=dict(width=0),
                                         fillcolor="rgba(148,163,184,0.25)", name="과거 범위(최소~최대)"))
                fig.add_trace(go.Scatter(x=days_x, y=nc_.loc[d_, "normal"], name="평년",
                                         line=dict(color="#64748b", dash="dash")))
            prev_ = obs_.loc[str(yr_ - 1), "rate"] if str(yr_ - 1) in obs_.index.year.astype(str) else None
            if prev_ is not None and len(prev_):
                fig.add_trace(go.Scatter(x=prev_.index + pd.DateOffset(years=1), y=prev_, name=f"{yr_ - 1}년",
                                         line=dict(color="#ea580c", width=1.5)))
            cur_ = obs_.loc[str(yr_), "rate"]
            fig.add_trace(go.Scatter(x=cur_.index, y=cur_, name=f"{yr_}년", line=dict(color="#2563eb", width=3)))
            for c_, nm_ in zip([60, 40, 25], ["관심", "주의", "경계"]):
                fig.add_hline(y=c_, line_dash="dot", line_color="#94a3b8", annotation_text=nm_,
                              annotation_position="top left")
            fig.update_layout(height=360, yaxis_title="저수율 (%)", hovermode="x unified",
                              margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12),
                              xaxis=dict(tickformat="%m월"))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"자료: 한국농어촌공사 저수지 수위정보 ({obs_.index.min():%Y-%m-%d} ~ {obs_.index.max():%Y-%m-%d}) · "
                       "평년 = 올해를 뺀 같은 날짜 평균. 이 자료는 '실시간 운영'의 현재 저수율과 '모델 검증'의 관측값으로 자동 사용됩니다.")

    with st.expander("10분 수위 (재난안전데이터 공유플랫폼, 선택 — 일별 저수율만으로도 충분합니다)", expanded=False):
        st.caption("일별 저수율보다 촘촘한 10분 간격 수위가 필요할 때만 씁니다. 재난안전데이터 공유플랫폼 인증키"
                   "(SAFETYDATA_KEY)와 config/sungju.toml [api] 의 데이터셋 ID가 있어야 합니다.")
        api = BASE_CFG.get("api", {})
        raw_cache = krc_api.cache_path(code, "raw")
        sd_ready = bool(krc_api.get_key() and (api.get("dataset_10min") or api.get("dataset_daily")))
        if st.button("API에서 최신 자료 받기", disabled=not sd_ready):
            try:
                with st.spinner("저수지 수위 API 호출 중…"):
                    extra = {api["code_param"]: code} if api.get("code_param") else None
                    df_api = krc_api.normalize(
                        krc_api.fetch_pages(api, api.get("dataset_10min") or api.get("dataset_daily"), extra, max_pages=5),
                        api.get("fields"))
                    sub = krc_api.filter_reservoir(df_api, code, RES_NAME)
                if sub.empty:
                    st.warning("응답에서 이 저수지를 찾지 못했습니다. `python data/fetch_reservoir.py inspect` 로 필드를 확인하세요.")
                else:
                    sub.to_csv(raw_cache, index=False)
                    st.success(f"{len(sub)}건 저장")
            except Exception as e:
                st.error(f"API 호출 실패: {e}")
                st.caption("인증키(SAFETYDATA_KEY)와 config [api] 의 데이터셋 ID를 확인하세요.")
        if raw_cache.exists():
            live = pd.read_csv(raw_cache, parse_dates=["time"])
            last = live.iloc[-1]
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("관측 시각", str(last["time"])[:16])
            if "rate" in live: b2.metric("저수율", f"{last['rate']:.1f} %")
            if "level" in live: b3.metric("저수위", f"EL.{last['level']:.2f} m")
            if "storage" in live: b4.metric("저수량", f"{last['storage']:,.0f}")
            ycol = "rate" if "rate" in live else ("level" if "level" in live else None)
            if ycol:
                fig = go.Figure(go.Scatter(x=live["time"], y=live[ycol], mode="lines", line=dict(color="#2563eb")))
                fig.update_layout(height=300, yaxis_title="저수율 (%)" if ycol == "rate" else "저수위 (EL.m)",
                                  margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch")
            if "rate" in live:
                st.caption("이 저수율을 사이드바의 '시작 유효저수율'에 넣으면 지금 상태에서 출발하는 운영 시나리오를 볼 수 있습니다.")
        else:
            st.caption("아직 받은 10분 자료가 없습니다." + ("" if sd_ready else " (키·데이터셋 ID 미설정으로 버튼 비활성)"))

    st.markdown("#### 지형(DEM) 기반 유역·저수구역 분석")
    dem_a, dem_state = get_dem_analysis(cfg)
    if dem_state == "no_file":
        st.info(DEM_HELP)
    elif dem_state == "no_coord":
        st.info("좌표가 없어 DEM 분석을 할 수 없습니다. 사이드바 '위치'에서 좌표를 입력하세요.")
    elif dem_a is None and "범위 밖" in str(dem_state) and cfg["reservoir"].get("lat") is not None:
        need_ = demlib.sheets_for(float(cfg["reservoir"]["lon"]), float(cfg["reservoir"]["lat"]),
                             demlib.window_radius_m(cfg["catchment"]["area_km2"]) * 0.5)
        have_ = sorted({p_.stem for p_ in (ROOT / "data" / "dem").glob("*.img")})
        st.info(f"이 저수지는 지금 가진 DEM({', '.join(have_) or '없음'}) 범위 밖이라 3D·유역 분석을 건너뜁니다. "
                f"지도·모의·운영 분석은 그대로 됩니다.\n\n"
                f"분석하려면 국토정보플랫폼에서 **{need_[0]}** 도엽 DEM(90m)을 받아 `data/dem/`에 넣으세요"
                + (f" (유역이 걸치는 인접 도엽: {', '.join(need_[1:])})" if len(need_) > 1 else "") + ".")
    elif dem_a is None:
        st.error(f"DEM 분석 실패 — {dem_state}")
    else:
        g_ = dem_a.grid
        c1_, c2_, c3_, c4_ = st.columns(4)
        c1_.metric("DEM 유역면적", f"{dem_a.basin_area_km2:,.1f} km²",
                   f"{(dem_a.basin_area_km2 / dem_a.target_area_km2 - 1) * 100:+.1f}% (제원표 대비)", delta_color="off")
        ls_ = demlib.level_storage_from_cfg(cfg)
        full_ext, _ = dem_a.extent(cfg["reservoir"]["full_level"])
        full_ha = full_ext.sum() * g_.cell ** 2 / 1e4
        c2_.metric("만수위 수면적 (DEM)", f"{full_ha:,.0f} ha",
                   f"제원표 {cfg['reservoir'].get('full_area_ha') or 0:,.0f} ha", delta_color="off")
        c3_.metric("DEM 격자", f"{g_.cell:,.0f} m", f"{g_.shape[1]}×{g_.shape[0]}", delta_color="off")
        if dem_a.outlet_lonlat:
            c4_.metric("댐(출구) 위치", f"{dem_a.outlet_lonlat[1]:.4f}N", f"{dem_a.outlet_lonlat[0]:.4f}E", delta_color="off")
        if abs(dem_a.basin_area_km2 / dem_a.target_area_km2 - 1) > 0.15:
            st.warning("DEM에서 찾은 유역면적이 제원표와 15% 이상 다릅니다. DEM이 유역 전체를 덮는지, "
                       "config의 댐 대략 좌표(lat/lon)와 [dem] search_radius_m를 확인하세요.")
        lv_min = float(cfg["reservoir"]["dead_level"]); lv_max = float(cfg["reservoir"]["full_level"]) + 8
        lv = st.slider("수위 (EL.m)", lv_min, lv_max, float(cfg["reservoir"]["full_level"]), 0.5)
        ext_, below_ = dem_a.extent(lv)
        info_ = f"수위 EL.{lv:.1f}m · 수면적 {ext_.sum() * g_.cell ** 2 / 1e4:,.0f} ha"
        if ls_:
            info_ += f" · 저수량 {float(ls_.storage(lv)):,.0f} 천㎥ (수위-저수량 곡선)"
        view_ = st.radio("보기", ["3D 지형", "2D 지도"], horizontal=True, key="dem_view_status")
        if view_ == "3D 지형":
            ex_ = st.slider("수직 과장", 1.0, 5.0, 2.0, 0.5, key="exag_status")
            show_3d(dem_a, ext_, lv, below_, info_, ex_, key="status")
        else:
            st.plotly_chart(map_figure(dem_a, ext_, below_, info_), width="stretch")
        notes = ["빨간 선 = DEM으로 추출한 유역 경계, 파란 면 = 해당 수위의 수면 범위."]
        if below_:
            notes.append(f"수위가 DEM 측량 당시 수면(EL.{dem_a.survey_level:.1f}m)보다 낮아 호소 바닥 지형을 알 수 없으므로 "
                         "측량 당시 수면을 연한 파랑으로 표시합니다.")
        notes.append("자료: 국토지리정보원 수치표고모델 (" + ", ".join(g_.source[:3]) + (" 외" if len(g_.source) > 3 else "") + ")")
        st.caption(" ".join(notes))

    st.markdown(f"#### {row['본부']} 본부 주요 저수지와 비교")
    cat = catalog.load_catalog()
    peers = cat[cat["본부"] == row["본부"]].nlargest(12, "총저수량")
    if code not in peers["표준코드"].values:
        peers = pd.concat([peers, cat[cat["표준코드"] == code]])
    fig = go.Figure(go.Bar(
        x=peers["시설명"], y=peers["유효저수량"],
        marker_color=["#ea580c" if c == code else "#94a3b8" for c in peers["표준코드"]],
        text=(peers["유역면적"] / peers["수혜면적"].replace(0, float("nan"))).round(1).astype(str),
        hovertemplate="%{x}<br>유효저수량 %{y:,.0f}천㎥<br>유역배율 %{text}<extra></extra>"))
    fig.update_layout(height=320, yaxis_title="유효저수량 (천㎥)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")


# ============================================================== 실시간 운영 ==
@st.cache_data(ttl=3600, show_spinner="기상청 단기예보 받는 중…")
def cached_forecast(lat: float, lon: float, hour_key: str):
    return kma.fetch_forecast(lat, lon)


with tab_live:
    st.subheader(f"{RES_NAME} 저수지 실시간 운영 판단")
    k1, k2, k3 = st.columns(3)
    k1.markdown(("🟢" if vworld.get_key() else "⚪") + " 브이월드 (지도·위치·항공영상)")
    _kk = kma.get_key()
    k2.markdown(("🟢" if _kk else "⚪") + " 기상청 " + ({"apihub": "(API허브)", "datago": "(공공데이터포털)"}.get(_kk[1]) if _kk else ""))
    k3.markdown(("🟢" if krc_level.get_key() else "⚪") + " 농어촌공사 일별 저수율 (공공데이터포털)")
    r_ = cfg["reservoir"]
    if r_.get("lat") is None:
        st.warning("저수지 좌표가 없어 예보 격자를 정할 수 없습니다. 사이드바 '위치'에서 좌표를 입력하세요.")
    else:
        # 현재 저수율: 농어촌공사 일별 저수율 → 10분 수위 API 저장값 → 직접 입력
        raw_cache_ = krc_api.cache_path(code, "raw")
        api_pct = None
        if krc_level.obs_path(code).exists():
            _st = krc_level.status(pd.read_csv(krc_level.obs_path(code), parse_dates=["date"], index_col="date"))
            if _st:
                api_pct = float(min(_st["rate"], 100.0))
                st.caption(f"현재 저수율 기본값: 농어촌공사 관측 {_st['date']:%Y-%m-%d} {_st['rate']:.1f}% "
                           f"(평년 대비 {_st['pct_of_normal']:.0f}%)" if not pd.isna(_st["pct_of_normal"]) else "")
        elif raw_cache_.exists():
            live_ = pd.read_csv(raw_cache_, parse_dates=["time"])
            if "rate" in live_ and live_["rate"].notna().any():
                api_pct = float(live_["rate"].dropna().iloc[-1])
        if RAW_NOW and pd.notna(RAW_NOW.get("저수율")) and \
                pd.Timestamp.now() - pd.Timestamp(RAW_NOW["시각"]) < pd.Timedelta(days=1):
            api_pct = float(min(RAW_NOW["저수율"], 100.0))  # 10분 실시간 값이 가장 최신
            st.caption(f"현재 저수율 기본값: RAWRIS 실시간 {RAW_NOW['시각']:%m-%d %H:%M} {RAW_NOW['저수율']:.1f}% · "
                       f"평년 대비 {RAW_NOW['평년대비']:.0f}% → 농업가뭄 단계 **{RAW_NOW['단계']}**")
        c1, c2, c3 = st.columns(3)
        cur_pct = c1.number_input("현재 유효저수율 (%)", 0.0, 100.0, api_pct if api_pct is not None else 50.0, 1.0,
                                  help="농어촌공사 일별 저수율을 받아두면 최신 관측값이 기본으로 들어갑니다.")
        days_ = c2.slider("전망 기간 (일)", 7, 90, 30)
        assume_ = c3.radio("예보 이후 가정", ["무강우", "평년"], horizontal=True,
                           help="무강우 = 비가 오지 않는다고 보는 보수적 가정, 평년 = 관측 이력의 같은 날짜 평균")
        fc_df, fc_info = None, None
        if kma.get_key():
            try:
                fc_df, fc_info = cached_forecast(r_["lat"], r_["lon"], pd.Timestamp.now().strftime("%Y%m%d%H"))
            except Exception as e:
                st.warning(f"기상청 단기예보를 받지 못했습니다 — {e}")
        else:
            st.info("기상청 인증키(API허브 KMA_APIHUB_KEY 또는 공공데이터포털 DATA_GO_KR_KEY)를 설정하면 단기예보가 자동으로 반영됩니다. "
                    "지금은 예보 없이 '예보 이후 가정'만으로 전망합니다.")
        today_ = pd.Timestamp.today().normalize()
        fc_future = fc_df.loc[today_:] if fc_df is not None and len(fc_df) else None
        fc_rain = float(fc_future["rain"].sum()) if fc_future is not None else 0.0
        if fc_future is not None and len(fc_future):
            st.markdown(f"**기상청 단기예보** (발표 {fc_info['base']}, 격자 {fc_info['grid'][0]},{fc_info['grid'][1]}) — "
                        f"향후 {len(fc_future)}일 강수 합계 **{fc_rain:.1f} mm**")
            st.dataframe(fc_future.rename(columns={"rain": "강수(mm)", "tmax": "최고(℃)", "tmin": "최저(℃)",
                                                   "rh": "습도(%)", "wind": "풍속(m/s)"}).round(1).T,
                         width="stretch")
        st.markdown("#### 누적 강수 현황 (가뭄 판단)")
        try:
            cum_ = kma.cumulative_rain(weather)
            cc_ = st.columns(len(cum_))
            for col_, (lab_, row_) in zip(cc_, cum_.iterrows()):
                pct_ = row_["평년 대비(%)"]
                col_.metric(lab_, f"{row_['누적강수(mm)']:,.0f} mm",
                            None if pd.isna(pct_) else f"평년 대비 {pct_:.0f}%",
                            delta_color="off")
            st.caption(f"기준일 {weather.index.max():%Y-%m-%d} · 평년 = 같은 기간 과거 {int(cum_['비교 연수'].max())}개년 평균 · "
                       "평년 대비 70% 미만이 이어지면 가뭄 단계 진입을 검토합니다."
                       + (" (가상 기상자료 기준)" if wfile == "weather_sample.csv" else ""))
        except Exception as e:
            st.caption(f"누적 강수 계산 불가: {e}")
        try:
            _scs = dict(presets_for(cfg))
            if code in st.session_state.opt_by_code:
                _scs["최적 운영규칙"] = Scenario.from_dict(st.session_state.opt_by_code[code]["scenario"].to_dict())
            if code in st.session_state.get("esp_opt", {}):
                _scs["앙상블 최적 규칙"] = Scenario.from_dict(st.session_state.esp_opt[code].to_dict())
            pr = cached_project(weather, fc_future, cfg, float(cur_pct), int(days_), assume_, today_.date(),
                                _scs, g, engine)
        except Exception as e:
            pr = None
            st.error(f"전망 모의 실패: {e}")
        if pr:
            dem_today = float(pr["inputs"]["demand"].iloc[0])
            st.markdown("#### 오늘의 공급 판단")
            cols_ = st.columns(len(pr["runs"]))
            for col_, (nm_, rr_) in zip(cols_, pr["runs"].items()):
                _nr = normal_series(cfg, pd.DatetimeIndex([today_]))
                d_ = today_decision(cur_pct, fc_rain, dem_today, rr_.scenario,
                                    float(_nr.iloc[0]) if _nr is not None else None)
                col_.metric(nm_, f"{d_['factor']:.0%} 공급", f"단계: {d_['stage']}", delta_color="off")
                col_.caption(f"필요 {dem_today:,.0f} → 공급 {d_['supply_k_m3']:,.0f} 천㎥/일"
                             + (" · 예보 강수 부족으로 추가 절감" if d_["forecast_cut"] else ""))
            if dem_today < 0.01:
                st.caption("오늘은 관개기(5/1~9/10)가 아니어서 필요수량이 0입니다. 전망은 하천유지유량·증발만 반영합니다.")
            fig = go.Figure()
            src_ = pr["source"]
            fc_days = src_[src_ == "예보"].index
            if len(fc_days):
                fig.add_vrect(x0=fc_days.min() - pd.Timedelta(hours=12), x1=fc_days.max() + pd.Timedelta(hours=12),
                              fillcolor="rgba(37,99,235,0.08)", line_width=0, annotation_text="기상청 예보",
                              annotation_position="top left")
            for i, (nm_, rr_) in enumerate(pr["runs"].items()):
                fig.add_trace(go.Scatter(x=rr_.series.index, y=rr_.series["storage_pct"], name=nm_,
                                         line=dict(color=COLORS[i], width=2)))
            fig.add_trace(go.Bar(x=pr["weather"].index, y=pr["weather"]["rain"], name="강수(mm)", yaxis="y2",
                                 marker_color=["#2563eb" if s_ == "예보" else "rgba(100,116,139,0.4)" for s_ in src_]))
            fig.update_layout(height=380, yaxis_title="유효저수율 (%)", hovermode="x unified",
                              yaxis2=dict(title="강수 (mm)", overlaying="y", side="right", autorange="reversed",
                                          showgrid=False),
                              margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width="stretch")
            tbl_ = pd.DataFrame({nm_: {"기말 저수율(%)": rr_.kpis["end_storage_pct"],
                                       "최저 저수율(%)": rr_.kpis["min_storage_pct"],
                                       "공급량(천㎥)": rr_.kpis["total_supply_k_m3"],
                                       "부족량(천㎥)": rr_.kpis["total_deficit_k_m3"]}
                                 for nm_, rr_ in pr["runs"].items()}).T
            st.dataframe(tbl_.style.format("{:,.1f}"), width="stretch")
            st.caption(f"기상: 관측 이력 = {wfile} · 예보 = 기상청 단기예보 · 그 이후 = {assume_} 가정. "
                       "저수율이 낮을수록, 예보 강수가 적을수록 제한급수의 효과가 커집니다.")

        # ---------------- 장기 확률 전망 (ESP: 기상청 과거 관측으로 만든 앙상블) ----------------
        st.divider()
        st.markdown("#### 📈 장기 저수율 확률 전망 (기상청 과거 관측 앙상블, ESP)")
        st.caption("오늘까지는 실제 관측, 1~3일은 기상청 단기예보, 그 이후는 '과거 각 해의 같은 날짜 기상'을 이어 붙여 "
                   "해마다 하나씩 '있을 법한 미래'를 만들고, 모두 모의해 저수율의 확률 분포를 봅니다 "
                   "(미국 기상청·K-water 장기 유량전망과 같은 방식). 기상청 1·3개월 전망 확률을 넣으면 과거 해에 가중치를 줍니다.")
        if wfile == "weather_sample.csv":
            st.warning("지금 기상자료가 가상자료입니다. 기상청 관측자료(weather_kma_auto_*)를 쓰면 실제 과거 기상으로 전망합니다.")
        e1_, e2_ = st.columns([1, 2])
        esp_days_ = e1_.slider("전망 기간 (일)", 30, 270, 150, 10, key="esp_days")
        with e2_:
            st.caption("기상청 강수 전망 (적음 / 비슷 / 많음 확률, %) — 기상청 날씨누리 '1개월·3개월 전망' 값을 넣으세요. 모르면 33/34/33")
            t1_, t2_, t3_ = st.columns(3)
            p_lo = t1_.number_input("적음", 0, 100, 33, 5, key="esp_lo")
            p_mid = t2_.number_input("비슷", 0, 100, 34, 5, key="esp_mid")
            p_hi = t3_.number_input("많음", 0, 100, 33, 5, key="esp_hi")
        probs_ = None if (p_lo, p_mid, p_hi) == (33, 34, 33) else (p_lo, p_mid, p_hi)
        try:
            members_, prep_cached_ = cached_esp_members(weather, fc_future, today_, esp_days_, cfg, g)
        except Exception as e:
            members_, prep_cached_ = {}, {}
            st.error(f"앙상블을 만들지 못했습니다 — {e}")
        if len(members_) < 3:
            st.info("과거 관측이 3개 해 이상 있어야 확률 전망을 만들 수 있습니다. 사이드바에서 기상청 관측자료를 받아 주세요.")
        else:
            prep_ = prep_cached_
            wts_ = esp.tercile_weights(members_, today_, esp_days_, probs_)
            errs_ = None
            if krc_level.obs_path(code).exists() and wfile != "weather_sample.csv":
                use_corr_ = st.checkbox("관측 오차 보정 (과거 재예측 오차 반영 — 권장)", True, key="esp_corr",
                                        help="과거 여러 시점에서 그때 관측 저수율로 시작해 실제 기상으로 다시 예측한 값과 실제 저수율의 차이를 "
                                             "앙상블에 더합니다. 성주·묘곡 2019~2025 역검증에서 10~90% 범위 적중률이 약 50% → 70~86%로 개선.")
                if use_corr_:
                    _ob = pd.read_csv(krc_level.obs_path(code), parse_dates=["date"], index_col="date")["rate"]
                    errs_ = cached_hindcast(inputs, cfg, _ob, esp_days_)
            rules_ = dict(presets_for(cfg))
            if code in st.session_state.opt_by_code:
                rules_["최적 운영규칙"] = st.session_state.opt_by_code[code]["scenario"]
            esp_opt_ = st.session_state.setdefault("esp_opt", {}).get(code)
            if esp_opt_:
                rules_["앙상블 최적 규칙"] = esp_opt_
            sel_rule_ = st.selectbox("전망에 쓸 운영규칙", list(rules_),
                                     index=list(rules_).index("앙상블 최적 규칙") if esp_opt_ else 1, key="esp_rule")
            er_ = esp.run_esp(prep_, cfg, Scenario.from_dict(rules_[sel_rule_].to_dict()), today_, esp_days_, cur_pct, wts_, errs_)
            b_ = er_["band"]
            fig_e = go.Figure()
            for y_, col_ in er_["members"].items():
                fig_e.add_trace(go.Scatter(x=col_.index, y=col_, mode="lines", line=dict(color="rgba(100,116,139,0.25)", width=1),
                                           name=f"{y_}년 기상", showlegend=False, hovertemplate=f"{y_}년 기상: %{{y:.1f}}%<extra></extra>"))
            fig_e.add_trace(go.Scatter(x=b_.index, y=b_["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_e.add_trace(go.Scatter(x=b_.index, y=b_["p10"], fill="tonexty", fillcolor="rgba(37,99,235,0.18)",
                                       line=dict(width=0), name="10~90% 범위"))
            fig_e.add_trace(go.Scatter(x=b_.index, y=b_["p50"], line=dict(color="#2563eb", width=3), name="중앙값(50%)"))
            for c_, nm_ in zip(rules_[sel_rule_].curves_pct or [60, 40, 25], ["관심", "주의", "경계"]):
                fig_e.add_hline(y=c_, line_dash="dot", line_color="gray", annotation_text=f"{nm_} {c_:.0f}%",
                                annotation_position="top left", annotation_font_size=11)
            fig_e.update_layout(height=420, yaxis_title="유효저수율 (%)", hovermode="x unified",
                                margin=dict(l=10, r=20, t=40, b=10), legend=dict(orientation="h", y=1.08))
            fig_e.update_yaxes(range=[0, 100], autorange=False)
            fig_e.update_xaxes(range=[b_.index[0] - pd.Timedelta(days=2), b_.index[-1] + pd.Timedelta(days=2)],
                               autorange=False)
            st.plotly_chart(fig_e, width="stretch")
            rk_ = st.columns(len(er_["risk"]))
            for col_, (k_, v_) in zip(rk_, er_["risk"].items()):
                col_.metric(k_, f"{v_:.0f} %")
            end_ = b_.index[-1]
            st.caption(f"{len(members_)}개 해({min(members_)}~{max(members_)}) 기상으로 만든 앙상블 · "
                       f"{end_:%m/%d} 저수율: 10% {b_['p10'].iloc[-1]:.0f}% · 중앙 {b_['p50'].iloc[-1]:.0f}% · 90% {b_['p90'].iloc[-1]:.0f}%"
                       + (f" · 기상청 전망 가중치 적용(적음 {p_lo}/비슷 {p_mid}/많음 {p_hi})" if probs_ else "")
                       + (" · 파란 범위 = 관측 오차 보정 후, 회색 선 = 각 해 기상의 모형 결과" if er_["corrected"] else ""))
            st.markdown("**운영규칙별 앙상블 기대 성능** (불확실성을 고려한 비교 — 기대 SI가 작을수록 좋음)")
            _extra = {k_: v_ for k_, v_ in rules_.items() if k_ not in presets_for(cfg)}
            cmp_ = cached_esp_compare(prep_, (code, W_MTIME, esp_days_, probs_, bool(errs_), str(g)), cfg, today_,
                                      esp_days_, round(cur_pct, 1), wts_, _extra, errs_)
            show_ = cmp_[["shortage_index", "worst_10day_shortage_pct", "volumetric_reliability_pct",
                          "min_storage_pct", "end_storage_pct"] + list(er_["risk"])].astype(float)
            show_.columns = ["기대 SI", "최악10일 부족률(%)", "공급 만족률(%)", "최저 저수율(%)", "기말 저수율(%)"] + list(er_["risk"])
            st.dataframe(show_.style.format("{:,.1f}").highlight_min(subset=["기대 SI"], color="#dcfce7"), width="stretch")
            if st.button("🎯 앙상블 최적 운영규칙 찾기 (1~4분)", key="esp_opt_btn"):
                bar_e = st.progress(0.0, "시작…")
                oe_ = esp.optimize_rules_esp(prep_, cfg, today_, esp_days_, cur_pct, wts_,
                                             progress=lambda f, t: bar_e.progress(f, t))
                bar_e.empty()
                st.session_state.esp_opt[code] = oe_["scenario"]
                st.rerun()
            if esp_opt_:
                st.success(f"**앙상블 최적 규칙**: {esp_opt_.description} — 과거 모든 해의 기상에 대해 평균적으로 부족이 가장 덜 심한 규칙입니다. "
                           "위 '오늘의 공급 판단'과 ② 모의 실행에도 추가됩니다.")

        # ---------------- 한 번에: 가뭄 예측 → 공급계획 ----------------
        st.divider()
        st.markdown("#### 🎯 가뭄 예측 → 공급계획 (한 번에)")
        st.caption("현재 저수율(RAWRIS) → 기상청 예보 + 과거 기상 앙상블로 가뭄 예측 → 운영규칙 비교·최적화 → "
                   "권장 규칙, 오늘 공급 비율, 다음 관개기(9/30까지) 월별 공급계획을 한 번에 만듭니다. "
                   "MCP에서는 '성주댐 가뭄 예측해서 공급량 설정해줘'로 같은 계산을 합니다.")
        pl1_, pl2_ = st.columns([1, 1])
        pl_res_ = pl1_.slider("지킬 비상 유효저수율 (%)", 0, 40, 20, 5, key="plan_res")
        if pl2_.button("가뭄 예측해서 공급량 설정 (1~2분)", type="primary", width="stretch", key="plan_btn"):
            bar_p = st.progress(0.0, "시작…")
            _obs_p = (pd.read_csv(krc_level.obs_path(code), parse_dates=["date"], index_col="date")["rate"]
                      if krc_level.obs_path(code).exists() else None)
            try:
                st.session_state.setdefault("plan_res_by_code", {})[code] = plan.make_plan(
                    code, cfg, weather, inputs, g, _obs_p, RAW_NOW, fc_future, None, probs_ if "probs_" in dir() else None,
                    pl_res_, True, progress=lambda f, t: bar_p.progress(min(f, 1.0), t))
            except Exception as e:
                st.error(f"공급계획을 만들지 못했습니다 — {e}")
            bar_p.empty()
        pr_ = st.session_state.get("plan_res_by_code", {}).get(code)
        if pr_:
            st.success(pr_["summary"].replace("\n", "  \n"))
            pm_ = pr_["monthly"]
            st.dataframe(pm_.style.format("{:,.1f}", na_rep="-"), width="stretch")
            st.caption(f"권장 규칙: {pr_['rule_name']} — {pr_['rule'].description} · 앙상블 {min(pr_['members'])}~{max(pr_['members'])}년 "
                       f"{len(pr_['members'])}개 해 · 관측 오차 보정 {'적용' if pr_['corrected'] else '미적용'} · "
                       "공급계획 = 각 해 기상으로 모의한 공급량의 가중 평균")
            st.session_state.setdefault("esp_opt", {})[code] = pr_["rule"]  # 오늘의 판단·모의 실행에도 추가


# =============================================================== 모의 실행 ==
with tab_run:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("운영 시나리오")
        _opt_ = st.session_state.opt_by_code.get(code)
        _eopt_ = st.session_state.get("esp_opt", {}).get(code)
        preset = st.selectbox("기본 시나리오", list(presets_for(cfg)) + (["최적 운영규칙"] if _opt_ else [])
                              + (["앙상블 최적 규칙"] if _eopt_ else []) + ["사용자 정의"])
        if preset == "사용자 정의":
            n = st.number_input("저수율 구간 수", 0, 4, 3)
            curves, factors = [], []
            defaults_c, defaults_f = [60, 40, 25, 15], [1.0, 0.9, 0.75, 0.55, 0.4]
            factors.append(st.number_input("기준선 위 공급비율", 0.0, 1.5, 1.0, 0.05, key="f0"))
            for i in range(int(n)):
                cc1, cc2 = st.columns(2)
                curves.append(cc1.number_input(f"경계 {i+1} (%)", 0.0, 100.0, float(defaults_c[i]), 5.0, key=f"c{i}"))
                factors.append(cc2.number_input(f"아래 공급비율 {i+1}", 0.0, 1.5, defaults_f[i + 1], 0.05, key=f"f{i+1}"))
            fd = st.number_input("예보 참조 일수 (0=미사용)", 0, 30, 0)
            ft = st.number_input("예보 강수 기준 (mm)", 0.0, 200.0, 20.0)
            fcut = st.number_input("예보 부족 시 추가 절감", 0.0, 0.5, 0.1, 0.05)
            env = st.checkbox("하천유지유량 공급", True)
            sc = Scenario("사용자 정의", "사용자 입력 규칙", curves, factors, int(fd), ft, fcut, env)
        elif preset == "앙상블 최적 규칙":
            sc = Scenario.from_dict(_eopt_.to_dict())
            st.caption(sc.description + " (실시간 운영 탭의 확률 전망에서 찾은 규칙)")
        elif preset == "최적 운영규칙":
            sc = Scenario.from_dict(_opt_["scenario"].to_dict())
            st.caption(sc.description + " (③ 탭에서 찾은 규칙)")
        else:
            sc = Scenario.from_dict(PRESETS[preset].to_dict())
            st.caption(sc.description)
        run_name = st.text_input("결과 이름", f"{RES_NAME} · {sc.name} | 유입×{inflow_scale:g} 수요×{demand_scale:g} 초기{initial_pct}%")
        if st.button("▶ 모델 실행", type="primary", width="stretch"):
            try:
                with st.spinner(f"{engine} 엔진으로 모의 중…"):
                    res = run_scenario(inputs, cfg, sc, start, end, initial_pct, demand_scale, inflow_scale, engine)
                st.session_state.runs[run_name] = res
                st.session_state.last = run_name
                st.success(f"완료 · {len(res.series)}일 모의 ({res.engine})")
            except Exception as e:  # 사용자 입력 오류를 화면에 표시
                st.error(f"실행 실패: {e}")

    with c2:
        last = st.session_state.get("last")
        if not last or last not in st.session_state.runs:
            st.info("왼쪽에서 시나리오를 고르고 **모델 실행**을 누르세요.")
        else:
            res = st.session_state.runs[last]
            s = res.series
            st.subheader(last)
            kpi_row(res.kpis)

    if st.session_state.get("last") in st.session_state.runs:
        res = st.session_state.runs[st.session_state.last]
        s = res.series
        st.divider()
        # 상호작용: 기간·변수 선택
        f1, f2 = st.columns([2, 1])
        rng = f1.slider("표시 기간", s.index.min().to_pydatetime(), s.index.max().to_pydatetime(),
                        (s.index.min().to_pydatetime(), s.index.max().to_pydatetime()), format="YYYY-MM-DD")
        vars_all = {"필요수량": "demand", "공급량": "supply", "유입량": "inflow", "부족량": "deficit",
                    "월류": "spill", "하천유지": "env_flow", "수면증발": "evap"}
        chosen = f2.multiselect("유량 변수", list(vars_all), ["필요수량", "공급량", "유입량"])
        v = s.loc[rng[0]:rng[1]]

        g1, g2 = st.columns([3, 2])
        with g1:
            st.plotly_chart(storage_chart({last: type(res)(res.scenario, res.engine, v, res.kpis)}, res.scenario),
                            width="stretch")
            fig = go.Figure()
            for i, name in enumerate(chosen):
                fig.add_trace(go.Scatter(x=v.index, y=v[vars_all[name]], name=name, mode="lines",
                                         line=dict(color=COLORS[i % len(COLORS)])))
            fig.add_trace(go.Bar(x=v.index, y=v["rain"], name="강수량(mm)", yaxis="y2",
                                 marker_color="rgba(100,116,139,0.45)"))
            fig.update_layout(height=340, yaxis_title="천㎥/일", hovermode="x unified",
                              yaxis2=dict(title="강수 (mm)", overlaying="y", side="right", autorange="reversed", showgrid=False),
                              margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")
        with g2:
            st.markdown("**공간 분포: 모의 저수위의 수면 범위**")
            dem_a, dem_state = get_dem_analysis(cfg)
            ls_ = demlib.level_storage_from_cfg(cfg)
            if dem_a is not None and ls_ is not None:
                day = st.select_slider("날짜", options=list(v.index), value=v["storage"].idxmin(),
                                       format_func=lambda d: d.strftime("%Y-%m-%d"))
                lv = float(ls_.level(v.at[day, "storage"]))
                ext_, below_ = dem_a.extent(lv)
                ttl_ = f"{day:%Y-%m-%d} 유효저수율 {v.at[day, 'storage_pct']:.0f}% · EL.{lv:.1f}m"
                if st.toggle("3D로 보기", value=True, key="dem_view_run"):
                    show_3d(dem_a, ext_, lv, below_, ttl_, 2.0, height=420, max_px=140, key="run")
                else:
                    st.plotly_chart(map_figure(dem_a, ext_, below_, ttl_, height=420, max_px=300), width="stretch")
                st.caption("기본값은 최저 저수율 날짜. 수위는 제원표 만수위·사수위·만수면적으로 만든 수위-저수량 곡선으로 환산"
                           + (" · 연한 파랑 = DEM 측량 당시 수면(그보다 낮은 바닥 지형은 DEM에 없음)" if below_ else ""))
            else:
                r = cfg["reservoir"]
                if dem_state == "no_file":
                    st.caption("data/dem/ 에 국토지리정보원 DEM을 넣으면 여기에 수면 범위 지도가 표시됩니다 ('저수지 현황' 탭 안내 참고).")
                if r.get("lat") is not None:
                    st.map(pd.DataFrame({"lat": [r["lat"]], "lon": [r["lon"]]}), zoom=12)
                    st.caption("저수지 위치 (DEM이 없어 수면 범위 대신 위치만 표시)")
                else:
                    st.info(r["location"])
            # 월별 공급/부족
            m = v[["demand", "supply", "deficit"]].resample("MS").sum()
            m = m[m["demand"] > 0]
            if len(m):
                fig = go.Figure()
                fig.add_bar(x=m.index.strftime("%Y-%m"), y=m["supply"], name="공급", marker_color="#2563eb")
                fig.add_bar(x=m.index.strftime("%Y-%m"), y=m["deficit"], name="부족", marker_color="#dc2626")
                fig.update_layout(barmode="stack", height=300, title="월별 공급·부족 (천㎥)",
                                  margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, width="stretch")
        with st.expander("일별 결과표 / CSV 내려받기"):
            st.dataframe(v.round(2), width="stretch")
            st.download_button("CSV 다운로드", v.to_csv().encode("utf-8-sig"), f"{last}.csv", "text/csv")


# ============================================================ 시나리오 비교 ==
with tab_cmp:
    runs = st.session_state.runs
    st.subheader(f"🎯 {RES_NAME} 방류(공급) 운영규칙 최적화")
    st.caption("사이드바의 모의기간·시작 저수율·유입/수요 배율 조건에서 저수율 경계 3개와 구간별 공급 비율(+예보 연계 절감)을 "
               "자동으로 찾습니다. 목적: 부족 심도지수 SI 최소화(한꺼번에 크게 모자라지 않게 미리 나눠 아끼기). "
               "어느 저수지든 제원·기상·유입량 모형으로 같은 방식이 적용됩니다.")
    o1_, o2_, o3_ = st.columns([1, 1, 1])
    end_t_ = o1_.number_input("기말에 남길 최소 유효저수율 (%)", 0.0, 90.0, 0.0, 5.0,
                              help="다음 해·겨울을 위해 남겨 둘 물. 0이면 제약 없음")
    use_fc_ = o2_.checkbox("예보 연계 절감도 함께 최적화", True)
    if o3_.button("최적 운영규칙 찾기", type="primary", width="stretch"):
        bar_o = st.progress(0.0, "최적화 시작…")
        try:
            opt_ = optimize_rules(inputs, cfg, start, end, initial_pct, demand_scale, inflow_scale,
                                  end_target_pct=end_t_, use_forecast=use_fc_,
                                  progress=lambda f, t: bar_o.progress(f, t))
            bar_o.empty()
            opt_["cond"] = f"{start}~{end} · 초기{initial_pct}% · 유입×{inflow_scale:g} 수요×{demand_scale:g}"
            st.session_state.opt_by_code[code] = opt_
            for k_, p_ in {**presets_for(cfg), "최적 운영규칙": opt_["scenario"]}.items():
                nm_ = f"{RES_NAME} · {k_} | 유입×{inflow_scale:g} 수요×{demand_scale:g} 초기{initial_pct}%"
                runs[nm_] = run_scenario(inputs, cfg, Scenario.from_dict(p_.to_dict()), start, end,
                                         initial_pct, demand_scale, inflow_scale, engine)
        except Exception as e:
            bar_o.empty()
            st.error(f"최적화 실패: {e}")
    opt_ = st.session_state.opt_by_code.get(code)
    if opt_:
        sc_o = opt_["scenario"]
        kb_, ko_ = opt_["baseline_kpis"], opt_["kpis"]
        st.success(f"**최적 운영규칙** ({opt_['cond']}, {opt_['evaluations']:,}회 모의): {sc_o.description}")
        rule_ = pd.DataFrame({
            "유효저수율 구간": [f"{sc_o.curves_pct[0]:.0f}% 이상", f"{sc_o.curves_pct[1]:.0f}~{sc_o.curves_pct[0]:.0f}%",
                          f"{sc_o.curves_pct[2]:.0f}~{sc_o.curves_pct[1]:.0f}%", f"{sc_o.curves_pct[2]:.0f}% 미만"],
            "단계": ["정상", "관심", "주의", "경계"],
            "공급 비율": [f"{f:.0%}" for f in sc_o.supply_factors]})
        oc1_, oc2_ = st.columns([1, 2])
        oc1_.dataframe(rule_, hide_index=True, width="stretch")
        with oc2_:
            m_ = st.columns(3)
            m_[0].metric("부족 심도지수 SI", f"{ko_['shortage_index']:.2f}",
                         f"{ko_['shortage_index'] - kb_['shortage_index']:+.2f} (관행 대비)", delta_color="inverse")
            m_[1].metric("최악 10일 부족률", f"{ko_['worst_10day_shortage_pct']:.0f} %",
                         f"{ko_['worst_10day_shortage_pct'] - kb_['worst_10day_shortage_pct']:+.0f}%p", delta_color="inverse")
            m_[2].metric("용수공급 만족률", f"{ko_['volumetric_reliability_pct']:.1f} %",
                         f"{ko_['volumetric_reliability_pct'] - kb_['volumetric_reliability_pct']:+.1f}%p")
            if kb_["shortage_index"] < 0.05:
                st.info("이 조건에서는 관행 운영으로도 부족이 거의 없어 제한급수가 필요 없습니다. "
                        "사이드바에서 유입량 배율을 낮추거나(가뭄) 시작 저수율을 낮춰 가뭄 대응 규칙을 찾아보세요.")
        st.caption("이 규칙은 ② 모의 실행의 '기본 시나리오'와 '실시간 운영'의 오늘의 공급 판단에도 자동으로 추가됩니다. "
                   "아래 표에서 기본 3개 시나리오와 함께 비교됩니다.")
    st.divider()
    if len(runs) < 1:
        st.info("②에서 시나리오를 2개 이상 실행하면 여기서 비교할 수 있습니다.")
    else:
        b1, b2 = st.columns([3, 1])
        sel = b1.multiselect("비교할 결과", list(runs), list(runs)[-3:])
        if b2.button("기본 3개 시나리오 한 번에 실행", width="stretch"):
            with st.spinner("실행 중…"):
                for k, p in presets_for(cfg).items():
                    nm = f"{RES_NAME} · {k} | 유입×{inflow_scale:g} 수요×{demand_scale:g} 초기{initial_pct}%"
                    runs[nm] = run_scenario(inputs, cfg, Scenario.from_dict(p.to_dict()), start, end,
                                            initial_pct, demand_scale, inflow_scale, engine)
            st.rerun()
        if b2.button("결과 모두 지우기", width="stretch"):
            st.session_state.runs = {}
            st.rerun()
        if sel:
            chosen = {k: runs[k] for k in sel}
            base = st.selectbox("기준 시나리오(차이 계산)", sel)
            tbl = pd.DataFrame({k: v.kpis for k, v in chosen.items()}).T
            show = [k for k in KPI_LABELS if k in tbl.columns]
            t2 = tbl[show].astype(float).rename(columns={k: f"{KPI_LABELS[k][0]} ({KPI_LABELS[k][1]})" if KPI_LABELS[k][1] else KPI_LABELS[k][0] for k in show})
            st.dataframe(t2.style.format("{:,.1f}"), width="stretch")

            st.plotly_chart(storage_chart(chosen, chosen[base].scenario, 400), width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                for i, (k, v) in enumerate(chosen.items()):
                    fig.add_trace(go.Scatter(x=v.series.index, y=v.series["deficit"].cumsum(), name=k,
                                             line=dict(color=COLORS[i % len(COLORS)])))
                fig.update_layout(title="누적 부족량 (천㎥)", height=340, margin=dict(l=10, r=10, t=40, b=10),
                                  legend=dict(orientation="h", y=-0.25), hovermode="x unified")
                st.plotly_chart(fig, width="stretch")
            with c2:
                metric = st.selectbox("막대 비교 지표", show, format_func=lambda k: KPI_LABELS[k][0], index=1)
                fig = go.Figure(go.Bar(x=list(chosen), y=[v.kpis[metric] for v in chosen.values()],
                                       marker_color=COLORS[: len(chosen)]))
                fig.update_layout(title=KPI_LABELS[metric][0], height=340, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, width="stretch")
            st.caption("해석: 제한급수는 부족량 총량보다 '부족의 심도'(SI, 최악 10일 부족률)를 줄이고, "
                       "최저 저수율과 기말 이월 저수율을 지켜 다음 해 위험을 낮추는 대신 평상시 공급 만족률이 떨어집니다.")


# =============================================================== 모델 검증 ==
with tab_val:
    st.subheader("1) 유입량 모형 보정·검증 (관측 저수량 기준)")
    ofiles = [f"observed_storage_{code}.csv"] if (DATA / f"observed_storage_{code}.csv").exists() else []
    if code == BASE_CFG["reservoir"]["code"]:
        ofiles.append("observed_storage_sample.csv")
    if not ofiles:
        st.info(f"{RES_NAME} 관측자료가 없습니다. `python data/fetch_reservoir.py fetch --name \"{RES_NAME}\"` 로 "
                "저수지 수위 API 자료를 받으면 여기서 보정·검증할 수 있습니다.")
    ofile = st.selectbox("관측 저수량 자료", ofiles,
                         help="열: date, storage(천㎥), release(천㎥/일). 저수지 수위 API(fetch_reservoir.py)로 생성") if ofiles else None
    if ofile and "sample" in ofile:
        st.caption("⚠️ 가상 관측자료입니다 (참 매개변수 X1=520, X2=-0.6, X3=140, X4=2.2로 생성 + 계측잡음). "
                   "보정 알고리즘이 참값을 되찾는지 확인하는 '쌍둥이 실험'용입니다.")
    if ofile:
        obs = pd.read_csv(DATA / ofile, parse_dates=["date"], index_col="date")
        est_release = "release" not in obs or obs["release"].isna().all()
        if est_release:
            # 공급량 관측이 없으면 모델 필요수량 + 하천유지유량으로 대신함
            # 공급 기록이 없으므로 '이론 필요수량'(공급계수 1)으로 두고, 공급계수·운영상한을 함께 보정
            base_g = {k: g[k] for k in ("x1", "x2", "x3", "x4")}
            est = prepare_inputs(weather, cfg, base_g)["demand"] + cfg["environment"]["river_maintenance_flow"]
            obs["release"] = est.reindex(obs.index)
            st.caption("공급량 관측이 없어 이론 필요수량으로 방류량을 두고, 보정 때 공급계수와 운영 상한을 함께 추정합니다.")
        obs = obs.loc[obs.index.intersection(weather.index)].dropna(subset=["storage"])
        if len(obs) < 60:
            st.warning("기상자료와 겹치는 관측 기간이 60일 미만입니다. 기상자료 기간을 늘려주세요.")
        else:
            o0, o1 = obs.index.min().date(), obs.index.max().date()
            split = (obs.index.min() + (obs.index.max() - obs.index.min()) * 0.67).date()
            v1, v2 = st.columns(2)
            cal = v1.date_input("보정기간", (o0, split), min_value=o0, max_value=o1, key="cal")
            val = v2.date_input("검증기간", (split, o1), min_value=o0, max_value=o1, key="val")
            if st.button("GR4J 보정 실행 (1분 내외)") and len(cal) == 2 and len(val) == 2:
                with st.spinner("차등진화 알고리즘으로 보정 중…"):
                    inp_all = prepare_inputs(weather, cfg, {k: g[k] for k in ("x1", "x2", "x3", "x4")})
                    rc = calibrate_gr4j(inp_all.loc[str(cal[0]):str(cal[1])], obs, cfg, maxiter=20,
                                        fit_release_scale=est_release)
                    rv = evaluate_params(inp_all.loc[str(val[0]):str(val[1])], obs, cfg, rc["params"])
                st.session_state.calib = (rc, rv)
            if "calib" in st.session_state:
                rc, rv = st.session_state.calib
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("보정 NSE", f"{rc['nse']:.3f}")
                m2.metric("보정 RMSE", f"{rc['rmse_pct']:.2f} %p")
                m3.metric("검증 NSE", f"{rv['nse']:.3f}")
                m4.metric("검증 RMSE", f"{rv['rmse_pct']:.2f} %p")
                st.write("보정 매개변수:", {k: round(v, 2) for k, v in rc["params"].items()})
                if st.button("이 매개변수를 모의에 적용 (저수지별로 저장)"):
                    st.session_state.gr4j_by_code[code] = {**g, **rc["params"]}
                    catalog.save_calibration(code, rc["params"], {"nse_cal": round(rc["nse"], 3),
                                                                  "nse_val": round(rv["nse"], 3),
                                                                  "saved": pd.Timestamp.now().strftime("%Y-%m-%d")})
                    for k_ in ("x1", "x2", "x3", "x4", "rs", "cap"):
                        st.session_state.pop(f"{k_}_{code}", None)
                    st.rerun()
                fig = go.Figure()
                for lab, rr, dash in (("보정", rc, "solid"), ("검증", rv, "solid")):
                    sr = rr["series"]
                    fig.add_trace(go.Scatter(x=sr.index, y=sr["obs_pct"], name=f"관측({lab})", mode="lines",
                                             line=dict(color="#111827", width=1)))
                    fig.add_trace(go.Scatter(x=sr.index, y=sr["sim_pct"], name=f"모의({lab})", mode="lines",
                                             line=dict(color="#2563eb" if lab == "보정" else "#ea580c", width=2, dash="dash")))
                fig.update_layout(height=380, yaxis_title="유효저수율 (%)", hovermode="x unified",
                                  margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, width="stretch")

    st.subheader("2) 계산 엔진 교차검증 (pywr vs 기준 물수지)")
    st.caption("같은 입력·규칙으로 pywr(선형계획)와 독립 구현한 물수지를 돌려 저수량 차이를 확인합니다. "
               "또 모든 날짜에서 '유입 − 유출 = 저수량 변화' 물수지 보존을 점검합니다.")
    if st.button("교차검증 실행"):
        sc = Scenario.from_dict(PRESETS["단계별 제한급수"].to_dict())
        ref = run_scenario(inputs, cfg, sc, start, end, initial_pct, demand_scale, inflow_scale, "reference").series
        out = {"기준 물수지 오차(최대, 천㎥)": float(
            (ref["storage"].diff().iloc[1:] - (ref["inflow"] - ref["supply"] - ref["env_flow"] - ref["evap"] - ref["spill"]).iloc[1:]).abs().max())}
        if PYWR_AVAILABLE:
            pw = run_scenario(inputs, cfg, sc, start, end, initial_pct, demand_scale, inflow_scale, "pywr").series
            out["pywr−기준 저수량 최대차 (천㎥)"] = float((pw["storage"] - ref["storage"]).abs().max())
            out["pywr−기준 공급량 합 차이 (%)"] = float((pw["supply"].sum() - ref["supply"].sum()) / ref["supply"].sum() * 100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ref.index, y=ref["storage_pct"], name="기준 물수지", line=dict(width=3, color="#94a3b8")))
            fig.add_trace(go.Scatter(x=pw.index, y=pw["storage_pct"], name="pywr", line=dict(dash="dash", color="#2563eb")))
            fig.update_layout(height=320, yaxis_title="유효저수율 (%)", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        else:
            out["pywr"] = "미설치 — 배포 서버에서 확인"
        st.json(out)


# =============================================================== 모델 개요 ==
with tab_dry:
    st.subheader(f"{RES_NAME} 가뭄 대응 검증")
    st.caption("① 기상가뭄 지수(SPI)로 가뭄 해를 확인하고, ② 실제 가뭄 해를 그해 실제 기상으로 재현해 운영규칙을 비교하고, "
               "③ 규칙을 찾을 때 쓰지 않은 해에도 효과가 있는지(과적합) 확인합니다.")
    # ---- ① SPI ----
    st.markdown("#### ① 기상가뭄 지수 SPI (표준강수지수)")
    if wfile == "weather_sample.csv":
        st.warning("가상 기상자료입니다. 사이드바에서 기상청 관측자료를 선택하면 실제 SPI가 계산됩니다.")
    spi3_, spi6_ = spi.spi(weather["rain"], 3), spi.spi(weather["rain"], 6)
    cur_spi_ = spi.current(weather["rain"])
    if len(cur_spi_):
        cs_ = st.columns(len(cur_spi_))
        for col_, (_, r_s) in zip(cs_, cur_spi_.iterrows()):
            col_.metric(f"{r_s['지수']} ({r_s['기준월']})", f"{r_s['값']:+.2f}", r_s["판정"], delta_color="off")
    fig_s = go.Figure()
    fig_s.add_trace(go.Bar(x=spi3_.index, y=spi3_, name="SPI-3 (농업가뭄)",
                           marker_color=["#dc2626" if v <= -1 else "#94a3b8" for v in spi3_.fillna(0)]))
    fig_s.add_trace(go.Scatter(x=spi6_.index, y=spi6_, name="SPI-6 (저수지·수문가뭄)", line=dict(color="#2563eb", width=2)))
    for y_, nm_ in ((-1.0, "보통가뭄"), (-1.5, "심한가뭄"), (-2.0, "극한가뭄")):
        fig_s.add_hline(y=y_, line_dash="dot", line_color="gray", annotation_text=nm_, annotation_position="top left")
    obs_p_ = krc_level.obs_path(code)
    obs_rate_ = (pd.read_csv(obs_p_, parse_dates=["date"], index_col="date")["rate"] if obs_p_.exists() else None)
    if obs_rate_ is not None:
        fig_s.add_trace(go.Scatter(x=obs_rate_.index, y=obs_rate_.clip(upper=100), name="관측 저수율(%)", yaxis="y2",
                                   line=dict(color="rgba(16,185,129,0.6)", width=1)))
    fig_s.update_layout(height=360, yaxis_title="SPI", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                        yaxis2=dict(title="저수율 (%)", overlaying="y", side="right", range=[0, 100], showgrid=False),
                        legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig_s, width="stretch")
    st.caption("SPI = 최근 3·6개월 강수를 같은 시기 과거 분포와 비교한 표준값 (−1 이하 보통가뭄, −1.5 심한가뭄, −2 극한가뭄). "
               "SPI가 먼저 떨어지고 몇 달 뒤 저수율이 떨어지므로 선제 절감 판단의 근거가 됩니다. "
               f"자료 기간 {weather.index.min():%Y}~{weather.index.max():%Y} — 30년 미만이면 분포 추정이 불안정해 참고용입니다.")

    if obs_rate_ is None or wfile == "weather_sample.csv":
        st.info("② 가뭄 해 재현과 ③ 과적합 확인에는 이 저수지의 농어촌공사 일별 저수율과 기상청 관측자료가 필요합니다 "
                "('저수지 현황' 탭에서 일별 저수율 받기).")
    else:
        years_ = drought.years_available(inputs, obs_rate_)
        if len(years_) < 4:
            st.info("관측 기간이 짧아(관개기 4개 해 미만) 검증을 할 수 없습니다.")
        else:
            # ---- ② 가뭄 해 재현 ----
            st.markdown("#### ② 실제 가뭄 해 재현 (그해 4/1 관측 저수율 + 실제 기상)")
            mins_ = {y_: float(obs_rate_.loc[f"{y_}-04-01":f"{y_}-10-31"].min()) for y_ in years_}
            driest_ = min(mins_, key=mins_.get)
            d1_, d2_, d3_ = st.columns([1, 1, 1])
            year_ = d1_.selectbox("가뭄 해", years_, index=years_.index(driest_),
                                  format_func=lambda y_: f"{y_}년 (관측 최저 {mins_[y_]:.0f}%)", key="dry_year")
            reserve_ = d2_.slider("비상 저수율 하한 (%)", 0, 40, 20, 5, key="dry_reserve",
                                  help="최저 저수율이 이보다 내려가지 않도록 하는 운영 목표 (사수위 근처 취수 곤란·이듬해 대비)")
            st.caption("최적 규칙은 **선택한 해를 빼고 나머지 해로만** 찾습니다 → 그해 비를 미리 알지 못한 상태의 공정한 비교입니다.")
            if d3_.button("가뭄 해 재현 실행 (1~3분)", type="primary", width="stretch", key="dry_run"):
                bar_d = st.progress(0.0, "시작…")
                _train = drought.training_years(inputs, obs_rate_, exclude=[year_])
                opt_d = drought.optimize_multi(inputs, cfg, _train, obs_rate_, maxiter=15,
                                               reserve_pct=reserve_, progress=lambda f, t: bar_d.progress(f, t))
                bar_d.empty()
                st.session_state.setdefault("dry_replays", {})[(code, year_, reserve_)] = drought.replay(
                    inputs, cfg, obs_rate_, year_, {**{k_: v_ for k_, v_ in presets_for(cfg).items() if k_ != "예보 연계 운영"},
                                                     "최적 규칙(이 해 제외 학습)": opt_d["scenario"]}, engine=engine)
                st.session_state["dry_res_rule"] = opt_d["scenario"]
            rp_ = st.session_state.get("dry_replays", {}).get((code, year_, reserve_))
            if rp_:
                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(x=rp_["observed"].index, y=rp_["observed"], name="실제 관측",
                                           line=dict(color="black", width=3)))
                for i_, (nm_, rr_) in enumerate(rp_["runs"].items()):
                    fig_d.add_trace(go.Scatter(x=rr_.series.index, y=rr_.series["storage_pct"], name=nm_,
                                               line=dict(color=COLORS[i_], width=2, dash="dash" if i_ == 0 else "solid")))
                fig_d.add_hline(y=reserve_, line_dash="dot", line_color="red", annotation_text=f"비상 하한 {reserve_}%",
                                annotation_position="top left")
                fig_d.update_layout(height=380, yaxis_title="유효저수율 (%)", hovermode="x unified",
                                    margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12))
                fig_d.update_yaxes(range=[0, 100])
                st.plotly_chart(fig_d, width="stretch")
                t_ = rp_["table"][["shortage_index", "worst_10day_shortage_pct", "volumetric_reliability_pct",
                                   "min_storage_pct", "total_deficit_k_m3"]].astype(float)
                t_.columns = ["부족 심도지수 SI", "최악 10일 부족률(%)", "공급 만족률(%)", "최저 저수율(%)", "총 부족량(천㎥)"]
                st.dataframe(t_.style.format("{:,.1f}").highlight_min(subset=["최악 10일 부족률(%)"], color="#dcfce7"),
                             width="stretch")
                base_, best_ = t_.iloc[0], t_.iloc[-1]
                st.success(f"**{year_}년 재현**: 관측 최저 {rp_['observed_min_pct']:.0f}% ({rp_['observed_min_date']}). "
                           f"관행 대비 최적 규칙 — 최악 10일 부족률 {base_['최악 10일 부족률(%)']:.0f}% → {best_['최악 10일 부족률(%)']:.0f}%, "
                           f"최저 저수율 {base_['최저 저수율(%)']:.0f}% → {best_['최저 저수율(%)']:.0f}%, "
                           f"공급 만족률 {base_['공급 만족률(%)']:.0f}% → {best_['공급 만족률(%)']:.0f}%")
                st.caption(f"최적 규칙: {st.session_state['dry_res_rule'].description}. "
                           "모의는 관측 저수율로 보정한 모형 결과이며, '실제 관측'과의 차이는 모형 오차와 실제 운영(제한급수 등)의 차이를 함께 포함합니다.")

            # ---- ③ 과적합 확인 ----
            st.markdown("#### ③ 과적합 확인 — 규칙을 찾을 때 쓰지 않은 해에도 효과가 있는가")
            o1_, o2_ = st.columns([2, 1])
            split_ = o1_.select_slider("이 해까지로 규칙을 찾고, 그 다음 해부터 검증", years_[1:-1],
                                       value=2021 if 2021 in years_[1:-1] else years_[len(years_) // 2], key="dry_split")
            wy_ = drought.weather_years(inputs)
            long_ = o1_.checkbox(f"기상 장기자료의 가뭄 해도 학습에 포함 ({min(wy_)}~{split_}년 중 관개기 강수가 가장 적은 15개 해, "
                                 "관측 없는 해는 4/1 저수율 중앙값에서 시작)",
                                 value=min(wy_) < 2005, disabled=min(wy_) >= 2005, key="dry_long",
                                 help="사이드바 '장기 관측(1973년~) 받기' 후 사용. 1994·2001년 같은 극한 가뭄을 학습에 넣어 규칙이 더 튼튼해집니다.")
            train_ = (drought.training_years(inputs, obs_rate_, upto=split_) if long_
                      else [y_ for y_ in years_ if y_ <= split_])
            if o2_.button("과적합 확인 실행 (1~3분)", width="stretch", key="dry_oos"):
                bar_o2 = st.progress(0.0, "시작…")
                st.session_state.setdefault("dry_oos_res", {})[(code, split_, reserve_, long_)] = drought.out_of_sample(
                    inputs, cfg, obs_rate_, train_, [y_ for y_ in years_ if y_ > split_],
                    maxiter=15, reserve_pct=reserve_, progress=lambda f, t: bar_o2.progress(f, t))
                bar_o2.empty()
            oo_ = st.session_state.get("dry_oos_res", {}).get((code, split_, reserve_, long_))
            if oo_:
                st.dataframe(oo_["table"].style.format({c_: "{:,.1f}" for c_ in oo_["table"].columns[3:]}),
                             width="stretch", hide_index=True)
                st.caption(f"다년 최적 규칙: {oo_['scenario'].description}. '검증 해' 행이 규칙을 찾을 때 보지 않은 해의 성능입니다. "
                           "여기서도 관행보다 최악 10일 부족률이 낮으면 과적합이 아니라고 볼 수 있습니다.")


with tab_info:
    st.markdown("""
### 모델 구성
| 단계 | 모델 | 내용 |
|---|---|---|
| 기준증발산 | FAO-56 Penman-Monteith | 기상청 ASOS 일자료 (일사 자료 없으면 Hargreaves) |
| 유입량 | **GR4J** (Perrin et al., 2003) | 4매개변수 일 강우-유출 모형, 관측 저수량으로 보정 |
| 필요수량 | 논 담수심 물수지 | 써레질 용수 + (Kc×ET0 + 침투 − 유효우량) ÷ 송수효율 |
| 저수지 운영 | **pywr** | 선형계획 기반 수자원 시스템 모델, 저수율 구간(ControlCurve)별 공급비율 |

### 운영 규칙
관개 공급 상한 = 필요수량 × 저수율 구간 계수 × 예보 계수.
우선순위는 수면증발 → 관개 → 하천유지유량 → 저류 → 월류 순입니다.
""")
    st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="sans-serif"];
  wx [label="기상청 ASOS\\n강수·기온·습도·풍속·일조"]; et [label="ET0\\n(FAO-56 PM)"];
  gr [label="GR4J\\n유역 유출"]; dm [label="논 수요량\\n(담수심 물수지)"];
  res [label="성주댐 (pywr Storage)", fillcolor="#dbeafe"]; irr [label="관개구역"]; env [label="하천유지유량"];
  sp [label="월류"]; ev [label="수면증발"]; rule [label="운영규칙\\n저수율 구간·예보", fillcolor="#fef3c7"];
  wx -> et; wx -> gr; et -> gr; et -> dm; wx -> dm; gr -> res [label="유입량"]; dm -> rule [label="필요수량"];
  rule -> irr [style=dashed]; res -> irr; res -> env; res -> sp; res -> ev; res -> rule [style=dashed, label="저수율"]; }
""")
