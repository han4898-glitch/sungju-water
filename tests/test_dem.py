"""DEM 분석 테스트 (합성 지형 — 실제 DEM 없이 알고리즘 검증)"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import dem as D  # noqa: E402

H, W, CELL, DAM_R = 220, 180, 50.0, 150


@pytest.fixture(scope="module")
def synthetic():
    """남쪽으로 흐르는 V자 계곡 + 댐체(EL.200) + DEM 측량 당시 수면(EL.180)"""
    rr, cc = np.indices((H, W))
    z = 120 + (H - rr) * 0.4 + np.abs(cc - W / 2) * 1.5 + np.random.default_rng(0).normal(0, 0.3, (H, W))
    z[DAM_R - 1:DAM_R + 2, 60:120] = np.maximum(z[DAM_R - 1:DAM_R + 2, 60:120], 200)
    z[(rr < DAM_R - 1) & (z < 180)] = 180.0
    g = D.DemGrid(z, 0.0, H * CELL, CELL, None)
    target = DAM_R * W * CELL ** 2 / 1e6  # 댐 위쪽 전체 = 유역
    return g, D.analyze(g, g.rc_to_xy(DAM_R + 3, W // 2), target, search_radius_m=1500)


def test_watershed_area_matches_target(synthetic):
    g, a = synthetic
    assert abs(a.basin_area_km2 / a.target_area_km2 - 1) < 0.03
    assert a.outlet_rc[0] >= DAM_R  # 출구가 댐 하류 쪽


def test_survey_lake_detected(synthetic):
    _, a = synthetic
    assert a.survey_level == pytest.approx(180.0)
    assert a.lake_seed is not None and a.lake_seed.sum() > 1000


def test_water_extent_stays_upstream_and_grows(synthetic):
    _, a = synthetic
    areas = []
    for lv in (175, 180, 185, 190):
        w, below = a.extent(lv)
        assert (np.nonzero(w)[0].max() < DAM_R - 1) if w.any() else True  # 댐 아래로 넘어가지 않음
        assert below == (lv < 180)
        areas.append(w.sum())
    assert areas[1] <= areas[2] <= areas[3]


def test_fill_removes_pits():
    z = np.full((20, 20), 10.0)
    z[:, 0] = 5.0
    z[10, 10] = 1.0  # 함몰지
    f = D.fill_depressions(z)
    down = D.flow_direction(f)
    assert f[10, 10] >= 10.0
    # 함몰지에서 출발한 흐름이 순환 없이 격자 밖까지 빠져나감
    i, steps = 10 * 20 + 10, 0
    while i >= 0 and steps < 400:
        i, steps = down[i], steps + 1
    assert i == -1
    acc = D.flow_accumulation(f, down)
    assert acc.sum() >= 400 and acc.max() > 1


def test_level_storage_curve_passes_specs():
    ls = D.LevelStorage(162.2, 10090, 184.7, 38240, 2050)
    assert ls.storage(184.7) == pytest.approx(38240)
    assert ls.storage(162.2) == pytest.approx(10090)
    assert ls.level(ls.storage(175.0)) == pytest.approx(175.0)
    # 만수위에서 dV/dh ≈ 만수면적
    assert (ls.storage(184.7) - ls.storage(184.6)) / 0.1 == pytest.approx(2050, rel=0.01)


def test_xyz_loader(tmp_path):
    xs, ys = np.meshgrid(np.arange(0, 50, 5.0), np.arange(0, 30, 5.0))
    zs = xs + ys
    p = tmp_path / "tile.txt"
    np.savetxt(p, np.c_[xs.ravel(), ys.ravel(), zs.ravel()])
    g = D.load_xyz([p], "EPSG:5186")
    assert g.shape == (6, 10) and g.cell == 5.0
    assert g.z[0, 0] == pytest.approx(25.0)  # 북서쪽 (x=0, y=25)


def test_map_figure_builds(synthetic):
    pytest.importorskip("plotly")
    from engine.dem_plot import map_figure

    _, a = synthetic
    w, b = a.extent(185)
    fig = map_figure(a, w, b, "test")
    assert len(fig.data) >= 4


def test_outlet_uses_dam_elevation():
    """면적이 같은 두 계곡 중 표고(만수위−제체높이)가 맞는 쪽을 고른다"""
    rr, cc = np.indices((120, 160))
    left = 100 + (120 - rr) * 0.5 + np.abs(cc - 40) * 1.0     # 낮은 계곡 (출구 근처 표고 ~110)
    right = 300 + (120 - rr) * 0.5 + np.abs(cc - 120) * 1.0   # 높은 계곡 (~310)
    z = np.where(cc < 80, left, right)
    z[:, 79:81] = 600  # 분수령
    g = D.DemGrid(z, 0.0, 120 * 50.0, 50.0, None)
    target = 100 * 78 * 50.0 ** 2 / 1e6
    mid = g.rc_to_xy(110, 80)
    a_low = D.analyze(g, mid, target, search_radius_m=4000, full_level=150, dam_height=40)
    a_high = D.analyze(g, mid, target, search_radius_m=4000, full_level=350, dam_height=40)
    assert a_low.outlet_rc[1] < 80 < a_high.outlet_rc[1]


def test_window_radius():
    assert D.window_radius_m(149.6) == pytest.approx(5000 + 2000 * np.sqrt(149.6))
    assert D.window_radius_m(5000) == 35000


def test_raster_window_read(tmp_path):
    """전국 DEM처럼 큰 GeoTIFF에서 저수지 주변 창만 읽는지 (rasterio가 있는 PC에서 실행)"""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    z = (np.arange(400)[:, None] + np.arange(500)[None, :]).astype("float32")
    p = tmp_path / "dem90.tif"
    with rasterio.open(p, "w", driver="GTiff", height=400, width=500, count=1, dtype="float32",
                       crs="EPSG:5186", transform=from_origin(200000, 400000, 90, 90), nodata=-9999) as dst:
        dst.write(z, 1)
    cx, cy = 200000 + 250 * 90, 400000 - 200 * 90
    lon, lat = D.xy_to_lonlat(cx, cy, "EPSG:5186")
    g = D.load_dem(tmp_path, center_lonlat=(lon, lat), radius_m=4500)
    assert g.cell == pytest.approx(90)
    assert 95 <= g.shape[0] <= 102 and 95 <= g.shape[1] <= 102
    r, c = g.xy_to_rc(cx, cy)
    assert abs(g.z[r, c] - (200 + 250)) <= 2


def test_surface_figure_builds(synthetic):
    pytest.importorskip("plotly")
    from engine.dem_plot import surface_figure

    _, a = synthetic
    w, b = a.extent(185)
    fig = surface_figure(a, w, 185, b, "3D")
    assert any(t.type == "surface" for t in fig.data)


def test_vworld_tile_math_and_palette():
    from engine import vworld as V

    x, y = V.lonlat_to_tile_frac(np.array([128.168]), np.array([35.858]), 12)
    assert 3500 < x[0] < 3510 and 1600 < y[0] < 1620  # 성주 부근 z12 타일
    z = V.choose_zoom(128.0, 35.7, 128.4, 36.0, target_px=1600)
    assert 9 <= z <= 14
    mosaic = np.zeros((512, 512, 3), np.uint8)
    mosaic[:256, :256] = [200, 10, 10]
    tx0, ty0 = 10, 20
    col = V.sample_mosaic(mosaic, 5, tx0, ty0, *[np.array([v]) for v in (-180 + (tx0 + 0.5) / 32 * 360, 0)])
    assert col.shape == (1, 3)
    pytest.importorskip("PIL")
    rgb = np.random.default_rng(0).integers(0, 255, (20, 30, 3)).astype(np.uint8)
    idx, scale, n = V.rgb_to_surfacecolor(rgb, 16)
    assert idx.shape == (20, 30) and idx.max() <= n - 1 and len(scale) == n


def test_vworld_live_tile():
    """VWORLD_KEY가 설정된 PC에서만: 성주 부근 위성영상 타일 1장 받기"""
    from engine import vworld as V

    key = V.get_key()
    if not key:
        pytest.skip("VWORLD_KEY 없음")
    x, y = V.lonlat_to_tile_frac(np.array([128.168]), np.array([35.858]), 12)
    t = V.fetch_tile(key, 12, int(x[0]), int(y[0]), "http://localhost:8501")
    assert t.shape[0] == 256 and t.shape[2] == 3


def test_lake_center_needs_crs(synthetic):
    _, a = synthetic
    assert a.lake_center_lonlat(185) is None  # 합성 격자는 좌표계가 없음


def test_sheet_numbers():
    assert D.sheet_50k(128.146, 35.908) == "35801"  # 성주댐 (갖고 있는 35801.img)
    assert D.sheet_50k(128.307, 36.712) == "36806"  # 문경 경천
    s = D.sheets_for(128.146, 35.908, 15000)
    assert s[0] == "35801" and len(s) == len(set(s)) and "35802" in s and "35704" in s
