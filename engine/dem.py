"""국토지리정보원 수치표고모델(DEM) 기반 지형 분석

국토지리정보원 DEM은 실시간 API가 아니라 국토정보플랫폼(map.ngii.go.kr)에서 내려받는 파일로 제공됩니다.
내려받은 파일을 data/dem/ 폴더에 넣으면 다음을 계산합니다.

1) 함몰지 채우기(Priority-Flood, Barnes et al. 2014) → D8 흐름방향 → 흐름누적
2) 댐 위치(유역 출구) 자동 탐색: 대략 좌표 주변에서 '상류 면적이 제원표 유역면적과 가장 가까운' 하천 셀
3) 유역 경계 추출 → DEM 유역면적과 제원표 유역면적 비교 (검증)
4) 저수위별 수면 범위: 유역 안에서 출구와 연결된 '수위 이하' 셀
5) 저수량 → 저수위 환산(제원표 만수위·사수위·만수면적으로 맞춘 수위-저수량 곡선)

지원 형식: GeoTIFF(.tif), ERDAS Imagine(.img) 등 rasterio로 읽히는 래스터, XYZ 텍스트(.txt/.xyz: x y z)
※ DEM은 측량 당시 수면을 평평한 면으로 기록하므로, 그 수면보다 낮은 수위의 호소 바닥 지형은 알 수 없습니다.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEM_DIR = ROOT / "data" / "dem"
RASTER_EXT = {".tif", ".tiff", ".img", ".asc", ".bil", ".dem"}
XYZ_EXT = {".txt", ".xyz", ".csv"}

# D8 이웃: (행, 열) 이동과 거리 배수
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DISTS = np.array([np.sqrt(2), 1, np.sqrt(2), 1, 1, np.sqrt(2), 1, np.sqrt(2)])


# ------------------------------------------------------------- 읽기 ----
@dataclass
class DemGrid:
    z: np.ndarray            # 표고 (nan = 자료 없음), 행 0 = 북쪽
    x0: float                # 왼쪽 위 셀 모서리 x (투영좌표, m)
    y0: float                # 왼쪽 위 셀 모서리 y
    cell: float              # 셀 크기 (m)
    crs: str | None          # 예: "EPSG:5186"
    source: list[str] = field(default_factory=list)

    @property
    def shape(self):
        return self.z.shape

    def rc_to_xy(self, r, c):
        return self.x0 + (np.asarray(c) + 0.5) * self.cell, self.y0 - (np.asarray(r) + 0.5) * self.cell

    def xy_to_rc(self, x, y):
        return int((self.y0 - y) // self.cell), int((x - self.x0) // self.cell)


def find_dem_files(folder: Path = DEM_DIR) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in RASTER_EXT | XYZ_EXT and p.name.lower() != "readme.txt")


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return a
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    b = a[:h, :w].reshape(h // f, f, w // f, f)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.nanmean(b, axis=3), axis=1)


def load_xyz(paths: list[Path], crs: str | None, target_cell: float | None = None) -> DemGrid:
    """국토지리정보원 5m DEM 등 'x y z' 텍스트 격자를 읽어 하나의 격자로 합칩니다."""
    pts = np.vstack([np.loadtxt(p, delimiter=None if p.suffix != ".csv" else ",", usecols=(0, 1, 2)) for p in paths])
    xs, ys = np.unique(pts[:, 0]), np.unique(pts[:, 1])
    res = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
    nx = int(round((xs.max() - xs.min()) / res)) + 1
    ny = int(round((ys.max() - ys.min()) / res)) + 1
    z = np.full((ny, nx), np.nan)
    ci = np.rint((pts[:, 0] - xs.min()) / res).astype(int)
    ri = np.rint((ys.max() - pts[:, 1]) / res).astype(int)
    z[ri, ci] = pts[:, 2]
    g = DemGrid(z, xs.min() - res / 2, ys.max() + res / 2, res, crs, [p.name for p in paths])
    if target_cell and target_cell > res:
        f = int(round(target_cell / res))
        g = DemGrid(_block_mean(z, f), g.x0, g.y0, res * f, crs, g.source)
    return g


def load_raster(paths: list[Path], crs_override: str | None, max_cells: int,
                center_lonlat: tuple[float, float] | None = None, radius_m: float | None = None) -> DemGrid:
    """래스터 DEM 읽기. center_lonlat·radius_m을 주면 그 주변 창(window)만 읽습니다(전국 DEM용)."""
    import rasterio  # 사용자 PC에서만 필요 (requirements.txt)
    from rasterio.enums import Resampling
    from rasterio.merge import merge

    srcs = [rasterio.open(p) for p in paths]
    try:
        crs = crs_override or (srcs[0].crs.to_string() if srcs[0].crs else None)
        res0 = abs(srcs[0].res[0])
        left = min(s.bounds.left for s in srcs); right = max(s.bounds.right for s in srcs)
        bottom = min(s.bounds.bottom for s in srcs); top = max(s.bounds.top for s in srcs)
        if center_lonlat is not None and radius_m:
            if not crs:
                raise ValueError("DEM 좌표계를 알 수 없습니다. config [dem] crs 를 지정하세요 (예: EPSG:5179).")
            cx, cy = lonlat_to_xy(center_lonlat[0], center_lonlat[1], crs)
            if not (left <= cx <= right and bottom <= cy <= top):
                raise ValueError("이 저수지 위치가 DEM 범위 밖입니다. 해당 지역 DEM을 data/dem/ 에 추가하세요.")
            left, right = max(left, cx - radius_m), min(right, cx + radius_m)
            bottom, top = max(bottom, cy - radius_m), min(top, cy + radius_m)
        ncell = (right - left) * (top - bottom) / res0 ** 2
        res = res0 * max(1.0, float(np.ceil(np.sqrt(ncell / max_cells))))
        kw = dict(bounds=(left, bottom, right, top), res=res, nodata=np.nan)
        for extra in ({"dtype": "float32", "resampling": Resampling.average}, {"dtype": "float32"}, {}):
            try:
                arr, t = merge(srcs, **kw, **extra)
                break
            except (TypeError, ValueError):
                continue
        else:  # pragma: no cover
            raise RuntimeError("DEM을 읽지 못했습니다.")
        z = arr[0].astype(float)
        nod = srcs[0].nodata
        if nod is not None and not np.isnan(nod):
            z[z == nod] = np.nan
        z[z < -100] = np.nan  # nodata 값(-9999 등) 정리
        return DemGrid(z, t.c, t.f, abs(t.a), crs, [p.name for p in paths])
    finally:
        for s_ in srcs:
            s_.close()


def crop(g: DemGrid, cx: float, cy: float, radius_m: float) -> DemGrid:
    r0, c0 = g.xy_to_rc(cx - radius_m, cy + radius_m)
    r1, c1 = g.xy_to_rc(cx + radius_m, cy - radius_m)
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(g.shape[0], r1 + 1), min(g.shape[1], c1 + 1)
    if r0 >= r1 or c0 >= c1:
        raise ValueError("이 저수지 위치가 DEM 범위 밖입니다.")
    return DemGrid(g.z[r0:r1, c0:c1], g.x0 + c0 * g.cell, g.y0 - r0 * g.cell, g.cell, g.crs, g.source)


def load_dem(folder: Path = DEM_DIR, crs_override: str | None = None, max_cells: int = 600_000,
             center_lonlat: tuple[float, float] | None = None, radius_m: float | None = None) -> DemGrid | None:
    files = find_dem_files(folder)
    if not files:
        return None
    rasters = [p for p in files if p.suffix.lower() in RASTER_EXT]
    if rasters:
        return load_raster(rasters, crs_override, max_cells, center_lonlat, radius_m)
    g = load_xyz(files, crs_override)
    if center_lonlat is not None and radius_m and g.crs:
        g = crop(g, *lonlat_to_xy(center_lonlat[0], center_lonlat[1], g.crs), radius_m)
    f = max(1, int(np.ceil(np.sqrt(g.z.size / max_cells))))
    if f > 1:
        g = DemGrid(_block_mean(g.z, f), g.x0, g.y0, g.cell * f, g.crs, g.source)
    return g


def xy_to_lonlat_arrays(x: np.ndarray, y: np.ndarray, crs: str) -> tuple[np.ndarray, np.ndarray]:
    from rasterio.warp import transform

    xs, ys = transform(crs, "EPSG:4326", np.ravel(x).tolist(), np.ravel(y).tolist())
    return np.reshape(xs, np.shape(x)), np.reshape(ys, np.shape(y))


def sheet_50k(lon: float, lat: float) -> str:
    """1:50,000 도엽번호 (예: 성주댐 → 35801). 위경도 1°칸을 15'×15' 16칸으로 나눠 북서쪽부터 01~16"""
    la, lo = int(np.floor(lat)), int(np.floor(lon))
    row = min(3, int((la + 1 - lat) / 0.25))
    col = min(3, int((lon - lo) / 0.25))
    return f"{la}{lo % 10}{row * 4 + col + 1:02d}"


def sheets_for(lon: float, lat: float, radius_m: float) -> list[str]:
    """저수지 주변 반경을 덮는 1:50,000 도엽번호들 (가운데 도엽 먼저)"""
    dlat = radius_m / 111_000.0
    dlon = radius_m / (111_000.0 * np.cos(np.radians(lat)))
    out = [sheet_50k(lon, lat)]
    for y in np.arange(lat - dlat, lat + dlat + 1e-9, 0.125):
        for x in np.arange(lon - dlon, lon + dlon + 1e-9, 0.125):
            s = sheet_50k(float(x), float(y))
            if s not in out:
                out.append(s)
    for x, y in ((lon - dlon, lat - dlat), (lon + dlon, lat + dlat), (lon - dlon, lat + dlat), (lon + dlon, lat - dlat)):
        s = sheet_50k(x, y)
        if s not in out:
            out.append(s)
    return out


def window_radius_m(catchment_km2: float) -> float:
    """유역을 덮을 만큼의 읽기 반경: 5km + 유역 크기의 2배 (최대 35km)"""
    return float(min(35_000.0, 5_000.0 + 2_000.0 * np.sqrt(max(catchment_km2, 0.1))))


def lonlat_to_xy(lon: float, lat: float, crs: str) -> tuple[float, float]:
    from rasterio.warp import transform

    xs, ys = transform("EPSG:4326", crs, [lon], [lat])
    return xs[0], ys[0]


def xy_to_lonlat(x: float, y: float, crs: str) -> tuple[float, float]:
    from rasterio.warp import transform

    xs, ys = transform(crs, "EPSG:4326", [x], [y])
    return xs[0], ys[0]


# --------------------------------------------------------- 수문 분석 ----
def fill_depressions(z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Priority-Flood + ε (Barnes et al., 2014). 모든 셀이 가장자리까지 내리막으로 흐르도록 채웁니다."""
    H, W = z.shape
    valid = ~np.isnan(z)
    filled = np.where(valid, z, np.nan)
    closed = ~valid
    heap: list = []
    # 가장자리 셀: 격자 테두리 또는 nan 옆
    pad = np.pad(valid, 1, constant_values=False)
    nb_invalid = np.zeros_like(valid)
    for dr, dc in OFFSETS:
        nb_invalid |= ~pad[1 + dr:H + 1 + dr, 1 + dc:W + 1 + dc]
    edge = valid & nb_invalid
    for r, c in zip(*np.nonzero(edge)):
        heapq.heappush(heap, (filled[r, c], r, c))
        closed[r, c] = True
    while heap:
        h, r, c = heapq.heappop(heap)
        for dr, dc in OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and not closed[rr, cc]:
                closed[rr, cc] = True
                v = filled[rr, cc]
                if v <= h:
                    v = h + eps
                    filled[rr, cc] = v
                heapq.heappush(heap, (v, rr, cc))
    return filled


def flow_direction(filled: np.ndarray) -> np.ndarray:
    """D8: 각 셀의 하류 셀 일렬 인덱스 (-1 = 격자 밖으로 유출)"""
    H, W = filled.shape
    big = np.inf
    pad = np.pad(np.where(np.isnan(filled), -big, filled), 1, constant_values=-big)
    best = np.zeros((H, W)); arg = np.full((H, W), -1)
    for k, (dr, dc) in enumerate(OFFSETS):
        nb = pad[1 + dr:H + 1 + dr, 1 + dc:W + 1 + dc]
        with np.errstate(invalid="ignore"):
            slope = (filled - nb) / DISTS[k]
        slope = np.where(np.isfinite(slope), slope, 0.0)
        out = np.isinf(nb)  # 격자 밖/자료 없음 → 유출구
        better = (slope > best) & ~out
        best = np.where(better, slope, best); arg = np.where(better, k, arg)
    rr, cc = np.indices((H, W))
    dr = np.array([o[0] for o in OFFSETS]); dc = np.array([o[1] for o in OFFSETS])
    down = np.where(arg >= 0, (rr + dr[arg]) * W + (cc + dc[arg]), -1)
    down[np.isnan(filled)] = -1
    return down.ravel()


def flow_accumulation(filled: np.ndarray, down: np.ndarray) -> np.ndarray:
    """상류 셀 수(자기 자신 포함)"""
    f = filled.ravel()
    acc = np.where(np.isnan(f), 0.0, 1.0)
    order = np.argsort(np.where(np.isnan(f), -np.inf, f))[::-1]
    for i in order:
        d = down[i]
        if d >= 0 and acc[i] > 0:
            acc[d] += acc[i]
    return acc.reshape(filled.shape)


def watershed(filled: np.ndarray, down: np.ndarray, outlet: int) -> np.ndarray:
    f = filled.ravel()
    inside = np.zeros(f.size, bool)
    inside[outlet] = True
    order = np.argsort(np.where(np.isnan(f), np.inf, f))  # 하류(낮은 곳)부터
    for i in order:
        d = down[i]
        if d >= 0 and inside[d]:
            inside[i] = True
    return inside.reshape(filled.shape)


def locate_outlet(acc: np.ndarray, near_rc: tuple[int, int], radius_cells: int, target_cells: float,
                  filled: np.ndarray | None = None, dam_base_elev: float | None = None,
                  dam_height: float | None = None) -> tuple[int, int]:
    """대략 위치 반경 안에서 '상류 면적이 제원표 유역면적에 가깝고, 표고가 댐 기초 표고에 가까운' 하천 셀.

    점수 = 면적 오차율 + 0.5 × 표고 오차/댐높이 + 0.05 × 거리비
    (리 중심 좌표가 수 km 어긋나도 같은 규모의 다른 하천을 고르지 않도록 표고 조건을 함께 씀)
    """
    H, W = acc.shape
    r0, c0 = near_rc
    r1, r2 = max(0, r0 - radius_cells), min(H, r0 + radius_cells + 1)
    c1, c2 = max(0, c0 - radius_cells), min(W, c0 + radius_cells + 1)
    sub = acc[r1:r2, c1:c2]
    rr, cc = np.indices(sub.shape)
    dist = np.hypot(rr + r1 - r0, cc + c1 - c0) / max(radius_cells, 1)
    score = np.abs(sub - target_cells) / target_cells + 0.05 * dist
    if filled is not None and dam_base_elev is not None and np.isfinite(dam_base_elev):
        scale = max(float(dam_height or 10.0), 5.0)
        ez = np.abs(filled[r1:r2, c1:c2] - dam_base_elev) / scale
        score = score + 0.5 * np.where(np.isfinite(ez), ez, 10.0)
    score = np.where(dist <= 1, score, np.inf)
    k = int(np.argmin(score))
    return r1 + k // sub.shape[1], c1 + k % sub.shape[1]


def water_extent(z: np.ndarray, basin: np.ndarray, outlet_rc: tuple[int, int], level: float,
                 cell: float, lake_seed: np.ndarray | None = None, survey_level: float | None = None,
                 seed_radius_m: float = 1500.0) -> tuple[np.ndarray, bool]:
    """수위 level일 때 저수지 수면 범위.

    반환: (수면 마스크, below_survey)
      below_survey=True 이면 수위가 DEM 측량 당시 수면보다 낮아 호소 바닥 지형을 알 수 없으므로
      측량 당시 수면 범위를 대신 돌려줍니다(화면에서 따로 표시).
    """
    from scipy import ndimage

    if lake_seed is not None and lake_seed.any() and survey_level is not None and level < survey_level:
        return lake_seed.copy(), True
    wet = basin & (z <= level)
    if not wet.any():
        return wet, False
    lab, _ = ndimage.label(wet, structure=np.ones((3, 3)))
    if lake_seed is not None and lake_seed.any():
        ids = np.unique(lab[lake_seed & wet])
        ids = ids[ids > 0]
        return np.isin(lab, ids), False
    rr, cc = np.indices(z.shape)
    near = np.hypot(rr - outlet_rc[0], cc - outlet_rc[1]) * cell <= seed_radius_m
    cand = near & wet
    if not cand.any():
        return np.zeros_like(wet), False
    ids = np.unique(lab[cand])
    sizes = ndimage.sum(wet, lab, ids)
    return lab == ids[int(np.argmax(sizes))], False


def hillshade(z: np.ndarray, cell: float, azimuth=315.0, altitude=45.0) -> np.ndarray:
    zz = np.where(np.isnan(z), np.nanmean(z), z)
    gy, gx = np.gradient(zz, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.deg2rad(azimuth), np.deg2rad(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


# ------------------------------------------------------ 수위-저수량 ----
@dataclass
class LevelStorage:
    """V(h) = Vd + k (h - hd)^b,  사수위(hd, Vd)·만수위(hf, Vf)를 지나고 만수위에서 dV/dh = 만수면적"""
    hd: float
    vd: float
    hf: float
    vf: float
    area_full_k_m2: float  # 만수면적 (천㎡)

    def __post_init__(self):
        dh, dv = self.hf - self.hd, self.vf - self.vd
        self.b = float(np.clip(self.area_full_k_m2 * dh / dv, 1.0, 4.0))
        self.k = dv / dh ** self.b

    def level(self, v):
        v = np.asarray(v, float)
        return self.hd + (np.clip(v - self.vd, 0, None) / self.k) ** (1 / self.b)

    def storage(self, h):
        h = np.asarray(h, float)
        return self.vd + self.k * np.clip(h - self.hd, 0, None) ** self.b


def level_storage_from_cfg(cfg: dict) -> LevelStorage | None:
    r = cfg["reservoir"]
    try:
        return LevelStorage(float(r["dead_level"]), float(r["dead_storage"]), float(r["full_level"]),
                            float(r["total_storage"]), float(r["full_area_ha"]) * 10.0)
    except (KeyError, TypeError, ValueError):
        return None


# ----------------------------------------------------------- 전체 분석 ----
@dataclass
class DemAnalysis:
    grid: DemGrid
    filled: np.ndarray
    acc: np.ndarray
    basin: np.ndarray
    outlet_rc: tuple[int, int]
    basin_area_km2: float
    target_area_km2: float
    outlet_lonlat: tuple[float, float] | None
    survey_level: float | None  # DEM에 기록된 저수지 수면 표고(추정)
    lake_seed: np.ndarray | None = None  # DEM 측량 당시 수면 셀

    def lake_center_lonlat(self, level: float | None = None) -> tuple[float, float] | None:
        """만수위(또는 DEM 측량 당시 수면) 수면 범위의 중심 경위도 — 지도용 저수지 좌표"""
        if not self.grid.crs:
            return None
        mask = self.lake_seed if (self.lake_seed is not None and self.lake_seed.any()) else None
        if level is not None:
            ext, _ = self.extent(level)
            if ext.any():
                mask = ext
        if mask is None or not mask.any():
            return None
        r, c = np.nonzero(mask)
        x, y = self.grid.rc_to_xy(r.mean(), c.mean())
        lon, lat = xy_to_lonlat(float(x), float(y), self.grid.crs)
        return float(lon), float(lat)

    def extent(self, level: float) -> tuple[np.ndarray, bool]:
        return water_extent(self.grid.z, self.basin, self.outlet_rc, level, self.grid.cell,
                            self.lake_seed, self.survey_level)


def analyze(grid: DemGrid, approx_xy: tuple[float, float] | None, target_area_km2: float,
            search_radius_m: float = 6000.0, full_level: float | None = None,
            dam_height: float | None = None) -> DemAnalysis:
    """full_level(만수위)·dam_height(제체높이)를 주면 댐 기초 표고 ≈ 만수위 − 제체높이 조건을 함께 씁니다."""
    filled = fill_depressions(grid.z)
    down = flow_direction(filled)
    acc = flow_accumulation(filled, down)
    cell_km2 = grid.cell ** 2 / 1e6
    target = target_area_km2 / cell_km2
    if approx_xy is not None:
        near = grid.xy_to_rc(*approx_xy)
        if not (0 <= near[0] < grid.shape[0] and 0 <= near[1] < grid.shape[1]):
            raise ValueError("댐 대략 위치가 DEM 범위 밖입니다. DEM 범위 또는 config [dem] 좌표를 확인하세요.")
        radius = int(search_radius_m / grid.cell)
    else:  # 좌표가 없으면 격자 전체에서 탐색
        near = (grid.shape[0] // 2, grid.shape[1] // 2)
        radius = max(grid.shape)
    base = (full_level - 0.9 * dam_height) if (full_level and dam_height) else None
    orc = locate_outlet(acc, near, radius, target, filled, base, dam_height)
    basin = watershed(filled, down, orc[0] * grid.shape[1] + orc[1])
    lonlat = None
    if grid.crs:
        try:
            lonlat = xy_to_lonlat(*grid.rc_to_xy(*orc), grid.crs)
        except Exception:
            lonlat = None
    # DEM 수면 추정: 출구 1.5km 이내 유역 셀 중 가장 넓은 평탄면의 표고
    z = grid.z
    rr, cc = np.indices(z.shape)
    near_mask = basin & (np.hypot(rr - orc[0], cc - orc[1]) * grid.cell <= 1500)
    vals = np.round(z[near_mask & ~np.isnan(z)], 1)
    survey, seed = None, None
    if vals.size:
        u, n = np.unique(vals, return_counts=True)
        if n.max() >= 20:  # 같은 표고 셀이 20개 이상 = 평평한 수면
            survey = float(u[np.argmax(n)])
            from scipy import ndimage

            flat = basin & (np.abs(z - survey) <= 0.05)
            lab, _ = ndimage.label(flat, structure=np.ones((3, 3)))
            ids = np.unique(lab[near_mask & flat])
            ids = ids[ids > 0]
            if ids.size:
                sizes = ndimage.sum(flat, lab, ids)
                seed = lab == ids[int(np.argmax(sizes))]
    return DemAnalysis(grid, filled, acc, basin, orc, float(basin.sum() * cell_km2), target_area_km2,
                       lonlat, survey, seed)
