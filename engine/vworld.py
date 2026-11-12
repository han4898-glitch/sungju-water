"""브이월드(국토교통부 공간정보 오픈플랫폼) 항공·위성영상 타일 → DEM 3D 지형에 입힐 텍스처

WMTS 타일:  https://api.vworld.kr/req/wmts/1.0.0/{key}/Satellite/{z}/{y}/{x}.jpeg
 - 좌표계: 웹 메르카토르(EPSG:3857) 표준 타일 체계
 - 인증키는 환경변수 VWORLD_KEY 또는 .streamlit/secrets.toml 의 VWORLD_KEY
 - 키를 발급받을 때 등록한 서비스 URL을 Referer로 보냅니다(config [vworld] referer).
받은 타일은 data/cache_tiles/ 에 저장해 재사용합니다(호출 횟수 절약).
"""
from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path

import numpy as np

from .geocode import UA, _env

ROOT = Path(__file__).resolve().parents[1]
TILE_DIR = ROOT / "data" / "cache_tiles"
TILE = 256


def get_key() -> str | None:
    return _env("VWORLD_KEY")


# ------------------------------------------------------- 타일 좌표 계산 --
def lonlat_to_tile_frac(lon: np.ndarray, lat: np.ndarray, z: int) -> tuple[np.ndarray, np.ndarray]:
    """경위도 → (타일 x, 타일 y) 실수 좌표 (웹 메르카토르)"""
    lat = np.clip(np.asarray(lat, float), -85.0511, 85.0511)
    n = 2.0 ** z
    x = (np.asarray(lon, float) + 180.0) / 360.0 * n
    y = (1.0 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def choose_zoom(lon_min, lat_min, lon_max, lat_max, target_px: int = 1600, zmin: int = 7, zmax: int = 17) -> int:
    for z in range(zmax, zmin - 1, -1):
        x0, y0 = lonlat_to_tile_frac(lon_min, lat_max, z)
        x1, y1 = lonlat_to_tile_frac(lon_max, lat_min, z)
        if max(x1 - x0, y1 - y0) * TILE <= target_px:
            return z
    return zmin


# ------------------------------------------------------------ 타일 받기 --
def tile_url(key: str, z: int, x: int, y: int, layer: str = "Satellite") -> str:
    ext = "jpeg" if layer == "Satellite" else "png"
    return f"https://api.vworld.kr/req/wmts/1.0.0/{key}/{layer}/{z}/{y}/{x}.{ext}"


def fetch_tile(key: str, z: int, x: int, y: int, referer: str | None, layer: str = "Satellite"):
    from PIL import Image

    cache = TILE_DIR / layer / str(z) / str(x) / f"{y}.png"
    if cache.exists():
        return np.asarray(Image.open(cache).convert("RGB"))
    headers = {"User-Agent": UA}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(tile_url(key, z, x, y, layer), headers=headers)
    with urllib.request.urlopen(req, timeout=20) as r:
        data = r.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")  # 오류 응답(XML 등)이면 여기서 예외
    cache.parent.mkdir(parents=True, exist_ok=True)
    img.save(cache)
    return np.asarray(img)


def fetch_mosaic(key: str, bounds_lonlat: tuple[float, float, float, float], referer: str | None = None,
                 target_px: int = 1600, max_tiles: int = 100, layer: str = "Satellite"):
    """범위를 덮는 타일을 받아 한 장으로 붙입니다. 반환: (RGB 배열, z, 왼쪽 위 타일 x, y)"""
    lon_min, lat_min, lon_max, lat_max = bounds_lonlat
    z = choose_zoom(lon_min, lat_min, lon_max, lat_max, target_px)
    fx0, fy0 = lonlat_to_tile_frac(lon_min, lat_max, z)
    fx1, fy1 = lonlat_to_tile_frac(lon_max, lat_min, z)
    tx0, ty0, tx1, ty1 = int(fx0), int(fy0), int(fx1), int(fy1)
    if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > max_tiles:
        raise ValueError("요청 범위가 너무 큽니다.")
    mosaic = np.zeros(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE, 3), np.uint8)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            t = fetch_tile(key, z, tx, ty, referer, layer)
            mosaic[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE, (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = t[:TILE, :TILE]
    return mosaic, z, tx0, ty0


def sample_mosaic(mosaic: np.ndarray, z: int, tx0: int, ty0: int, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """경위도 격자의 각 점에 해당하는 영상 색을 뽑습니다 → (H, W, 3)"""
    fx, fy = lonlat_to_tile_frac(lon, lat, z)
    px = np.clip(((fx - tx0) * TILE).astype(int), 0, mosaic.shape[1] - 1)
    py = np.clip(((fy - ty0) * TILE).astype(int), 0, mosaic.shape[0] - 1)
    return mosaic[py, px]


def texture_for_grid(key: str, lon: np.ndarray, lat: np.ndarray, referer: str | None = None,
                     target_px: int = 1600) -> np.ndarray:
    """DEM 표시 격자(경위도 배열)에 입힐 항공영상 RGB"""
    b = (float(np.nanmin(lon)), float(np.nanmin(lat)), float(np.nanmax(lon)), float(np.nanmax(lat)))
    mosaic, z, tx0, ty0 = fetch_mosaic(key, b, referer, target_px)
    return sample_mosaic(mosaic, z, tx0, ty0, lon, lat)


# ----------------------------------------- plotly Surface에 RGB 입히기 --
def rgb_to_surfacecolor(rgb: np.ndarray, n_colors: int = 96):
    """plotly Surface는 RGB 텍스처를 직접 받지 못하므로, 영상을 n색으로 줄여
    (색 번호 배열, 색표) 형태로 바꿉니다 → surfacecolor=번호, colorscale=색표"""
    from PIL import Image

    h, w, _ = rgb.shape
    q = Image.fromarray(rgb.astype(np.uint8)).quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    idx = np.asarray(q, dtype=float).reshape(h, w)
    pal = np.asarray(q.getpalette()[: 3 * n_colors]).reshape(-1, 3)
    n = len(pal)
    scale = [[i / (n - 1), f"rgb({r},{g},{b})"] for i, (r, g, b) in enumerate(pal)] if n > 1 else \
        [[0, f"rgb({pal[0][0]},{pal[0][1]},{pal[0][2]})"], [1, f"rgb({pal[0][0]},{pal[0][1]},{pal[0][2]})"]]
    return idx, scale, n
