"""무료 위성영상 타일 (Esri World Imagery) — 인증키 없이 3D 지형 텍스처·첫 화면 배너에 사용

타일: https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
 - 웹 메르카토르(EPSG:3857) 표준 타일, 키 없음
 - 화면에 출처를 함께 표시해야 합니다(ATTRIBUTION).
브이월드 키가 없거나 클라우드에서 브이월드가 막힐 때 대신 씁니다.
받은 타일은 data/cache_tiles/esri/ 에 저장해 재사용합니다.
"""
from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path

import numpy as np

from .geocode import UA
from .vworld import TILE, choose_zoom, lonlat_to_tile_frac, sample_mosaic

ROOT = Path(__file__).resolve().parents[1]
TILE_DIR = ROOT / "data" / "cache_tiles" / "esri"
URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ATTRIBUTION = "위성영상: Esri World Imagery (Esri, Maxar, Earthstar Geographics, GIS User Community)"


def fetch_tile(z: int, x: int, y: int) -> np.ndarray:
    from PIL import Image

    cache = TILE_DIR / str(z) / str(x) / f"{y}.jpg"
    if cache.exists():
        return np.asarray(Image.open(cache).convert("RGB"))
    req = urllib.request.Request(URL.format(z=z, x=x, y=y), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = r.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        img.save(cache, quality=90)
    except OSError:
        pass
    return np.asarray(img)


def _mosaic_tiles(z: int, tx0: int, ty0: int, tx1: int, ty1: int) -> np.ndarray:
    m = np.zeros(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE, 3), np.uint8)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            t = fetch_tile(z, tx, ty)
            m[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE, (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = t[:TILE, :TILE]
    return m


def texture_for_grid(lon: np.ndarray, lat: np.ndarray, target_px: int = 1400, max_tiles: int = 64) -> np.ndarray:
    """DEM 표시 격자(경위도 배열)에 입힐 위성영상 RGB (vworld.texture_for_grid 와 같은 형식)"""
    lon_min, lat_min = float(np.nanmin(lon)), float(np.nanmin(lat))
    lon_max, lat_max = float(np.nanmax(lon)), float(np.nanmax(lat))
    z = choose_zoom(lon_min, lat_min, lon_max, lat_max, target_px, zmax=16)
    fx0, fy0 = lonlat_to_tile_frac(lon_min, lat_max, z)
    fx1, fy1 = lonlat_to_tile_frac(lon_max, lat_min, z)
    tx0, ty0, tx1, ty1 = int(fx0), int(fy0), int(fx1), int(fy1)
    while (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > max_tiles and z > 7:
        z -= 1
        fx0, fy0 = lonlat_to_tile_frac(lon_min, lat_max, z)
        fx1, fy1 = lonlat_to_tile_frac(lon_max, lat_min, z)
        tx0, ty0, tx1, ty1 = int(fx0), int(fy0), int(fx1), int(fy1)
    mosaic = _mosaic_tiles(z, tx0, ty0, tx1, ty1)
    return sample_mosaic(mosaic, z, tx0, ty0, lon, lat)


def thumbnail(lat: float, lon: float, width_px: int = 1600, height_px: int = 420, zoom: int = 15,
              fmt: str = "JPEG") -> bytes:
    """저수지 중심 위성사진 한 장 (첫 화면 배너용) → 이미지 바이트"""
    from PIL import Image

    fx, fy = lonlat_to_tile_frac(lon, lat, zoom)
    cx, cy = float(fx) * TILE, float(fy) * TILE  # 전역 픽셀 좌표
    px0, py0 = cx - width_px / 2, cy - height_px / 2
    tx0, ty0 = int(math.floor(px0 / TILE)), int(math.floor(py0 / TILE))
    tx1, ty1 = int(math.floor((px0 + width_px) / TILE)), int(math.floor((py0 + height_px) / TILE))
    m = _mosaic_tiles(zoom, tx0, ty0, tx1, ty1)
    ox, oy = int(round(px0 - tx0 * TILE)), int(round(py0 - ty0 * TILE))
    crop = m[oy:oy + height_px, ox:ox + width_px]
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, fmt, quality=82)
    return buf.getvalue()


def map_layer() -> dict:
    """plotly 지도(white-bg 스타일)에 깔 위성영상 래스터 레이어"""
    return {"sourcetype": "raster", "source": [URL], "below": "traces",
            "sourceattribution": "Esri World Imagery"}
