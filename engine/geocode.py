"""저수지 소재지 주소 → 좌표 (지오코딩) 와 좌표 캐시

농어촌공사 제원표에는 좌표가 없고 소재지(예: '경상북도 성주군 가천면 중산리')만 있으므로
주소를 좌표로 바꿔 data/reservoir_coords.csv 에 저장해 두고 재사용합니다.

지오코더 우선순위 (키가 있는 것부터):
  1) 브이월드(국토교통부 공간정보 오픈플랫폼)  환경변수 VWORLD_KEY
  2) 카카오 로컬 API                           환경변수 KAKAO_REST_KEY
  3) OpenStreetMap Nominatim                   키 불필요 (초당 1회 제한, 리 단위 정확도는 지역마다 다름)

리(里) 중심 좌표는 실제 댐과 수 km 떨어질 수 있습니다. DEM 분석에서 '유역면적·댐 표고가 맞는 하천 지점'을
찾아 보정하고, 그래도 틀리면 웹앱에서 좌표를 직접 고쳐 저장할 수 있습니다(source = manual).
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COORDS_CSV = ROOT / "data" / "reservoir_coords.csv"
COLUMNS = ["표준코드", "시설명", "lat", "lon", "source", "query"]
UA = "krc-reservoir-webapp/1.0 (graduate course project)"


def _env(name: str) -> str | None:
    """환경변수 → .streamlit/secrets.toml 순으로 읽음. 붙여넣을 때 섞인 공백·따옴표는 제거"""
    v = os.environ.get(name)
    if not v:
        try:
            import streamlit as st  # type: ignore

            v = st.secrets.get(name)  # type: ignore[attr-defined]
        except Exception:
            v = None
    if not v:
        return None
    v = str(v).strip().strip('"').strip("'").strip()
    return v or None


def _get_json(url: str, headers: dict | None = None, timeout: int = 15):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ------------------------------------------------------------ 주소 다듬기 --
def address_parts(addr: str) -> dict:
    """'경상북도 성주군 가천면 중산리 123' → 시도/시군구/읍면동/리"""
    toks = str(addr).split()
    out = {"sido": "", "sgg": "", "emd": "", "ri": ""}
    for t in toks:
        if t.endswith(("도", "특별시", "광역시", "특별자치시", "특별자치도")) and not out["sido"]:
            out["sido"] = t
        elif t.endswith(("시", "군", "구")) and not out["emd"]:
            out["sgg"] = (out["sgg"] + " " + t).strip()
        elif t.endswith(("읍", "면", "동")) and not out["emd"]:
            out["emd"] = t
        elif t.endswith("리") and not out["ri"]:
            out["ri"] = t
    return out


def queries_for(addr: str) -> list[str]:
    """정밀한 것부터 차례로 시도할 주소 문자열"""
    p = address_parts(addr)
    base = " ".join(x for x in (p["sido"], p["sgg"], p["emd"]) if x)
    qs = []
    if p["ri"]:
        qs.append(f"{base} {p['ri']}")
    if base:
        qs.append(base)
    return qs or [str(addr)]


# ------------------------------------------------------------- 지오코더 --
def parse_vworld(data: dict, want: str) -> tuple[float, float] | None:
    resp = data.get("response", {})
    if resp.get("status") != "OK":
        return None
    res = resp.get("result")
    if isinstance(res, dict) and "point" in res:  # getcoord
        pt = res["point"]
        return float(pt["y"]), float(pt["x"])
    items = (res or {}).get("items", []) if isinstance(res, dict) else []
    for it in items:  # search(district)
        title = it.get("title", "")
        if all(w in title for w in want.split()[-2:]):
            pt = it["point"]
            return float(pt["y"]), float(pt["x"])
    return None


def _vworld_headers() -> dict:
    try:
        import tomllib

        with open(ROOT / "config" / "sungju.toml", "rb") as f:
            ref = tomllib.load(f).get("vworld", {}).get("referer")
        return {"Referer": ref} if ref else {}
    except Exception:
        return {}


def geocode_vworld(q: str, key: str) -> tuple[float, float] | None:
    base = "https://api.vworld.kr/req/"
    h = _vworld_headers()
    p = {"service": "address", "request": "getcoord", "version": "2.0", "crs": "epsg:4326",
         "address": q, "format": "json", "type": "PARCEL", "key": key}
    r = parse_vworld(_get_json(base + "address?" + urllib.parse.urlencode(p), h), q)
    if r:
        return r
    level = "L4" if q.split()[-1].endswith("리") else "L3"
    p = {"service": "search", "request": "search", "version": "2.0", "crs": "EPSG:4326", "size": 20,
         "query": q.split()[-1], "type": "district", "category": level, "format": "json", "key": key}
    return parse_vworld(_get_json(base + "search?" + urllib.parse.urlencode(p), h), q)


def parse_kakao(data: dict) -> tuple[float, float] | None:
    docs = data.get("documents") or []
    if not docs:
        return None
    return float(docs[0]["y"]), float(docs[0]["x"])


def geocode_kakao(q: str, key: str) -> tuple[float, float] | None:
    url = "https://dapi.kakao.com/v2/local/search/address.json?" + urllib.parse.urlencode({"query": q})
    return parse_kakao(_get_json(url, {"Authorization": f"KakaoAK {key}"}))


def parse_nominatim(data: list, want: str) -> tuple[float, float] | None:
    if not data:
        return None
    last = want.split()[-1]
    for d in data:  # 원하는 리/면 이름이 들어간 결과 우선
        if last in d.get("display_name", ""):
            return float(d["lat"]), float(d["lon"])
    return float(data[0]["lat"]), float(data[0]["lon"])


_last_nominatim = [0.0]


def geocode_nominatim(q: str) -> tuple[float, float] | None:
    wait = 1.1 - (time.time() - _last_nominatim[0])
    if wait > 0:
        time.sleep(wait)
    _last_nominatim[0] = time.time()
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q, "format": "jsonv2", "countrycodes": "kr", "limit": 5, "accept-language": "ko"})
    return parse_nominatim(_get_json(url), q)


def geocode(addr: str) -> tuple[float, float, str, str] | None:
    """(lat, lon, source, 사용한 주소) 또는 None"""
    vk, kk = _env("VWORLD_KEY"), _env("KAKAO_REST_KEY")
    for q in queries_for(addr):
        for name, fn in (("vworld", (lambda s: geocode_vworld(s, vk)) if vk else None),
                         ("kakao", (lambda s: geocode_kakao(s, kk)) if kk else None),
                         ("nominatim", geocode_nominatim)):
            if fn is None:
                continue
            try:
                r = fn(q)
            except Exception:
                r = None
            if r and 33.0 < r[0] < 39.0 and 124.0 < r[1] < 132.0:  # 남한 범위 확인
                return r[0], r[1], name + ("" if q.split()[-1].endswith("리") else "(읍면)"), q
    return None


# ------------------------------------------------ 저수지 이름(POI)으로 찾기 --
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math

    p = math.pi / 180
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))


def poi_names(name: str) -> list[str]:
    n = str(name).strip()
    base = n[:-3] if n.endswith("저수지") else (n[:-1] if n.endswith(("지", "댐")) and len(n) > 2 else n)
    return [f"{base}저수지", f"{base}지", f"{base}댐", f"{base}못", f"{base}호"]


def parse_vworld_place(data: dict, name_variants: list[str], sgg: str) -> tuple[float, float] | None:
    """브이월드 장소 검색 결과에서 저수지 이름·시군이 맞는 항목"""
    resp = data.get("response", {})
    if resp.get("status") != "OK":
        return None
    items = (resp.get("result") or {}).get("items", []) or []
    best = None
    for it in items:
        title = str(it.get("title", "")).replace(" ", "")
        addr = it.get("address") or {}
        addr_s = " ".join(str(v) for v in addr.values()) if isinstance(addr, dict) else str(addr)
        if not any(v in title for v in name_variants):
            continue
        score = (2 if sgg and sgg.split()[0] in addr_s else 0) + (1 if title in name_variants else 0)
        pt = it.get("point") or {}
        if "x" in pt and "y" in pt and (best is None or score > best[0]):
            best = (score, float(pt["y"]), float(pt["x"]))
    return (best[1], best[2]) if best else None


def poi_vworld(name: str, addr: str, key: str) -> tuple[float, float] | None:
    sgg = address_parts(addr)["sgg"]
    variants = poi_names(name)
    for q in variants[:2]:
        p = {"service": "search", "request": "search", "version": "2.0", "crs": "EPSG:4326", "size": 20,
             "query": q, "type": "place", "format": "json", "key": key}
        r = parse_vworld_place(_get_json("https://api.vworld.kr/req/search?" + urllib.parse.urlencode(p),
                                         _vworld_headers()), variants, sgg)
        if r:
            return r
    return None


def parse_overpass(data: dict, lat: float, lon: float, name_variants: list[str]) -> tuple[float, float] | None:
    """OSM 수면(저수지) 중 이름이 맞는 것 → 없으면 가장 큰 수면"""
    best = None
    for el in data.get("elements", []):
        c = el.get("center") or ({"lat": el["lat"], "lon": el["lon"]} if "lat" in el else None)
        b = el.get("bounds")
        if not c:
            continue
        tags = el.get("tags", {})
        nm = (tags.get("name:ko") or tags.get("name") or "").replace(" ", "")
        named = any(v in nm for v in name_variants)
        area = ((b["maxlat"] - b["minlat"]) * (b["maxlon"] - b["minlon"])) if b else 0.0
        score = (1e6 if named else 0) + area * 1e6 - _haversine_km(lat, lon, c["lat"], c["lon"]) * 1e-3
        if best is None or score > best[0]:
            best = (score, c["lat"], c["lon"])
    return (best[1], best[2]) if best else None


def poi_overpass(name: str, lat: float, lon: float, radius_m: int = 5000) -> tuple[float, float] | None:
    q = f"""[out:json][timeout:25];(
      way["natural"="water"](around:{radius_m},{lat},{lon});
      relation["natural"="water"](around:{radius_m},{lat},{lon});
      way["landuse"="reservoir"](around:{radius_m},{lat},{lon});
      way["water"="reservoir"](around:{radius_m},{lat},{lon}););out center bb tags;"""
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    return parse_overpass(data, lat, lon, poi_names(name))


def find_reservoir_point(name: str, addr: str, near: tuple[float, float] | None = None,
                         max_km: float = 15.0) -> tuple[float, float, str] | None:
    """저수지 자체의 위치: 브이월드 장소검색 → OSM 수면. near(주소 좌표)에서 max_km 이내만 인정"""
    vk = _env("VWORLD_KEY")
    cands = []
    if vk:
        try:
            r = poi_vworld(name, addr, vk)
            if r:
                cands.append((r[0], r[1], "vworld-POI"))
        except Exception:
            pass
    if not cands and near is not None:
        try:
            r = poi_overpass(name, near[0], near[1])
            if r:
                cands.append((r[0], r[1], "OSM 수면"))
        except Exception:
            pass
    for la, lo, src in cands:
        if near is None or _haversine_km(near[0], near[1], la, lo) <= max_km:
            return la, lo, src
    return None


# ----------------------------------------------------------------- 캐시 --
def load_coords() -> pd.DataFrame:
    for _ in range(5):  # 다른 프로세스(웹앱·테스트)가 쓰는 중이면 잠깐 기다렸다 다시 읽음
        if not COORDS_CSV.exists():
            break
        try:
            return pd.read_csv(COORDS_CSV, dtype={"표준코드": str})
        except (pd.errors.EmptyDataError, PermissionError, OSError):
            time.sleep(0.2)
    return pd.DataFrame(columns=COLUMNS)


def write_coords(df: pd.DataFrame) -> None:
    """임시 파일에 쓴 뒤 한 번에 바꿔치기 → 동시에 읽는 쪽이 빈 파일을 보지 않음"""
    tmp = COORDS_CSV.with_suffix(f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False, encoding="utf-8")
    for _ in range(5):
        try:
            os.replace(tmp, COORDS_CSV)
            return
        except PermissionError:
            time.sleep(0.2)
    os.replace(tmp, COORDS_CSV)


def save_coord(code: str, name: str, lat: float, lon: float, source: str, query: str = "") -> None:
    df = load_coords()
    df = df[df["표준코드"] != str(code)]
    row = pd.DataFrame([[str(code), name, float(lat), float(lon), source, query]], columns=COLUMNS)
    write_coords(pd.concat([df, row], ignore_index=True).sort_values("표준코드"))


def get_coords(code: str, name: str, address: str, allow_network: bool = True) -> dict | None:
    """캐시에 있으면 그 값, 없으면 지오코딩 후 저장. 반환: {lat, lon, source}"""
    df = load_coords()
    hit = df[df["표준코드"] == str(code)]
    if not hit.empty:
        r = hit.iloc[0]
        return {"lat": float(r["lat"]), "lon": float(r["lon"]), "source": r["source"]}
    rw = _rawris_point(code)  # 농어촌공사 계측 위치 (저장본) — 가장 정확
    if rw:
        save_coord(code, name, rw[0], rw[1], "RAWRIS", "rawris")
        return {"lat": rw[0], "lon": rw[1], "source": "RAWRIS"}
    if not allow_network:
        return None
    g = geocode(address)
    near = (g[0], g[1]) if g else None
    poi = find_reservoir_point(name, address, near)
    if poi:  # 주소(리·읍면 중심)보다 저수지 자체 위치를 우선
        save_coord(code, name, poi[0], poi[1], poi[2], "")
        return {"lat": poi[0], "lon": poi[1], "source": poi[2]}
    if g is None:
        return None
    save_coord(code, name, g[0], g[1], g[2], g[3])
    return {"lat": g[0], "lon": g[1], "source": g[2]}


REFINED_SOURCES = ("manual", "RAWRIS", "DEM", "vworld-POI", "OSM 수면")


def _rawris_point(code: str) -> tuple[float, float] | None:
    """RAWRIS 저장본(data/rawris_snapshot.json)에 있는 계측 저수지 위치"""
    try:
        from . import rawris

        if not rawris.CACHE.exists():
            return None
        r = rawris.lookup(code, rawris.to_frame(json.loads(rawris.CACHE.read_text(encoding="utf-8"))))
        if r and pd.notna(r["lat"]) and 33 < r["lat"] < 39 and 124 < r["lon"] < 132:
            return float(r["lat"]), float(r["lon"])
    except Exception:  # noqa: BLE001
        return None
    return None


def refine_coords(code: str, name: str, address: str) -> dict | None:
    """주소 중심 좌표로 저장된 저수지를 저수지 위치(POI)로 한 번 다시 찾아봄. 실패해도 다시 시도하지 않도록 표시"""
    df = load_coords()
    hit = df[df["표준코드"] == str(code)]
    if hit.empty:
        return None
    r = hit.iloc[0]
    src = str(r["source"])
    if src.startswith(REFINED_SOURCES) or "POI없음" in src:
        return None
    poi = find_reservoir_point(name, address, (float(r["lat"]), float(r["lon"])))
    if poi:
        save_coord(code, name, poi[0], poi[1], poi[2], "")
        return {"lat": poi[0], "lon": poi[1], "source": poi[2]}
    save_coord(code, name, float(r["lat"]), float(r["lon"]), src + "·POI없음", str(r.get("query", "")))
    return None



def refine_all(catalog_df: pd.DataFrame, codes: list[str] | None = None, region: str | None = None,
               min_storage: float = 0.0, force: bool = False, progress=None) -> pd.DataFrame:
    """여러 저수지 위치를 저수지 이름(POI)으로 다시 찾음.
    codes 를 주면 그 저수지만, 아니면 region(본부)·min_storage(천㎥) 조건의 저수지 전체.
    force=True 면 이미 POI·DEM으로 찾은 것도 다시 찾음 (manual 은 항상 유지)."""
    cat = catalog_df
    if codes is not None:
        cat = cat[cat["표준코드"].isin([str(c) for c in codes])]
    else:
        if region:
            cat = cat[cat["본부"] == region]
        cat = cat[cat["총저수량"] >= min_storage]
    rows = []
    n = len(cat)
    for i, (_, r) in enumerate(cat.iterrows(), 1):
        code, name, addr = str(r["표준코드"]), r["시설명"], r["소재지"]
        if progress:
            progress(i / max(n, 1), f"{i}/{n} {name}")
        df = load_coords()
        hit = df[df["표준코드"] == code]
        src = str(hit.iloc[0]["source"]) if len(hit) else ""
        if src == "manual" or (src.startswith(("DEM",)) and not force):
            rows.append((code, name, src, "유지"))
            continue
        if len(hit) == 0:
            c = get_coords(code, name, addr)
            rows.append((code, name, c["source"] if c else "", "새로 찾음" if c else "실패"))
            continue
        near = (float(hit.iloc[0]["lat"]), float(hit.iloc[0]["lon"]))
        poi = find_reservoir_point(name, addr, near)
        if poi:
            save_coord(code, name, poi[0], poi[1], poi[2], "")
            moved = _haversine_km(near[0], near[1], poi[0], poi[1])
            rows.append((code, name, poi[2], f"이동 {moved:.1f}km"))
        else:
            if "POI없음" not in src:
                save_coord(code, name, near[0], near[1], src + "·POI없음", str(hit.iloc[0].get("query", "")))
            rows.append((code, name, src, "이름으로 못 찾음(주소 좌표 유지)"))
    return pd.DataFrame(rows, columns=["표준코드", "시설명", "출처", "결과"])
