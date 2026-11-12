"""DEM 지형·유역·수면 범위 지도 (plotly)"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .dem import DemAnalysis, hillshade


def _crop_to_basin(a: DemAnalysis, margin_cells: int = 15):
    rows, cols = np.nonzero(a.basin)
    if rows.size == 0:
        return slice(None), slice(None)
    H, W = a.basin.shape
    return (slice(max(0, rows.min() - margin_cells), min(H, rows.max() + margin_cells + 1)),
            slice(max(0, cols.min() - margin_cells), min(W, cols.max() + margin_cells + 1)))


def map_figure(a: DemAnalysis, water: np.ndarray | None = None, below_survey: bool = False,
               title: str = "", height: int = 520, max_px: int = 450) -> go.Figure:
    g = a.grid
    rs, cs = _crop_to_basin(a)
    z, basin = g.z[rs, cs], a.basin[rs, cs]
    wat = water[rs, cs] if water is not None else None
    step = max(1, int(np.ceil(max(z.shape) / max_px)))
    z, basin = z[::step, ::step], basin[::step, ::step]
    if wat is not None:
        wat = wat[::step, ::step]
    cell = g.cell * step
    r0, c0 = (rs.start or 0), (cs.start or 0)
    x = (g.x0 + (c0 + (np.arange(z.shape[1]) * step) + 0.5 * step) * g.cell) / 1000.0
    y = (g.y0 - (r0 + (np.arange(z.shape[0]) * step) + 0.5 * step) * g.cell) / 1000.0

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=x, y=y, z=z, colorscale="earth", colorbar=dict(title="표고(m)", len=0.6),
                             hovertemplate="x %{x:.2f} km<br>y %{y:.2f} km<br>표고 %{z:.1f} m<extra></extra>"))
    hs = hillshade(z, cell)
    fig.add_trace(go.Heatmap(x=x, y=y, z=hs, colorscale=[[0, "rgba(0,0,0,0.55)"], [1, "rgba(255,255,255,0)"]],
                             showscale=False, hoverinfo="skip"))
    if wat is not None and wat.any():
        color = "rgba(147,197,253,0.85)" if below_survey else "rgba(37,99,235,0.85)"
        fig.add_trace(go.Heatmap(x=x, y=y, z=np.where(wat, 1.0, np.nan), colorscale=[[0, color], [1, color]],
                                 showscale=False, hoverinfo="skip", name="수면"))
    fig.add_trace(go.Contour(x=x, y=y, z=basin.astype(float), showscale=False, hoverinfo="skip",
                             contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
                             line=dict(color="#dc2626", width=2), name="유역 경계"))
    ox, oy = g.rc_to_xy(*a.outlet_rc)
    fig.add_trace(go.Scatter(x=[ox / 1000], y=[oy / 1000], mode="markers+text", text=["댐(유역 출구)"],
                             textposition="top right", marker=dict(symbol="triangle-down", size=14, color="#111827"),
                             hoverinfo="text", name="댐"))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      showlegend=False, xaxis=dict(title="x (km)", constrain="domain"),
                      yaxis=dict(title="y (km)", scaleanchor="x", scaleratio=1))
    return fig


def surface_figure(a: DemAnalysis, water: np.ndarray | None = None, level: float | None = None,
                   below_survey: bool = False, title: str = "", exaggeration: float = 2.0,
                   height: int = 560, max_px: int = 180, texture_fn=None) -> go.Figure:
    """DEM 3D 지형 + 수면(수위 평면) + 유역 경계 + 댐 위치.

    exaggeration: 수직 과장 배율 (지형 기복을 잘 보이게)
    texture_fn  : (lon 배열, lat 배열) → RGB 배열. 주면 지형에 항공영상을 입힘 (브이월드)
    """
    g = a.grid
    rs, cs = _crop_to_basin(a, margin_cells=8)
    z, basin = g.z[rs, cs], a.basin[rs, cs]
    wat = water[rs, cs] if water is not None else None
    step = max(1, int(np.ceil(max(z.shape) / max_px)))
    z, basin = z[::step, ::step], basin[::step, ::step]
    if wat is not None:
        wat = wat[::step, ::step]
    r0, c0 = (rs.start or 0), (cs.start or 0)
    x = (g.x0 + (c0 + np.arange(z.shape[1]) * step + 0.5 * step) * g.cell) / 1000.0
    y = (g.y0 - (r0 + np.arange(z.shape[0]) * step + 0.5 * step) * g.cell) / 1000.0
    X, Y = np.meshgrid(x, y)
    zmin, zmax = np.nanmin(z), np.nanmax(z)

    fig = go.Figure()
    # 지형: 항공영상(브이월드)을 입히거나, 없으면 표고 색
    surf = dict(surfacecolor=z, colorscale="earth", cmin=zmin, cmax=zmax,
                lighting=dict(ambient=0.55, diffuse=0.8, roughness=0.9, specular=0.05))
    if texture_fn is not None and g.crs:
        from .dem import xy_to_lonlat_arrays
        from .vworld import rgb_to_surfacecolor

        lon, lat = xy_to_lonlat_arrays(X * 1000.0, Y * 1000.0, g.crs)
        idx, scale, n = rgb_to_surfacecolor(texture_fn(lon, lat))
        surf = dict(surfacecolor=idx, colorscale=scale, cmin=0, cmax=max(n - 1, 1),
                    lighting=dict(ambient=0.85, diffuse=0.35, roughness=1.0, specular=0.0))
    fig.add_trace(go.Surface(x=X, y=Y, z=z, showscale=False, opacity=1.0, name="지형",
                             customdata=z, hovertemplate="표고 %{z:.0f} m<extra></extra>", **surf))
    # 수면
    if wat is not None and wat.any():
        lv = float(level) if level is not None and not below_survey else (a.survey_level or float(np.nanmin(z[wat])))
        wz = np.where(wat, lv + 3.0, np.nan)  # 90m 격자 지형에 묻히지 않도록 살짝 띄움
        color = "rgb(147,197,253)" if below_survey else "rgb(37,99,235)"
        fig.add_trace(go.Surface(x=X, y=Y, z=wz, colorscale=[[0, color], [1, color]], showscale=False,
                                 opacity=0.95, name="수면", hovertemplate=f"수면 EL.{lv:.1f} m<extra></extra>"))
    # 유역 경계 (경계 셀을 점으로)
    edge = basin & ~(np.roll(basin, 1, 0) & np.roll(basin, -1, 0) & np.roll(basin, 1, 1) & np.roll(basin, -1, 1))
    er, ec = np.nonzero(edge)
    fig.add_trace(go.Scatter3d(x=X[er, ec], y=Y[er, ec], z=z[er, ec] + 15, mode="markers",
                               marker=dict(size=1.0, color="#f43f5e"), name="유역 경계", hoverinfo="skip"))
    # 댐
    ox, oy = g.rc_to_xy(*a.outlet_rc)
    oz = float(g.z[a.outlet_rc]) if np.isfinite(g.z[a.outlet_rc]) else zmin
    fig.add_trace(go.Scatter3d(x=[ox / 1000], y=[oy / 1000], z=[oz + 60], mode="markers+text",
                               text=["댐"], textposition="top center",
                               marker=dict(size=6, color="#111827", symbol="diamond"), name="댐"))
    xr, yr = x.max() - x.min(), y.max() - y.min()
    zr = (zmax - zmin) / 1000.0 * exaggeration
    m = max(xr, yr)
    fig.update_layout(
        title=title, height=height, margin=dict(l=0, r=0, t=40 if title else 0, b=0), showlegend=False,
        scene=dict(aspectmode="manual", aspectratio=dict(x=xr / m, y=yr / m, z=max(zr / m, 0.05)),
                   xaxis=dict(title="", showbackground=False, showticklabels=False, showgrid=False),
                   yaxis=dict(title="", showbackground=False, showticklabels=False, showgrid=False),
                   zaxis=dict(title="표고 (m)", showbackground=False, showgrid=False),
                   camera=dict(eye=dict(x=-0.55, y=-0.85, z=0.5), center=dict(x=0, y=0, z=-0.1))))
    return fig
