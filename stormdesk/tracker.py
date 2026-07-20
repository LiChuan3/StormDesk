"""Vortex tracker on global 69-channel fields (0.25 deg, 721x1440).

MSLP-minimum tracker with motion-extrapolated first guess and a bounded
search radius, following common AIWP TC-evaluation practice (DeMaria et al.
2025): from a first-guess position, find the sea-level-pressure minimum within
the search radius, then measure Vmax as the max 10-m wind within 2 deg.
"""
from __future__ import annotations

import numpy as np

from .config import CH_MSL, CH_U10, CH_V10, KT_PER_MS
from .geo import gc_distance_km, wrap_lon

LAT_AXIS = np.linspace(90.0, -90.0, 721)
LON_AXIS = np.linspace(0.0, 359.75, 1440)


def _window(lat0: float, lon0: float, half_deg: float):
    """Index windows (lat rows, lon cols) around a point; handles lon wrap."""
    i0 = int(round((90.0 - lat0) / 0.25))
    j0 = int(round((lon0 % 360.0) / 0.25)) % 1440
    h = int(round(half_deg / 0.25))
    ii = np.arange(max(i0 - h, 0), min(i0 + h + 1, 721))
    jj = np.arange(j0 - h, j0 + h + 1) % 1440
    return ii, jj


def track_step(field: np.ndarray, lat_g: float, lon_g: float,
               search_km: float = 450.0) -> dict:
    """Locate the TC in one global field near the first-guess position."""
    half = max(search_km / 111.0 * 1.3, 3.0)
    ii, jj = _window(lat_g, lon_g, half)
    msl = field[CH_MSL][np.ix_(ii, jj)]
    lats = LAT_AXIS[ii][:, None]
    lons = LON_AXIS[jj][None, :]
    dist = gc_distance_km(lats, lons, lat_g, lon_g % 360.0)
    masked = np.where(dist <= search_km, msl, np.inf)
    ki, kj = np.unravel_index(int(np.argmin(masked)), masked.shape)
    lat_c = float(lats[ki, 0])
    lon_c = float(lons[0, kj])
    mslp = float(masked[ki, kj]) / 100.0

    ii2, jj2 = _window(lat_c, lon_c, 2.0)
    u = field[CH_U10][np.ix_(ii2, jj2)]
    v = field[CH_V10][np.ix_(ii2, jj2)]
    vmax_ms = float(np.sqrt(u * u + v * v).max())
    return dict(lat=round(lat_c, 2), lon=round(wrap_lon(lon_c).item(), 2),
                mslp_hpa=round(mslp, 1), vmax_kt=round(vmax_ms * KT_PER_MS, 1))


def track_rollout(fields, lat0: float, lon0: float, dt_h: float = 6.0,
                  motion0: tuple[float, float] | None = None) -> list[dict]:
    """Track a storm through successive forecast fields.

    fields: iterable of (69,721,1440) arrays at successive lead times, spaced
    dt_h hours apart. motion0: optional initial (u_east, v_north) km/h motion
    used to extrapolate the first-guess position.
    """
    out = []
    lat, lon = float(lat0), float(lon0)
    u, v = motion0 if motion0 is not None else (0.0, 0.0)
    for f in fields:
        glat = float(np.clip(lat + v * dt_h / 111.19, -85.0, 85.0))
        glon = lon + u * dt_h / (111.19 * max(np.cos(np.radians(lat)), 0.2))
        search = float(np.clip(200.0 + 0.7 * np.hypot(u, v) * dt_h, 350.0, 850.0))
        r = track_step(f, glat, glon, search_km=search)
        dlat = r["lat"] - lat
        dlon = (r["lon"] - lon + 540.0) % 360.0 - 180.0
        v = dlat * 111.19 / dt_h
        u = dlon * 111.19 * max(np.cos(np.radians(lat)), 0.2) / dt_h
        lat, lon = r["lat"], r["lon"]
        out.append(r)
    return out
