"""Environmental diagnostics from storm-centered 80x80 (0.25 deg) ERA5 crops.

Conventions follow SHIPS-style predictors (DeMaria & Kaplan 1994; DeMaria et
al. 2005): deep-layer shear over the 200-800 km annulus, mid-level RH over
200-800 km, steering as the 850-300 hPa mass-weighted mean wind within 500 km,
SST within 200 km of the core, and potential intensity from the DeMaria-Kaplan
empirical SST relation.
"""
from __future__ import annotations

import os

import numpy as np

from . import config as C
from .config import KT_PER_MS

GRID = 80
HALF = GRID // 2  # crop is centered: index HALF is the storm center


def _dist_grid_km(lat0: float) -> np.ndarray:
    """Distance (km) of every crop pixel from the crop center."""
    dy = (np.arange(GRID) - HALF)[:, None] * 0.25 * C.DEG2KM
    dx = (np.arange(GRID) - HALF)[None, :] * 0.25 * C.DEG2KM * np.cos(np.radians(lat0))
    return np.hypot(np.broadcast_to(dy, (GRID, GRID)), np.broadcast_to(dx, (GRID, GRID)))


def _ring_mean(a: np.ndarray, dist: np.ndarray, r0: float, r1: float) -> float:
    m = (dist >= r0) & (dist <= r1) & np.isfinite(a)
    return float(a[m].mean()) if m.any() else float("nan")


def load_crop(sid: str, season: int, time, kind: str = "ERA5") -> np.ndarray | None:
    from .ibtracs import crop_dir
    d = crop_dir(sid, season, time)
    f = os.path.join(d, f"{kind}_data.npy")
    if os.path.exists(f):
        try:
            return np.load(f)
        except Exception:
            return None
    return None


def compute_diagnostics(era5: np.ndarray, lat0: float, sup: np.ndarray | None = None,
                        vmax_kt: float | None = None) -> dict:
    """SHIPS-class environmental diagnostics from one (69,80,80) crop.

    era5: 69-channel crop; sup: optional (27,80,80) SUPPLEMENT crop.
    Wind diagnostics are returned in kt, distances in km, SST in Celsius.
    """
    dist = _dist_grid_km(lat0)

    u200, v200 = era5[C.crop_u(200)], era5[C.crop_v(200)]
    u850, v850 = era5[C.crop_u(850)], era5[C.crop_v(850)]
    su = _ring_mean(u200, dist, 200, 800) - _ring_mean(u850, dist, 200, 800)
    sv = _ring_mean(v200, dist, 200, 800) - _ring_mean(v850, dist, 200, 800)
    shear_kt = float(np.hypot(su, sv) * KT_PER_MS)
    shear_dir = float((np.degrees(np.arctan2(su, sv)) + 360) % 360)

    # steering: mass-weighted 850-300 mean wind within 500 km
    levs = [850, 700, 600, 500, 400, 300]
    w = np.array([75.0, 125.0, 100.0, 100.0, 100.0, 75.0])
    us = np.array([_ring_mean(era5[C.crop_u(l)], dist, 0, 500) for l in levs])
    vs = np.array([_ring_mean(era5[C.crop_v(l)], dist, 0, 500) for l in levs])
    su_ms = float((us * w).sum() / w.sum())
    sv_ms = float((vs * w).sum() / w.sum())
    steer_speed_kmh = float(np.hypot(su_ms, sv_ms) * 3.6)
    steer_dir = float((np.degrees(np.arctan2(su_ms, sv_ms)) + 360) % 360)

    # 200 hPa divergence within 1000 km (finite differences, per second)
    dx = 0.25 * C.DEG2KM * 1000.0 * np.cos(np.radians(lat0))
    dy = 0.25 * C.DEG2KM * 1000.0
    div = np.gradient(u200, dx, axis=1) - np.gradient(v200, dy, axis=0)  # lat axis descends
    div200 = _ring_mean(div * 1e7, dist, 0, 1000)  # 1e-7 s^-1

    msl = era5[C.CH_MSL] / 100.0
    p_env = _ring_mean(msl, dist, 800, 1000)
    p_min = float(np.nanmin(np.where(dist <= 250, msl, np.nan)))

    out = dict(
        shear_kt=round(shear_kt, 1), shear_dir_deg=round(shear_dir),
        steering_speed_kmh=round(steer_speed_kmh, 1), steering_dir_deg=round(steer_dir),
        steering_u_ms=round(su_ms, 2), steering_v_ms=round(sv_ms, 2),
        div200_1e7=round(float(div200), 1),
        p_env_hpa=round(p_env, 1), p_min_hpa=round(p_min, 1),
        p_deficit_hpa=round(p_env - p_min, 1),
    )

    if sup is not None:
        sst = sup[C.SUP_SST]
        sst_c = _ring_mean(np.where(sst > 200, sst - 273.15, np.nan), dist, 0, 200)
        rh_mid = np.nanmean([_ring_mean(sup[C.sup_r(l)], dist, 200, 800) for l in (700, 600, 500)])
        w500 = _ring_mean(sup[C.sup_w(500)], dist, 0, 500)
        out.update(sst_c=round(float(sst_c), 2) if np.isfinite(sst_c) else None,
                   rh_mid_pct=round(float(rh_mid), 1) if np.isfinite(rh_mid) else None,
                   w500_pa_s=round(float(w500), 3) if np.isfinite(w500) else None)
        if np.isfinite(sst_c):
            # DeMaria-Kaplan (1994) empirical maximum potential intensity
            mpi_kt = 38.21 + 170.72 * np.exp(0.1909 * (min(sst_c, 30.5) - 30.0))
            out["mpi_kt"] = round(float(mpi_kt), 1)
            if vmax_kt is not None:
                out["pot_kt"] = round(float(mpi_kt) - vmax_kt, 1)
    return out


def compute_satellite(gridsat: np.ndarray) -> dict:
    """Convective descriptors from the (3,286,286) GridSat-B1 crop (~0.07 deg)."""
    ir = gridsat[0]
    n = ir.shape[0]
    c = n // 2
    px_km = 0.07 * C.DEG2KM
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.hypot(yy - c, xx - c) * px_km

    def ring(r0, r1):
        m = (dist >= r0) & (dist <= r1) & np.isfinite(ir)
        return ir[m] if m.any() else np.array([np.nan])

    core = ring(0, 100)
    mid = ring(0, 200)
    out = dict(
        bt_min_k=round(float(np.nanmin(ir)), 1) if np.isfinite(ir).any() else None,
        bt_core_mean_k=round(float(np.nanmean(core)), 1),
        cold_frac_208k=round(float(np.mean(mid < 208.0)), 3),
        eye_warm_k=round(float(np.nanmax(ring(0, 40))), 1),
    )
    # axisymmetry: spread of quadrant-mean core BT (smaller = more symmetric)
    quads = []
    for sy in (slice(0, c), slice(c, n)):
        for sx in (slice(0, c), slice(c, n)):
            q = ir[sy, sx]
            d = dist[sy, sx]
            m = (d <= 200) & np.isfinite(q)
            quads.append(float(q[m].mean()) if m.any() else np.nan)
    out["quadrant_bt_std_k"] = round(float(np.nanstd(quads)), 2)
    return out
