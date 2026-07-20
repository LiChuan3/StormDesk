"""Quantified physics audit of a candidate TC forecast.

Checks are classical TC forecasting constraints:
- translation speed cap between consecutive forecast positions
- intensification-rate envelope (cf. RI climatology, Kaplan & DeMaria 2003)
- potential-intensity ceiling from SST (DeMaria-Kaplan MPI)
- cold-SST / high-shear intensification implausibility (SHIPS predictors)
- overland intensification (Kaplan-DeMaria decay expects weakening)
"""
from __future__ import annotations

import numpy as np

from ..geo import gc_distance_km
from .. import config as C


def _landmask_lookup(sst_crop: np.ndarray | None, lat0: float, lon0: float,
                     lat: float, lon: float):
    """True if (lat,lon) falls on land inside the +-10 deg SST crop, None if
    outside the crop or no crop available."""
    if sst_crop is None:
        return None
    di = int(round((lat0 - lat) / 0.25)) + 40   # crop rows go N->S
    dj = int(round(((lon - lon0 + 540) % 360 - 180) / 0.25)) + 40
    if not (0 <= di < 80 and 0 <= dj < 80):
        return None
    v = sst_crop[di, dj]
    return bool(~np.isfinite(v) or v < 200.0)


def audit_forecast(forecast: dict, init: dict, diag: dict,
                   sst_crop: np.ndarray | None = None) -> list[dict]:
    """forecast: {lead(str|int): {lat, lon, vmax}}; init: dict(lat, lon, vmax).

    Returns a list of issues: dict(lead, code, severity, message).
    """
    issues = []
    leads = sorted(int(k) for k in forecast.keys())
    prev = dict(lat=init["lat"], lon=init["lon"], vmax=init["vmax"], lead=0)
    mpi = diag.get("mpi_kt")
    sst = diag.get("sst_c")
    shear = diag.get("shear_kt")

    for l in leads:
        e = forecast.get(str(l), forecast.get(l))
        if e is None or e.get("vmax") is None:
            continue
        dt = l - prev["lead"]
        dist = float(gc_distance_km(prev["lat"], prev["lon"], e["lat"], e["lon"]))
        speed = dist / max(dt, 1e-9)
        if speed > 55.0:
            issues.append(dict(lead=l, code="translation_speed", severity="violation",
                               message=f"implied motion {speed*24:.0f} km/day exceeds plausible bound"))
        dv = e["vmax"] - prev["vmax"]
        rate = dv * 24.0 / max(dt, 1e-9)
        if rate > 55.0:
            issues.append(dict(lead=l, code="intensification_rate", severity="violation",
                               message=f"+{rate:.0f} kt/24h exceeds the extreme-RI envelope (~50 kt/24h)"))
        elif rate > 40.0:
            issues.append(dict(lead=l, code="intensification_rate", severity="warning",
                               message=f"+{rate:.0f} kt/24h is extreme RI; requires strong support"))
        if rate < -80.0:
            land = _landmask_lookup(sst_crop, init["lat"], init["lon"], e["lat"], e["lon"])
            if land is not True:
                issues.append(dict(lead=l, code="collapse_rate", severity="warning",
                                   message=f"{rate:.0f} kt/24h weakening over water is unusually fast"))
        if mpi is not None and e["vmax"] > mpi + 20:
            issues.append(dict(lead=l, code="mpi_ceiling", severity="violation",
                               message=f"vmax {e['vmax']:.0f} kt exceeds SST-based MPI {mpi:.0f} kt"))
        if sst is not None and sst < 24.0 and rate > 5.0:
            issues.append(dict(lead=l, code="cold_sst", severity="violation",
                               message=f"intensification over {sst:.1f}C SST is implausible"))
        if shear is not None and shear > 30.0 and rate > 15.0 and l <= 24:
            issues.append(dict(lead=l, code="high_shear", severity="warning",
                               message=f"+{rate:.0f} kt/24h under {shear:.0f} kt shear is unlikely"))
        land = _landmask_lookup(sst_crop, init["lat"], init["lon"], e["lat"], e["lon"])
        if land is True and dv > 5:
            issues.append(dict(lead=l, code="overland_intensification", severity="violation",
                               message="storm intensifies while center is over land"))
        prev = dict(lat=e["lat"], lon=e["lon"], vmax=e["vmax"], lead=l)
    return issues


def apply_hard_caps(forecast: dict, init: dict, diag: dict) -> dict:
    """Deterministic guardrails applied after revision (belt and braces)."""
    mpi = diag.get("mpi_kt")
    out = {}
    prev_v = init["vmax"]
    prev_l = 0
    for l in sorted(int(k) for k in forecast.keys()):
        e = dict(forecast.get(str(l), forecast.get(l)))
        if e.get("vmax") is not None:
            dt = l - prev_l
            cap_up = prev_v + 60.0 * dt / 24.0
            cap_dn = prev_v - 90.0 * dt / 24.0
            v = float(np.clip(e["vmax"], max(cap_dn, 15.0), cap_up))
            if mpi is not None:
                v = min(v, mpi + 15.0)
            e["vmax"] = round(v, 1)
            prev_v = e["vmax"]
            prev_l = l
        out[str(l)] = e
    return out
