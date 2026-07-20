"""Great-circle geometry helpers (vectorized, degrees in/out)."""
from __future__ import annotations

import numpy as np

R_EARTH_KM = 6371.0


def gc_distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (haversine). Accepts scalars or arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=np.float64)) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=np.float64)) for x in (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def destination(lat, lon, bearing, dist_km):
    """Destination point given start, bearing (deg) and distance (km)."""
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    brg = np.radians(np.asarray(bearing, dtype=np.float64))
    d = np.asarray(dist_km, dtype=np.float64) / R_EARTH_KM
    lat2 = np.arcsin(np.sin(lat) * np.cos(d) + np.cos(lat) * np.sin(d) * np.cos(brg))
    lon2 = lon + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(lat), np.cos(d) - np.sin(lat) * np.sin(lat2))
    return np.degrees(lat2), (np.degrees(lon2) + 540.0) % 360.0 - 180.0


def motion_uv_kmh(lat0, lon0, lat1, lon1, dt_h: float):
    """Storm motion as (u_east, v_north) km/h from position at t-dt to t."""
    dy = (np.asarray(lat1) - np.asarray(lat0)) * 111.194927
    dx = (np.asarray(lon1) - np.asarray(lon0) + 540.0) % 360.0 - 180.0
    dx = dx * 111.194927 * np.cos(np.radians((np.asarray(lat0) + np.asarray(lat1)) / 2))
    return dx / dt_h, dy / dt_h


def wrap_lon(lon):
    """Wrap longitude to [-180, 180)."""
    return (np.asarray(lon) + 540.0) % 360.0 - 180.0
