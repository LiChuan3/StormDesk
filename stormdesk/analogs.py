"""Historical analog retrieval over the 1980-2015 library.

The analog tool mirrors the classical analog technique of human forecasters:
find historical storms whose state (position, season, intensity, trend,
motion) and environment (shear, SST, mid-level moisture, steering) resemble
the current storm, and summarize how those storms actually evolved.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from .config import LEADS_H, work_dir
from .geo import gc_distance_km, motion_uv_kmh

# feature name -> weight (weights chosen so that state and environment both
# matter; distances are computed in z-score space)
FEATURES = {
    "lat": 1.0, "abslat": 0.5, "doy_sin": 0.7, "doy_cos": 0.7,
    "vmax": 1.2, "dv12": 1.0, "mot_u": 0.8, "mot_v": 0.8,
    "shear_kt": 1.0, "sst_c": 1.0, "rh_mid_pct": 0.7,
    "steer_u": 0.6, "steer_v": 0.6,
}


def entry_features(lat, lon, vmax, dv12, mot_u, mot_v, diag: dict, init_time) -> dict | None:
    t = pd.Timestamp(init_time)
    doy = t.dayofyear
    sst = diag.get("sst_c")
    rh = diag.get("rh_mid_pct")
    f = dict(
        lat=lat, abslat=abs(lat),
        doy_sin=np.sin(2 * np.pi * doy / 365.25), doy_cos=np.cos(2 * np.pi * doy / 365.25),
        vmax=vmax, dv12=dv12, mot_u=mot_u, mot_v=mot_v,
        shear_kt=diag.get("shear_kt"), sst_c=sst if sst is not None else 27.0,
        rh_mid_pct=rh if rh is not None else 55.0,
        steer_u=diag.get("steering_u_ms"), steer_v=diag.get("steering_v_ms"),
    )
    if any(v is None or (isinstance(v, float) and not np.isfinite(v)) for v in f.values()):
        return None
    return f


class AnalogLibrary:
    def __init__(self, records: list[dict], stats: dict):
        self.records = records
        self.stats = stats
        names = list(FEATURES)
        self.X = np.array([[r["features"][k] for k in names] for r in records])
        self.w = np.array([FEATURES[k] for k in names])
        mu = np.array([stats[k][0] for k in names])
        sd = np.array([stats[k][1] for k in names])
        self.Xz = (self.X - mu) / sd
        self.mu, self.sd, self.names = mu, sd, names
        self.hemis = np.array([1.0 if r["lat"] >= 0 else -1.0 for r in records])
        self.sids = np.array([r["sid"] for r in records])

    @staticmethod
    def build(records: list[dict]) -> "AnalogLibrary":
        names = list(FEATURES)
        X = np.array([[r["features"][k] for k in names] for r in records])
        stats = {k: (float(X[:, i].mean()), float(X[:, i].std() + 1e-9)) for i, k in enumerate(names)}
        return AnalogLibrary(records, stats)

    def query(self, features: dict, lat: float, sid_exclude: str, k: int = 15) -> list[dict]:
        q = np.array([features[n] for n in self.names])
        qz = (q - self.mu) / self.sd
        d = np.sqrt(((self.Xz - qz) ** 2 * self.w).sum(1))
        d = d + np.where(self.hemis == (1.0 if lat >= 0 else -1.0), 0.0, 3.0)
        d = d + np.where(self.sids == sid_exclude, 1e6, 0.0)
        idx = np.argsort(d)[: k * 3]
        picked, seen = [], {}
        for i in idx:  # at most 2 samples per historical storm for diversity
            sid = self.records[i]["sid"]
            if seen.get(sid, 0) >= 2:
                continue
            seen[sid] = seen.get(sid, 0) + 1
            picked.append((i, float(d[i])))
            if len(picked) >= k:
                break
        out = []
        for i, dist in picked:
            r = self.records[i]
            out.append(dict(sid=r["sid"], name=r["name"], time=r["time"],
                            basin=r["basin"], lat=r["lat"], lon=r["lon"],
                            vmax=r["vmax"], similarity=round(float(np.exp(-dist / 4)), 3),
                            outcome=r["outcome"]))
        return out

    def save(self, path=None):
        path = path or os.path.join(work_dir("models"), "analog_library.pkl")
        with open(path, "wb") as f:
            pickle.dump(dict(records=self.records, stats=self.stats), f, protocol=4)

    @staticmethod
    def load(path=None) -> "AnalogLibrary":
        path = path or os.path.join(work_dir("models"), "analog_library.pkl")
        with open(path, "rb") as f:
            d = pickle.load(f)
        return AnalogLibrary(d["records"], d["stats"])


def summarize_analogs(analogs: list[dict], vmax_now: float) -> dict:
    """Aggregate analog outcomes into forecast-relevant statistics."""
    if not analogs:
        return {}
    out = {"n": len(analogs)}
    for l in (24, 48, 72):
        dvs = [a["outcome"].get(f"dv_{l}") for a in analogs]
        dvs = [x for x in dvs if x is not None]
        if dvs:
            out[f"dv{l}_median"] = round(float(np.median(dvs)), 1)
            out[f"dv{l}_p25"] = round(float(np.percentile(dvs, 25)), 1)
            out[f"dv{l}_p75"] = round(float(np.percentile(dvs, 75)), 1)
    ri = [1 for a in analogs if (a["outcome"].get("dv_24") or 0) >= 30]
    out["ri24_rate"] = round(len(ri) / len(analogs), 2)
    return out
