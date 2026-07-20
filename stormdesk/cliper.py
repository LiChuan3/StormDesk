"""CLIPER5/SHIFOR-class statistical baseline (skill benchmark).

Ridge regression per lead on climatology-and-persistence predictors, the
standard no-skill reference for TC verification (track: CLIPER5, Neumann 1972;
intensity: SHIFOR5-class). Trained on 1980-2015, all basins.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from .config import LEADS_H, work_dir
from .geo import motion_uv_kmh

BASINS = ["NA", "EP", "WP", "NI", "SI", "SP", "SA"]


def _features(row_hist: list[dict], lat: float, lon: float, vmax: float,
              basin: str, doy: float) -> np.ndarray | None:
    h = row_hist
    if len(h) < 4 or any(x["vmax"] is None for x in h):
        return None
    u12, v12 = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
    u24, v24 = motion_uv_kmh(h[0]["lat"], h[0]["lon"], h[-1]["lat"], h[-1]["lon"], 18.0)
    dv12 = h[-1]["vmax"] - h[-3]["vmax"]
    dv24 = h[-1]["vmax"] - h[0]["vmax"]
    f = [lat, np.cos(np.radians(lon)), np.sin(np.radians(lon)), vmax, dv12, dv24,
         u12, v12, u24, v24, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
         lat * u12 / 100.0, lat * v12 / 100.0, vmax * dv12 / 100.0]
    f += [1.0 if basin == b else 0.0 for b in BASINS]
    return np.array(f, dtype=np.float64)


def build_matrix(cases: pd.DataFrame):
    X, Ys, keep = [], {l: [] for l in LEADS_H}, []
    for i, r in cases.iterrows():
        doy = pd.Timestamp(r["init"]).dayofyear
        f = _features(r["history"], r["lat"], r["lon"], r["vmax"], r["basin"], doy)
        if f is None:
            continue
        X.append(f)
        keep.append(i)
        for l in LEADS_H:
            Ys[l].append([r[f"lat_{l}"] - r["lat"],
                          (r[f"lon_{l}"] - r["lon"] + 540) % 360 - 180 if np.isfinite(r[f"lon_{l}"]) else np.nan,
                          r[f"vmax_{l}"] - r["vmax"]])
    return np.array(X), {l: np.array(v) for l, v in Ys.items()}, keep


class _Ridge:
    """Closed-form ridge regression with unpenalized intercept (numpy only)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mx, my = X.mean(0), y.mean()
        Xc, yc = X - mx, y - my
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, Xc.T @ yc)
        self.b = my - mx @ self.w
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w + self.b


class Cliper:
    def __init__(self):
        self.models: dict = {}
        self.mu = None
        self.sd = None

    def fit(self, cases: pd.DataFrame):
        X, Ys, _ = build_matrix(cases)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xz = (X - self.mu) / self.sd
        for l in LEADS_H:
            Y = Ys[l]
            for k, name in enumerate(["dlat", "dlon", "dv"]):
                m = np.isfinite(Y[:, k])
                r = _Ridge(alpha=1.0).fit(Xz[m], Y[m, k])
                self.models[(l, name)] = r
        return self

    def predict_case(self, history, lat, lon, vmax, basin, init_time) -> dict | None:
        doy = pd.Timestamp(init_time).dayofyear
        f = _features(history, lat, lon, vmax, basin, doy)
        if f is None:
            return None
        fz = (f - self.mu) / self.sd
        out = {}
        for l in LEADS_H:
            dlat = float(self.models[(l, "dlat")].predict([fz])[0])
            dlon = float(self.models[(l, "dlon")].predict([fz])[0])
            dv = float(self.models[(l, "dv")].predict([fz])[0])
            out[l] = dict(lat=lat + dlat, lon=(lon + dlon + 540) % 360 - 180,
                          vmax=max(vmax + dv, 10.0))
        return out

    def save(self, path=None):
        path = path or os.path.join(work_dir("models"), "cliper.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=None) -> "Cliper":
        path = path or os.path.join(work_dir("models"), "cliper.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
