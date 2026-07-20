"""AIWP guidance: FengWu (cached 6-24 h + ONNX continuation to 72 h) and
Pangu-Weather (24-h model chained). Fields are tracked immediately and
discarded; only per-lead track/intensity summaries are stored.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd

from ..config import get_paths, LEADS_H, LEVMAJ_TO_VARMAJ, VARMAJ_TO_LEVMAJ
from ..tracker import track_rollout

_FMT = "%Y%m%d_%H"


def _ts(t) -> pd.Timestamp:
    return pd.Timestamp(t)


def fengwu_cache_file(init: pd.Timestamp) -> str:
    p = get_paths()
    return os.path.join(p.fengwu_cache, f"{init.strftime(_FMT)}_{(init + pd.Timedelta(hours=6)).strftime(_FMT)}.npy")


def era5_file(t: pd.Timestamp) -> str:
    return os.path.join(get_paths().era5_cache, f"{t.strftime(_FMT)}.npy")


class FengWuRunner:
    """Reads the 4-lead cache, then continues the rollout with fengwu_v2.onnx.

    FengWu v2 consumes two consecutive normalized 69-channel states
    (t-6h, t) concatenated on the channel axis and emits the next state
    (normalized) as output[:, :69]. The model and its data_mean/data_std use
    the official variable-major channel layout, whereas the cache stores
    level-major states; we permute at the model boundary.
    """

    def __init__(self, device_id: int = 0):
        self.p = get_paths()
        self.mean = np.load(self.p.fengwu_mean)[:, None, None].astype(np.float32)
        self.std = np.load(self.p.fengwu_std)[:, None, None].astype(np.float32)
        self.session = None
        self.device_id = device_id

    def _lazy_session(self):
        if self.session is None:
            import onnxruntime as ort
            opt = ort.SessionOptions()
            opt.enable_cpu_mem_arena = False
            opt.enable_mem_pattern = False
            self.session = ort.InferenceSession(
                self.p.fengwu_onnx, sess_options=opt,
                providers=[("CUDAExecutionProvider", {"device_id": self.device_id}),
                           "CPUExecutionProvider"])
        return self.session

    def forecast(self, init, lat0: float, lon0: float, max_lead: int = 72,
                 motion0: tuple[float, float] | None = None) -> dict | None:
        """Autoregressive rollout from the ERA5 analysis pair (t-6h, t)."""
        from ..geo import motion_uv_kmh
        init = _ts(init)
        f_prev = era5_file(init - pd.Timedelta(hours=6))
        f_now = era5_file(init)
        if not (os.path.exists(f_prev) and os.path.exists(f_now)):
            return self._forecast_from_cache(init, lat0, lon0, max_lead, motion0)
        sess = self._lazy_session()
        s_prev = np.load(f_prev).astype(np.float32)
        s_now = np.load(f_now).astype(np.float32)
        x = np.concatenate([(s_prev[LEVMAJ_TO_VARMAJ] - self.mean) / self.std,
                            (s_now[LEVMAJ_TO_VARMAJ] - self.mean) / self.std],
                           axis=0)[None].astype(np.float32)
        del s_prev, s_now
        gc.collect()
        leads = {}
        lat, lon = lat0, lon0
        m = motion0
        lead = 0
        while lead < max_lead:
            out = sess.run(None, {"input": x})[0]
            x = np.concatenate([x[:, 69:], out[:, :69]], axis=1)
            lead += 6
            field = (out[0, :69] * self.std + self.mean)[VARMAJ_TO_LEVMAJ]
            r = track_rollout([field], lat, lon, motion0=m)[0]
            leads[lead] = r
            mu = motion_uv_kmh(lat, lon, r["lat"], r["lon"], 6.0)
            m = (float(mu[0]), float(mu[1]))
            lat, lon = r["lat"], r["lon"]
            del out, field
        return {str(k): v for k, v in leads.items()}

    def _forecast_from_cache(self, init, lat0, lon0, max_lead, motion0):
        """Fallback: 4-lead cache (init from an earlier operational-style run)."""
        f = fengwu_cache_file(init)
        if not os.path.exists(f):
            return None
        frames = np.load(f)
        tracks = track_rollout(list(frames), lat0, lon0, motion0=motion0)
        leads = {6 * (i + 1): t for i, t in enumerate(tracks)}
        del frames
        gc.collect()
        return {str(k): v for k, v in leads.items()}


PANGU_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


def era5_to_pangu(state: np.ndarray):
    """Level-major 69-channel state -> Pangu (upper, surface) inputs.

    Pangu upper: (5,13,721,1440) vars Z,Q,T,U,V, levels 1000->50 hPa.
    Pangu surface: (4,721,1440) = MSLP, U10, V10, T2M.
    Our stack: surface [u10,v10,t2m,msl]; per level (50->1000): z,q,u,v,t.
    """
    lev = state[4:].reshape(13, 5, 721, 1440)   # levels 50->1000, vars z,q,u,v,t
    lev = lev[::-1]                              # -> 1000->50
    upper = np.stack([lev[:, 0], lev[:, 1], lev[:, 4], lev[:, 2], lev[:, 3]])
    surface = state[[3, 0, 1, 2]]
    return np.ascontiguousarray(upper, dtype=np.float32), \
        np.ascontiguousarray(surface, dtype=np.float32)


def pangu_to_era5(upper: np.ndarray, surface: np.ndarray) -> np.ndarray:
    out = np.empty((69, 721, 1440), dtype=np.float32)
    out[3], out[0], out[1], out[2] = surface[0], surface[1], surface[2], surface[3]
    lev = np.stack([upper[0], upper[1], upper[3], upper[4], upper[2]], axis=1)
    out[4:] = lev[::-1].reshape(65, 721, 1440)
    return out


class PanguRunner:
    """Chains the 24-h Pangu model for leads 24/48/72 h."""

    def __init__(self, device_id: int = 0):
        self.p = get_paths()
        self.session = None
        self.device_id = device_id

    def _lazy_session(self):
        if self.session is None:
            import onnxruntime as ort
            opt = ort.SessionOptions()
            opt.enable_cpu_mem_arena = False
            opt.enable_mem_pattern = False
            self.session = ort.InferenceSession(
                os.path.join(self.p.pangu_dir, "pangu_weather_24.onnx"),
                sess_options=opt,
                providers=[("CUDAExecutionProvider", {"device_id": self.device_id}),
                           "CPUExecutionProvider"])
        return self.session

    def forecast(self, init, lat0: float, lon0: float, max_lead: int = 72,
                 motion0: tuple[float, float] | None = None) -> dict | None:
        init = _ts(init)
        f = era5_file(init)
        if not os.path.exists(f):
            return None
        state = np.load(f).astype(np.float32)
        upper, surface = era5_to_pangu(state)
        del state
        sess = self._lazy_session()
        leads = {}
        lat, lon = lat0, lon0
        m = motion0
        lead = 0
        while lead < max_lead:
            upper, surface = sess.run(None, {"input": upper, "input_surface": surface})
            lead += 24
            field = pangu_to_era5(upper, surface)
            r = track_rollout([field], lat, lon, dt_h=24.0, motion0=m)[0]
            leads[lead] = r
            from ..geo import motion_uv_kmh
            mu = motion_uv_kmh(lat, lon, r["lat"], r["lon"], 24.0)
            m = (float(mu[0]), float(mu[1]))
            lat, lon = r["lat"], r["lon"]
            del field
            gc.collect()
        return {str(k): v for k, v in leads.items()}
