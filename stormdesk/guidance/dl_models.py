"""Track/intensity deep-learning specialists (GRU / Transformer seq2seq).

Trained on 1980-2015 with 2016-2017 validation (strict year split, no
leakage into the 2021-2022 test seasons). Inputs: 24 h of 6-hourly history
(lat, lon, vmax, pres) plus static context; outputs displacement and
intensity-change offsets for the 12 leads (6..72 h).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import LEADS_H, INPUT_STEPS, work_dir
from ..geo import wrap_lon

BASINS = ["NA", "EP", "WP", "NI", "SI", "SP", "SA"]
N_STATIC = 7 + len(BASINS)
N_SEQ = 6  # dlat, dlon, vmax/100, pres_anom/50, sin(hour), cos(hour)


def case_tensors(history: list[dict], lat: float, lon: float, vmax: float,
                 basin: str, init_time) -> tuple[np.ndarray, np.ndarray] | None:
    if len(history) < INPUT_STEPS or any(h["vmax"] is None for h in history):
        return None
    seq = []
    for h in history:
        t = pd.Timestamp(h["time"])
        pres = h["pres"] if h["pres"] is not None else 1005.0
        seq.append([h["lat"] - lat, wrap_lon(h["lon"] - lon).item(),
                    h["vmax"] / 100.0, (pres - 1000.0) / 50.0,
                    np.sin(2 * np.pi * t.hour / 24), np.cos(2 * np.pi * t.hour / 24)])
    t0 = pd.Timestamp(init_time)
    doy = t0.dayofyear
    static = [lat / 45.0, np.sin(np.radians(lon)), np.cos(np.radians(lon)),
              vmax / 100.0, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
              1.0 if lat >= 0 else -1.0]
    static += [1.0 if basin == b else 0.0 for b in BASINS]
    return np.array(seq, dtype=np.float32), np.array(static, dtype=np.float32)


class GRUSpecialist(nn.Module):
    def __init__(self, hidden: int = 192, layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(N_SEQ, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden + N_STATIC, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, len(LEADS_H) * 3))

    def forward(self, seq, static):
        _, h = self.rnn(seq)
        z = torch.cat([h[-1], static], dim=1)
        return self.head(z).view(-1, len(LEADS_H), 3)


class TransformerSpecialist(nn.Module):
    def __init__(self, d: int = 128, heads: int = 4, layers: int = 3):
        super().__init__()
        self.embed = nn.Linear(N_SEQ, d)
        self.pos = nn.Parameter(torch.randn(1, INPUT_STEPS, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(
            nn.Linear(d + N_STATIC, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, len(LEADS_H) * 3))

    def forward(self, seq, static):
        z = self.encoder(self.embed(seq) + self.pos).mean(dim=1)
        z = torch.cat([z, static], dim=1)
        return self.head(z).view(-1, len(LEADS_H), 3)


# target scaling: dlat deg, dlon deg, dv kt/50
def targets_from_case(row) -> np.ndarray:
    y = np.full((len(LEADS_H), 3), np.nan, dtype=np.float32)
    for k, l in enumerate(LEADS_H):
        la, lo, v = row[f"lat_{l}"], row[f"lon_{l}"], row[f"vmax_{l}"]
        if np.isfinite(la):
            y[k, 0] = la - row["lat"]
            y[k, 1] = wrap_lon(lo - row["lon"]).item()
        if np.isfinite(v):
            y[k, 2] = (v - row["vmax"]) / 50.0
    return y


def masked_huber(pred, target):
    mask = torch.isfinite(target)
    t = torch.where(mask, target, torch.zeros_like(target))
    loss = nn.functional.huber_loss(pred, t, reduction="none", delta=1.0)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def predict_case(model, history, lat, lon, vmax, basin, init_time, device="cpu") -> dict | None:
    tens = case_tensors(history, lat, lon, vmax, basin, init_time)
    if tens is None:
        return None
    seq, static = tens
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(seq)[None].to(device),
                    torch.from_numpy(static)[None].to(device))[0].cpu().numpy()
    res = {}
    for k, l in enumerate(LEADS_H):
        res[l] = dict(lat=round(lat + float(out[k, 0]), 2),
                      lon=round(wrap_lon(lon + float(out[k, 1])).item(), 2),
                      vmax=round(max(vmax + float(out[k, 2]) * 50.0, 10.0), 1))
    return res


def save_model(model, name: str):
    path = os.path.join(work_dir("models"), f"{name}.pt")
    torch.save(model.state_dict(), path)
    return path


def load_model(name: str, device="cpu"):
    path = os.path.join(work_dir("models"), f"{name}.pt")
    model = GRUSpecialist() if name.startswith("gru") else TransformerSpecialist()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)
