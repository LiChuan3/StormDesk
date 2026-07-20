"""Consensus baselines over guidance members and bias calibration.

- equal-weight consensus (the operational default, cf. NHC TVCN/IVCN)
- performance-weighted consensus (inverse-MSE weights fit on 2018-2020)
- bias-corrected consensus (per member/lead linear intensity correction)
"""
from __future__ import annotations

import json
import os

import numpy as np

from .config import work_dir
from .geo import wrap_lon, gc_distance_km

VERIF = [24, 48, 72]


def _members_at(guidance: dict, lead: int) -> dict:
    out = {}
    for m, fc in guidance.items():
        if fc and str(lead) in fc and fc[str(lead)] is not None:
            out[m] = fc[str(lead)]
        elif fc and lead in fc and fc[lead] is not None:
            out[m] = fc[lead]
    return out


def _mean_position(entries: list[dict]) -> tuple[float, float]:
    lats = np.array([e["lat"] for e in entries])
    lons = np.array([e["lon"] for e in entries])
    ref = lons[0]
    rel = (lons - ref + 540) % 360 - 180
    return float(lats.mean()), float(wrap_lon(ref + rel.mean()).item())


def consensus_equal(guidance: dict, leads=VERIF) -> dict:
    out = {}
    for l in leads:
        ms = _members_at(guidance, l)
        if not ms:
            continue
        lat, lon = _mean_position(list(ms.values()))
        vm = [e.get("vmax", e.get("vmax_kt")) for e in ms.values()]
        vm = [v for v in vm if v is not None and np.isfinite(v)]
        out[str(l)] = dict(lat=round(lat, 2), lon=round(lon, 2),
                           vmax=round(float(np.mean(vm)), 1) if vm else None)
    return out


class Calibration:
    """Per member/lead error stats and intensity bias maps from 2018-2020."""

    def __init__(self, table: dict):
        self.table = table  # member -> lead -> dict(track_rmse, v_bias, v_a, v_b, n)

    @staticmethod
    def fit(records: list[dict]) -> "Calibration":
        """records: per case/member/lead truth-vs-forecast pairs."""
        agg: dict = {}
        for r in records:
            key = (r["member"], r["lead"])
            agg.setdefault(key, []).append(r)
        table: dict = {}
        for (m, l), rows in agg.items():
            te = np.array([x["track_err_km"] for x in rows if np.isfinite(x["track_err_km"])])
            vf = np.array([[x["vmax_fc"], x["vmax_ob"]] for x in rows
                           if np.isfinite(x["vmax_fc"]) and np.isfinite(x["vmax_ob"])])
            ent = dict(n=len(rows))
            if len(te):
                ent["track_rmse"] = float(np.sqrt((te ** 2).mean()))
                ent["track_mae"] = float(te.mean())
            if len(vf) > 20:
                ent["v_bias"] = float((vf[:, 0] - vf[:, 1]).mean())
                ent["v_mae"] = float(np.abs(vf[:, 0] - vf[:, 1]).mean())
                b, a = np.polyfit(vf[:, 0], vf[:, 1], 1)  # ob ~ a + b*fc
                ent["v_a"], ent["v_b"] = float(a), float(np.clip(b, 0.5, 3.0))
            table.setdefault(m, {})[str(l)] = ent
        return Calibration(table)

    def correct_vmax(self, member: str, lead: int, v: float) -> float:
        e = self.table.get(member, {}).get(str(lead))
        if not e or "v_a" not in e:
            return v
        return float(np.clip(e["v_a"] + e["v_b"] * v, 15.0, 185.0))

    def weight(self, member: str, lead: int) -> float:
        # inverse squared MAE: robust to rare tracker blow-ups that inflate RMSE
        e = self.table.get(member, {}).get(str(lead))
        if not e or "track_mae" not in e or e.get("n", 0) < 20:
            return 0.0
        return 1.0 / (e["track_mae"] ** 2 + 1e-6)

    def save(self, path=None):
        path = path or os.path.join(work_dir("models"), "calibration.json")
        with open(path, "w") as f:
            json.dump(self.table, f, indent=1)

    @staticmethod
    def load(path=None) -> "Calibration":
        path = path or os.path.join(work_dir("models"), "calibration.json")
        with open(path) as f:
            return Calibration(json.load(f))


def consensus_weighted(guidance: dict, calib: Calibration, leads=VERIF,
                       correct_bias: bool = False) -> dict:
    out = {}
    for l in leads:
        ms = _members_at(guidance, l)
        if not ms:
            continue
        ws, lats, lons_rel, vs, vws = [], [], [], [], []
        ref = list(ms.values())[0]["lon"]
        for m, e in ms.items():
            w = calib.weight(m, l)
            if w <= 0:
                w = 1e-8
            ws.append(w)
            lats.append(e["lat"])
            lons_rel.append((e["lon"] - ref + 540) % 360 - 180)
            v = e.get("vmax", e.get("vmax_kt"))
            if v is not None and np.isfinite(v):
                vv = calib.correct_vmax(m, l, v) if correct_bias else v
                ve = calib.table.get(m, {}).get(str(l), {})
                vw = 1.0 / (ve.get("v_mae", 15.0) ** 2)
                vs.append(vv)
                vws.append(vw)
        ws = np.array(ws) / np.sum(ws)
        lat = float(np.dot(ws, lats))
        lon = float(wrap_lon(ref + np.dot(ws, lons_rel)).item())
        vmax = float(np.dot(np.array(vws) / np.sum(vws), vs)) if vs else None
        out[str(l)] = dict(lat=round(lat, 2), lon=round(lon, 2),
                           vmax=round(vmax, 1) if vmax is not None else None)
    return out
