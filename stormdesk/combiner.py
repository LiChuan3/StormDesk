"""Supervised combiner reference baselines (learned adaptive combination).

Provides the shared machinery for the review's decomposition of the learned
combiner's gain into (a) a calibration-optimal *static* per-lead weighting and
(b) genuine *case adaptation*:

  static_convex   per-lead convex weights minimizing calibration track error
                  (NNLS-style, simplex-constrained) -- the honest static ceiling
  gbt_static      GBT gate using only member identity, historical MAE and lead
                  (no case features) -- learnable static reweighting
  gbt_case        full GBT gate with case + member-specific features
  gbt_shuffled    full features but case features shuffled across cases
                  (destroys case signal, keeps member/marginal structure)

and the intensity side (static convex stack over bias-corrected members; GBT
residual stack, optionally clipped to the office's +-25 kt contract with the
office rate/MPI/hard caps and an independently fitted shrinkage).

All models are fit strictly on the calibration seasons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import Calibration, _members_at
from .evaluate import case_id
from .geo import gc_distance_km, motion_uv_kmh, wrap_lon

VERIF = [24, 48, 72]
# primary guidance members (excludes the derived AIWP-consensus member by
# default so the learned combiner is not handed a pre-averaged input)
PRIMARY = ["pangu", "fengwu", "gru", "transformer", "cliper"]
MEMBERS_ALL = ["pangu", "fengwu", "cons_aiwp", "gru", "transformer", "cliper"]

CASE_KEYS = ["v0", "dv12", "abslat", "mot_u", "mot_v", "shear", "sst", "rh",
             "pot", "div"]
MEM_KEYS = ["hist_mae", "dist_cluster", "motion_dev_deg", "speed_ratio", "disp24"]


# ---------------------------------------------------------------------------
# feature construction
# ---------------------------------------------------------------------------
def case_features(r, feats) -> dict:
    ft = feats.get(case_id(r), {})
    diag = ft.get("diag") or {}
    h = r["history"]
    mu, mv = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
    dv12 = (r["vmax"] - h[-3]["vmax"]) if h[-3]["vmax"] is not None else 0.0
    return dict(v0=r["vmax"], dv12=dv12, abslat=abs(r["lat"]),
                mot_u=float(mu), mot_v=float(mv),
                shear=diag.get("shear_kt"), sst=diag.get("sst_c"),
                rh=diag.get("rh_mid_pct"), pot=diag.get("pot_kt"),
                div=diag.get("div200_1e7"))


def member_rows(r, cf, g, calib, lead, members):
    ms = {m: e for m, e in _members_at(g, lead).items() if m in members}
    if len(ms) < 2:
        return []
    lat_c = float(np.mean([e["lat"] for e in ms.values()]))
    lon_c = float(np.mean([e["lon"] for e in ms.values()]))
    h = r["history"]
    mu, mv = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
    obs_dir = float(np.degrees(np.arctan2(mu, mv)) % 360)
    obs_spd = float(np.hypot(mu, mv))
    rows = []
    for name, e in ms.items():
        e24 = (g.get(name) or {}).get("24") or (g.get(name) or {}).get(24)
        if e24 is not None:
            du = ((e24["lon"] - r["lon"] + 540) % 360 - 180) * 111.19 * \
                max(np.cos(np.radians(r["lat"])), 0.2) / 24.0
            dv = (e24["lat"] - r["lat"]) * 111.19 / 24.0
            m_dir = float(np.degrees(np.arctan2(du, dv)) % 360)
            m_spd = float(np.hypot(du, dv))
            dev = abs((m_dir - obs_dir + 180) % 360 - 180)
            ratio = m_spd / max(obs_spd, 1.0)
            disp = float(gc_distance_km(r["lat"], r["lon"], e24["lat"], e24["lon"]))
        else:
            dev, ratio, disp = np.nan, np.nan, np.nan
        rows.append(dict(
            member=name, lat=e["lat"], lon=e["lon"],
            hist_mae=(calib.table.get(name, {}).get(str(lead), {}) or {}).get("track_mae"),
            dist_cluster=float(gc_distance_km(e["lat"], e["lon"], lat_c, lon_c)),
            motion_dev_deg=dev, speed_ratio=ratio, disp24=disp, **cf))
    return rows


def build_track_table(cases, feats, guidance, calib, members, with_truth=True):
    recs = []
    for _, r in cases.iterrows():
        g = guidance.get(case_id(r))
        if not g:
            continue
        cf = case_features(r, feats)
        for lead in VERIF:
            la, lo = r.get(f"lat_{lead}"), r.get(f"lon_{lead}")
            has_truth = np.isfinite(la) and np.isfinite(lo)
            if with_truth and not has_truth:
                continue
            for row in member_rows(r, cf, g, calib, lead, members):
                row.update(case_id=case_id(r), lead=lead)
                if has_truth:
                    row["err"] = float(gc_distance_km(la, lo, row["lat"], row["lon"]))
                    row["tlat"] = float(la)
                    row["tlon"] = float(lo)
                recs.append(row)
    return pd.DataFrame(recs)


def enc(df, keys, members, med=None):
    X = df[keys].astype(float)
    if med is None:
        med = X.median()
    X = X.fillna(med).fillna(0.0)  # second fill catches all-NaN columns
    for m in members:
        X[f"is_{m}"] = (df["member"] == m).astype(float)
    return X.values, med


def fit_gbt_reg(X, y, depth=3, n=200, seed=0):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=n, max_depth=depth,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=seed).fit(X, y)


# ---------------------------------------------------------------------------
# weighted-consensus assembly from per-member weights
# ---------------------------------------------------------------------------
def assemble_from_weights(grp_lat, grp_lon, w):
    w = np.asarray(w, float)
    w = w / w.sum()
    ref = grp_lon[0]
    lat = float(np.dot(w, grp_lat))
    lon_rel = np.dot(w, ((np.asarray(grp_lon) - ref + 540) % 360 - 180))
    return round(lat, 2), round(float(wrap_lon(ref + lon_rel).item()), 2)


# ---------------------------------------------------------------------------
# static convex stack: per-lead weights minimizing calibration track error
# ---------------------------------------------------------------------------
def fit_static_convex(tr, members):
    """Returns {lead: {member: weight}} minimizing the mean great-circle error
    of the *weighted consensus position* on the calibration member-rows table,
    over the simplex (the calibration-optimal static per-lead weighting)."""
    from scipy.optimize import minimize
    out = {}
    for lead in VERIF:
        sub = tr[tr.lead == lead]
        lat_p = sub.pivot_table(index="case_id", columns="member", values="lat", aggfunc="first")
        lon_p = sub.pivot_table(index="case_id", columns="member", values="lon", aggfunc="first")
        tlat_p = sub.pivot_table(index="case_id", columns="member", values="tlat", aggfunc="first")
        tlon_p = sub.pivot_table(index="case_id", columns="member", values="tlon", aggfunc="first")
        M = [m for m in members if m in lat_p.columns]
        keep = lat_p[M].dropna().index
        if len(keep) < 30:
            out[lead] = {m: 1.0 / len(M) for m in M}
            continue
        LAT = lat_p.loc[keep, M].values          # (N, M)
        LON = lon_p.loc[keep, M].values
        TLAT = tlat_p.loc[keep, M[0]].values     # truth same across members
        TLON = tlon_p.loc[keep, M[0]].values
        ref = LON[:, 0]
        REL = (LON - ref[:, None] + 540) % 360 - 180

        def obj(w):
            w = np.clip(w, 0, None)
            w = w / (w.sum() + 1e-9)
            clat = LAT @ w
            clon = wrap_lon(ref + REL @ w)
            return float(np.mean(gc_distance_km(TLAT, TLON, clat, np.asarray(clon))))
        w0 = np.ones(len(M)) / len(M)
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
        bnds = [(0, 1)] * len(M)
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-9})
        w = np.clip(res.x, 0, None)
        w = w / w.sum()
        out[lead] = {m: float(wi) for m, wi in zip(M, w)}
    return out


def static_convex_track(cases, feats, guidance, calib, members, weights):
    """Assemble the static-convex-weighted consensus positions on `cases`."""
    out = {}
    for _, r in cases.iterrows():
        g = guidance.get(case_id(r))
        if not g:
            continue
        fc = {}
        for lead in VERIF:
            ms = {m: e for m, e in _members_at(g, lead).items() if m in members}
            if len(ms) < 2:
                continue
            w = weights.get(lead, {})
            names = [m for m in ms if m in w]
            if len(names) < 2:
                names = list(ms)
                ww = np.ones(len(names))
            else:
                ww = np.array([w[m] for m in names])
            lat, lon = assemble_from_weights(
                np.array([ms[m]["lat"] for m in names]),
                [ms[m]["lon"] for m in names], ww)
            fc[str(lead)] = dict(lat=lat, lon=lon)
        if fc:
            out[case_id(r)] = fc
    return out
