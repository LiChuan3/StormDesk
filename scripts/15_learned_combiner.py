"""Learned adaptive combiner baseline (the reviewer's W1 reference).

Does a small learned model, given the same briefing information, reproduce the
office's case-adaptive combination? Two components, both fit strictly on the
calibration seasons (2018-2020):

Track gating   per-(case, member, lead) gradient-boosted regression of the
               member's log track error from case features + member-specific
               features (implied-motion deviation from observed motion,
               distance from the cluster mean, historical skill); test-time
               weights proportional to 1/exp(prediction)^2, combined with the
               same relative-longitude weighted mean the office uses.
Intensity      per-lead gradient-boosted regression of (truth - bias-corrected
               consensus prior) on the same briefing features the RI
               classifiers use; the prediction is an unbounded learned delta.

Writes <work>/forecasts/<split>_learned_gbt.jsonl.
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.baselines import Calibration, _members_at
from stormdesk.evaluate import case_id, load_forecasts
from stormdesk.geo import gc_distance_km, motion_uv_kmh, wrap_lon
from stormdesk.guidance.merge import load_guidance, load_features

VERIF = [24, 48, 72]
MEMBERS = ["pangu", "fengwu", "cons_aiwp", "gru", "transformer", "cliper"]

_spec = importlib.util.spec_from_file_location(
    "ri12", os.path.join(os.path.dirname(os.path.abspath(__file__)), "12_ri_baselines.py"))
_ri12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ri12)


def case_features(r, feats, guidance):
    """Case-level features shared by both learners (medians imputed later)."""
    cid = case_id(r)
    ft = feats.get(cid, {})
    diag = ft.get("diag") or {}
    h = r["history"]
    mu, mv = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
    dv12 = (r["vmax"] - h[-3]["vmax"]) if h[-3]["vmax"] is not None else 0.0
    g = guidance.get(cid) or {}
    ms = [e.get("24") or e.get(24) for e in g.values() if e]
    ms = [e for e in ms if e]
    spread = np.nan
    if len(ms) >= 2:
        lat_c = float(np.mean([e["lat"] for e in ms]))
        lon_c = float(np.mean([e["lon"] for e in ms]))
        spread = float(np.mean([gc_distance_km(e["lat"], e["lon"], lat_c, lon_c)
                                for e in ms]))
    return dict(v0=r["vmax"], dv12=dv12, abslat=abs(r["lat"]),
                mot_u=float(mu), mot_v=float(mv),
                shear=diag.get("shear_kt"), sst=diag.get("sst_c"),
                rh=diag.get("rh_mid_pct"), pot=diag.get("pot_kt"),
                div=diag.get("div200_1e7"), spread24=spread)


CASE_KEYS = ["v0", "dv12", "abslat", "mot_u", "mot_v", "shear", "sst", "rh",
             "pot", "div", "spread24"]
MEM_KEYS = ["hist_mae", "dist_cluster", "motion_dev_deg", "speed_ratio", "disp24"]


def member_rows(r, cf, g, calib, lead):
    """One row per member with a forecast at this lead."""
    ms = _members_at(g, lead)
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


def build_track_table(split, cases, feats, guidance, calib, with_truth=True):
    recs = []
    for _, r in cases.iterrows():
        cid = case_id(r)
        g = guidance.get(cid)
        if not g:
            continue
        cf = case_features(r, feats, guidance)
        for lead in VERIF:
            la, lo = r.get(f"lat_{lead}"), r.get(f"lon_{lead}")
            if with_truth and not (np.isfinite(la) and np.isfinite(lo)):
                continue
            for row in member_rows(r, cf, g, calib, lead):
                row.update(case_id=cid, lead=lead)
                if with_truth:
                    row["err"] = float(gc_distance_km(la, lo, row["lat"], row["lon"]))
                recs.append(row)
    return pd.DataFrame(recs)


def fit_gbt_reg(X, y, depth=3, n=200):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=n, max_depth=depth,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=0).fit(X, y)


def enc(df, keys, med=None):
    X = df[keys].astype(float)
    if med is None:
        med = X.median()
    X = X.fillna(med)
    # member one-hot
    for m in MEMBERS:
        X[f"is_{m}"] = (df["member"] == m).astype(float)
    return X.values, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-split", default="calib")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    calib = Calibration.load()

    def load_split(split):
        with open(os.path.join(work_dir("cases"), f"{split}.pkl"), "rb") as f:
            cases = pickle.load(f)
        return cases, load_features(split), load_guidance(split)

    print("building calib track table ...")
    c_cases, c_feats, c_guid = load_split(args.calib_split)
    tr = build_track_table(args.calib_split, c_cases, c_feats, c_guid, calib)
    print(f"calib member-rows: {len(tr)}")
    print("building test track table ...")
    t_cases, t_feats, t_guid = load_split(args.split)
    te = build_track_table(args.split, t_cases, t_feats, t_guid, calib, with_truth=False)

    keys = CASE_KEYS + MEM_KEYS
    models, meds = {}, {}
    for lead in VERIF:
        sub = tr[tr.lead == lead]
        X, med = enc(sub, keys)
        y = np.log(np.clip(sub["err"].values, 1.0, None))
        models[lead] = fit_gbt_reg(X, y)
        meds[lead] = med
        print(f"lead {lead}: fit on {len(sub)} rows")

    # intensity delta learner on the RI feature matrix (truth at all leads)
    print("building intensity matrices ...")
    dfc = _ri12.build_matrix(args.calib_split)
    dft = _ri12.build_matrix(args.split)
    bc_c = load_forecasts(os.path.join(work_dir("forecasts"),
                                       f"{args.calib_split}_cons_bc.jsonl"))
    bc_t = load_forecasts(os.path.join(work_dir("forecasts"),
                                       f"{args.split}_cons_bc.jsonl"))

    def truth_prior(cases, bc, cid_index):
        """per case: truth vmax and cons_bc prior at each lead"""
        out = {}
        for _, r in cases.iterrows():
            cid = case_id(r)
            if cid not in cid_index:
                continue
            d = bc.get(cid)
            fc = d.get("forecast") if d else None
            ent = {}
            for lead in VERIF:
                vt = r.get(f"vmax_{lead}")
                pv = None
                if fc and fc.get(str(lead)) and fc[str(lead)].get("vmax") is not None:
                    pv = float(fc[str(lead)]["vmax"])
                ent[lead] = (float(vt) if np.isfinite(vt) else None, pv)
            out[cid] = ent
        return out

    tp_c = truth_prior(c_cases, bc_c, set(dfc.case_id))
    tp_t = truth_prior(t_cases, bc_t, set(dft.case_id))
    Zc, stats = _ri12.impute_standardize(dfc)
    Zt, _ = _ri12.impute_standardize(dft, stats)
    int_models = {}
    for lead in VERIF:
        Xl, yl = [], []
        for i, cid in enumerate(dfc.case_id.values):
            vt, pv = tp_c.get(cid, {}).get(lead, (None, None))
            if vt is None or pv is None:
                continue
            Xl.append(Zc[i])
            yl.append(vt - pv)
        int_models[lead] = fit_gbt_reg(np.array(Xl), np.array(yl), depth=2, n=150)
        print(f"intensity lead {lead}: fit on {len(yl)} rows")

    # assemble test forecasts: unbounded gate, and a variant whose weights are
    # clipped to the office's contract envelope (skill-prior x trust in
    # [0.25, 4]) to separate contract expressiveness from judgement quality
    cidx = {c: i for i, c in enumerate(dft.case_id.values)}
    out_rows, out_rows_b = {}, {}
    for lead in VERIF:
        sub = te[te.lead == lead]
        if sub.empty:
            continue
        X, _ = enc(sub, keys, meds[lead])
        pred = models[lead].predict(X)
        sub = sub.assign(w=1.0 / np.exp(pred) ** 2)
        for cid, grp in sub.groupby("case_id"):
            w = grp["w"].values
            w = w / w.sum()
            prior = np.array([
                (calib.weight(m, lead) or 0.0) for m in grp["member"].values])
            if prior.sum() <= 1e-12:
                prior = np.ones(len(w))
            prior = prior / prior.sum()
            wb = prior * np.clip((w / w.sum()) / prior, 0.25, 4.0)
            wb = wb / wb.sum()
            vmax = None
            if cid in cidx:
                _, pv = tp_t.get(cid, {}).get(lead, (None, None))
                if pv is not None:
                    vmax = round(float(pv + int_models[lead].predict(Zt[cidx[cid]][None])[0]), 1)
            for weights, store in ((w, out_rows), (wb, out_rows_b)):
                ref = grp["lon"].values[0]
                lat = float((grp["lat"].values * weights).sum())
                lon_rel = (((grp["lon"].values - ref + 540) % 360 - 180) * weights).sum()
                lon = float(wrap_lon(ref + lon_rel).item())
                store.setdefault(cid, {})[str(lead)] = dict(
                    lat=round(lat, 2), lon=round(lon, 2), vmax=vmax)

    for name, rows_ in (("learned_gbt", out_rows), ("learned_gbt_bounded", out_rows_b)):
        path = os.path.join(work_dir("forecasts"), f"{args.split}_{name}.jsonl")
        with open(path, "w") as f:
            for cid, fc in rows_.items():
                f.write(json.dumps(dict(case_id=cid, method=name, forecast=fc)) + "\n")
        print(f"{name}: {len(rows_)} forecasts -> {path}")


if __name__ == "__main__":
    main()
