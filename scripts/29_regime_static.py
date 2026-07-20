"""Regime-conditioned static baselines and a linear gate.

Tests whether the learned (GBT) gate's track gain over the global static convex
stack could be explained by coarse regime-conditional reweighting rather than
genuine per-case adaptation. All fits are on the calibration seasons.

  static_basin   per-(basin, lead) convex simplex weights (global fallback when
                 a basin has < MIN_CASES calibration cases at a lead)
  static_icat    per-(initial-intensity category, lead) convex weights
                 (categories: <64 / 64-95 / >=96 kt)
  gate_ridge     linear ridge gate on the identical features as the GBT gate
                 (standardized case + member features + member one-hot),
                 predicting log track error; weights ~ 1/exp(pred)^2

Writes <split>_static_basin.jsonl, <split>_static_icat.jsonl,
<split>_gate_ridge.jsonl (track from the regime weights; vmax from the
bias-corrected consensus, as for the other track-ladder methods).
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk import combiner as CB
from stormdesk.baselines import Calibration, _members_at
from stormdesk.config import work_dir
from stormdesk.evaluate import case_id, load_forecasts
from stormdesk.guidance.merge import load_guidance, load_features

VERIF = [24, 48, 72]
MIN_CASES = 40


def load_split(split):
    with open(os.path.join(work_dir("cases"), f"{split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    return cases, load_features(split), load_guidance(split)


def write(split, method, rows):
    path = os.path.join(work_dir("forecasts"), f"{split}_{method}.jsonl")
    with open(path, "w") as f:
        for cid, fc in rows.items():
            f.write(json.dumps(dict(case_id=cid, method=method, forecast=fc)) + "\n")
    print(f"{method}: {len(rows)} -> {path}")


def icat(v):
    return "td_ts" if v < 64 else ("cat12" if v < 96 else "cat3plus")


def fit_group_weights(tr, cases, group_fn, members):
    """{group: {lead: {member: w}}} with a global fallback."""
    grp_of = {case_id(r): group_fn(r) for _, r in cases.iterrows()}
    tr = tr.assign(grp=tr["case_id"].map(grp_of))
    wts = {"__global__": CB.fit_static_convex(tr, members)}
    for g in sorted(tr["grp"].dropna().unique()):
        sub = tr[tr.grp == g]
        counts = {l: sub[sub.lead == l]["case_id"].nunique() for l in VERIF}
        if min(counts.values()) < MIN_CASES:
            print(f"  group {g}: too few cases {counts}, global fallback")
            continue
        wts[g] = CB.fit_static_convex(sub, members)
        print(f"  group {g}: cases {counts}")
    return wts, grp_of


def group_track(cases, guidance, members, wts, group_fn):
    out = {}
    for _, r in cases.iterrows():
        g = guidance.get(case_id(r))
        if not g:
            continue
        w_grp = wts.get(group_fn(r), wts["__global__"])
        fc = {}
        for lead in VERIF:
            ms = {m: e for m, e in _members_at(g, lead).items() if m in members}
            if len(ms) < 2:
                continue
            w = w_grp.get(lead, {})
            names = [m for m in ms if m in w]
            if len(names) < 2:
                names = list(ms)
                ww = np.ones(len(names))
            else:
                ww = np.array([w[m] for m in names])
            if ww.sum() <= 1e-9:
                ww = np.ones(len(names))
            lat, lon = CB.assemble_from_weights(
                np.array([ms[m]["lat"] for m in names]),
                [ms[m]["lon"] for m in names], ww)
            fc[str(lead)] = dict(lat=lat, lon=lon)
        if fc:
            out[case_id(r)] = fc
    return out


def with_vmax(track_rows, vmax_src):
    out = {}
    for cid, fc in track_rows.items():
        merged = {}
        for lead, e in fc.items():
            ve = (vmax_src.get(cid) or {}).get("forecast", {}).get(lead) \
                if vmax_src.get(cid) else None
            merged[lead] = dict(e, vmax=ve.get("vmax") if ve else None)
        out[cid] = merged
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--calib-split", default="calib")
    args = ap.parse_args()

    calib = Calibration.load()
    c_cases, c_feats, c_guid = load_split(args.calib_split)
    t_cases, t_feats, t_guid = load_split(args.split)
    bc_test = load_forecasts(os.path.join(work_dir("forecasts"),
                                          f"{args.split}_cons_bc.jsonl"))

    tr = CB.build_track_table(c_cases, c_feats, c_guid, calib, CB.PRIMARY)
    summary = {}

    print("fitting basin-conditioned static convex weights ...")
    wts_b, _ = fit_group_weights(tr, c_cases, lambda r: r["basin"], CB.PRIMARY)
    fb = group_track(t_cases, t_guid, CB.PRIMARY, wts_b, lambda r: r["basin"])
    write(args.split, "static_basin", with_vmax(fb, bc_test))
    summary["basin_groups_fit"] = sorted(k for k in wts_b if k != "__global__")

    print("fitting intensity-category-conditioned static convex weights ...")
    wts_i, _ = fit_group_weights(tr, c_cases, lambda r: icat(r["vmax"]), CB.PRIMARY)
    fi = group_track(t_cases, t_guid, CB.PRIMARY, wts_i, lambda r: icat(r["vmax"]))
    write(args.split, "static_icat", with_vmax(fi, bc_test))
    summary["icat_groups_fit"] = sorted(k for k in wts_i if k != "__global__")

    print("fitting linear ridge gate (same features as the GBT gate) ...")
    keys_full = CB.CASE_KEYS + CB.MEM_KEYS
    models, meds, stats = {}, {}, {}
    for lead in VERIF:
        sub = tr[tr.lead == lead]
        y = np.log(np.clip(sub["err"].values, 1.0, None))
        X, med = CB.enc(sub, keys_full, CB.PRIMARY)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        # closed-form ridge (avoids sklearn dependency drift)
        lam = 10.0
        A = Z.T @ Z + lam * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (y - y.mean()))
        models[lead] = (w, float(y.mean()))
        meds[lead] = med
        stats[lead] = (mu, sd)

    te = CB.build_track_table(t_cases, t_feats, t_guid, calib, CB.PRIMARY,
                              with_truth=False)
    out = {}
    for lead in VERIF:
        sub = te[te.lead == lead]
        if sub.empty:
            continue
        X, _ = CB.enc(sub, keys_full, CB.PRIMARY, meds[lead])
        mu, sd = stats[lead]
        w, ybar = models[lead]
        pred = ((X - mu) / sd) @ w + ybar
        sub = sub.assign(wgt=1.0 / np.exp(pred) ** 2)
        for cid, grp in sub.groupby("case_id"):
            lat, lon = CB.assemble_from_weights(
                grp["lat"].values, grp["lon"].values.tolist(), grp["wgt"].values)
            out.setdefault(cid, {})[str(lead)] = dict(lat=lat, lon=lon)
    write(args.split, "gate_ridge", with_vmax(out, bc_test))

    with open(os.path.join(work_dir("results"),
                           f"{args.split}_regime_static.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
