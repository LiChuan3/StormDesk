"""Decompose the learned combiner's gain into static reweighting vs. genuine
case adaptation, and produce a fully contract-constrained intensity combiner.

Track ladder (intensity held at bias-corrected consensus so track is isolated):
  static_convex   calibration-optimal per-lead convex weights (no case info)
  gbt_static      GBT gate on member id + historical MAE + lead only
  gbt_case        full GBT gate (case + member-specific features)
  gbt_shuffled    full features, case features shuffled across cases
  gbt_case_withaiwp  full gate but including the derived AIWP-consensus member

Intensity (P0-3):
  learned_gbt2            GBT residual on bias-corrected prior (unbounded)
  learned_gbt_contract    same residual clipped to +-25 kt with the office
                          rate/MPI/hard caps and an independently fit shrinkage
Reports the static->case increment and cap-crossing statistics.

Writes forecast files and results/decomp_summary.json.
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

from stormdesk import combiner as CB
from stormdesk.agents.auditor import apply_hard_caps
from stormdesk.baselines import Calibration
from stormdesk.config import work_dir
from stormdesk.evaluate import case_id, load_forecasts, evaluate_methods
from stormdesk.geo import gc_distance_km
from stormdesk.guidance.merge import load_guidance, load_features

_spec = importlib.util.spec_from_file_location(
    "ri12", os.path.join(os.path.dirname(os.path.abspath(__file__)), "12_ri_baselines.py"))
_ri12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ri12)

VERIF = [24, 48, 72]


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


def gate_forecast(cases, feats, guidance, calib, members, model, meds, keys,
                  shuffle_case=False, seed=0, bounded=False):
    """Assemble track from a per-member-error GBT gate (weights ~ 1/err^2).
    With bounded=True, the gate is expressed inside the office contract: the
    ratio of the gate weight to the skill-prior weight is clipped to [0.25,4]."""
    te = CB.build_track_table(cases, feats, guidance, calib, members, with_truth=False)
    if shuffle_case:
        rng = np.random.default_rng(seed)
        for k in CB.CASE_KEYS:
            te[k] = rng.permutation(te[k].values)
    out = {}
    for lead in VERIF:
        sub = te[te.lead == lead]
        if sub.empty:
            continue
        X, _ = CB.enc(sub, keys, members, meds[lead])
        pred = model[lead].predict(X)
        sub = sub.assign(w=1.0 / np.exp(pred) ** 2)
        for cid, grp in sub.groupby("case_id"):
            w = grp["w"].values
            if bounded:
                prior = np.array([(calib.weight(m, lead) or 0.0)
                                  for m in grp["member"].values])
                if prior.sum() <= 1e-12:
                    prior = np.ones(len(w))
                prior = prior / prior.sum()
                w = w / w.sum()
                w = prior * np.clip(w / np.clip(prior, 1e-9, None), 0.25, 4.0)
            lat, lon = CB.assemble_from_weights(
                grp["lat"].values, grp["lon"].values.tolist(), w)
            out.setdefault(cid, {})[str(lead)] = dict(lat=lat, lon=lon)
    return out


def with_vmax(track_rows, vmax_src):
    """Attach the bias-corrected-consensus vmax to a track-only forecast set."""
    out = {}
    for cid, fc in track_rows.items():
        v = vmax_src.get(cid, {}).get("forecast", {}) if isinstance(
            vmax_src.get(cid), dict) else {}
        merged = {}
        for lead, e in fc.items():
            vm = None
            ve = (vmax_src.get(cid) or {}).get("forecast", {}).get(lead) if vmax_src.get(cid) else None
            if ve:
                vm = ve.get("vmax")
            merged[lead] = dict(e, vmax=vm)
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
    bc_test = load_forecasts(os.path.join(work_dir("forecasts"), f"{args.split}_cons_bc.jsonl"))

    summary = {}

    # ---- static convex (over primary members; no derived consensus) ----
    print("fitting static convex weights ...")
    tr_primary = CB.build_track_table(c_cases, c_feats, c_guid, calib, CB.PRIMARY)
    wts = CB.fit_static_convex(tr_primary, CB.PRIMARY)
    summary["static_convex_weights"] = {str(l): wts[l] for l in VERIF}
    print("  weights:", {l: {m: round(v, 2) for m, v in wts[l].items()} for l in VERIF})
    static_track = CB.static_convex_track(t_cases, t_feats, t_guid, calib, CB.PRIMARY, wts)
    write(args.split, "static_convex", with_vmax(static_track, bc_test))

    # ---- GBT gates: static / case / shuffled (primary members) ----
    print("fitting GBT track gates ...")
    tr = CB.build_track_table(c_cases, c_feats, c_guid, calib, CB.PRIMARY)
    keys_full = CB.CASE_KEYS + CB.MEM_KEYS
    keys_static = ["hist_mae"]  # + member one-hot + lead handled via per-lead fit
    models_full, models_static, meds = {}, {}, {}
    for lead in VERIF:
        sub = tr[tr.lead == lead]
        y = np.log(np.clip(sub["err"].values, 1.0, None))
        Xf, med = CB.enc(sub, keys_full, CB.PRIMARY)
        models_full[lead] = CB.fit_gbt_reg(Xf, y)
        meds[lead] = med
        Xs, _ = CB.enc(sub, keys_static, CB.PRIMARY, med)
        models_static[lead] = CB.fit_gbt_reg(Xs, y, depth=2, n=120)
    gbt_case = gate_forecast(t_cases, t_feats, t_guid, calib, CB.PRIMARY,
                             models_full, meds, keys_full)
    gbt_static = gate_forecast(t_cases, t_feats, t_guid, calib, CB.PRIMARY,
                               models_static, meds, keys_static)
    gbt_shuf = gate_forecast(t_cases, t_feats, t_guid, calib, CB.PRIMARY,
                             models_full, meds, keys_full, shuffle_case=True, seed=1)
    gbt_case_bnd = gate_forecast(t_cases, t_feats, t_guid, calib, CB.PRIMARY,
                                 models_full, meds, keys_full, bounded=True)
    write(args.split, "gbt_case", with_vmax(gbt_case, bc_test))
    write(args.split, "gbt_static", with_vmax(gbt_static, bc_test))
    write(args.split, "gbt_shuffled", with_vmax(gbt_shuf, bc_test))

    # ---- GBT case including derived AIWP-consensus member ----
    tr_all = CB.build_track_table(c_cases, c_feats, c_guid, calib, CB.MEMBERS_ALL)
    models_all, meds_all = {}, {}
    for lead in VERIF:
        sub = tr_all[tr_all.lead == lead]
        y = np.log(np.clip(sub["err"].values, 1.0, None))
        Xf, med = CB.enc(sub, keys_full, CB.MEMBERS_ALL)
        models_all[lead] = CB.fit_gbt_reg(Xf, y)
        meds_all[lead] = med
    gbt_withaiwp = gate_forecast(t_cases, t_feats, t_guid, calib, CB.MEMBERS_ALL,
                                 models_all, meds_all, keys_full)
    write(args.split, "gbt_case_withaiwp", with_vmax(gbt_withaiwp, bc_test))

    # ---- track-error summary (homogeneous within this method set) ----
    methods = {m: load_forecasts(os.path.join(work_dir("forecasts"), f"{args.split}_{m}.jsonl"))
               for m in ["static_convex", "gbt_static", "gbt_case", "gbt_shuffled",
                         "gbt_case_withaiwp", "hybrid_static"]}
    tab = evaluate_methods(t_cases, methods)
    trk = {m: {int(l): float(tab[(tab.method == m) & (tab.lead == l)]["track_km"].iloc[0])
               for l in VERIF} for m in methods}
    summary["track_km"] = trk
    summary["static_to_case_increment_km"] = {
        str(l): round(trk["static_convex"][l] - trk["gbt_case"][l], 2) for l in VERIF}
    summary["gbt_static_to_case_increment_km"] = {
        str(l): round(trk["gbt_static"][l] - trk["gbt_case"][l], 2) for l in VERIF}
    print("\nTRACK km:", json.dumps(trk, indent=1))
    print("static->case:", summary["static_to_case_increment_km"])
    print("gbt_static->case:", summary["gbt_static_to_case_increment_km"])

    # ---- intensity residual GBT: unbounded vs contract-constrained ----
    print("\nfitting intensity residual GBT ...")
    dfc = _ri12.build_matrix(args.calib_split)
    dft = _ri12.build_matrix(args.split)
    Zc, stats = _ri12.impute_standardize(dfc)
    Zt, _ = _ri12.impute_standardize(dft, stats)
    bc_c = load_forecasts(os.path.join(work_dir("forecasts"), f"{args.calib_split}_cons_bc.jsonl"))

    def truth_prior(cases, bc, idxset):
        out = {}
        for _, r in cases.iterrows():
            cid = case_id(r)
            if cid not in idxset:
                continue
            d = bc.get(cid)
            fc = d.get("forecast") if d else None
            ent = {}
            for lead in VERIF:
                vt = r.get(f"vmax_{lead}")
                pv = None
                if fc and fc.get(str(lead)) and fc[str(lead)].get("vmax") is not None:
                    pv = float(fc[str(lead)]["vmax"])
                ent[lead] = (float(vt) if np.isfinite(vt) else None, pv, r["vmax"])
            out[cid] = ent
        return out

    tp_c = truth_prior(c_cases, bc_c, set(dfc.case_id))
    tp_t = truth_prior(t_cases, bc_test, set(dft.case_id))
    int_models, shrink = {}, {}
    for lead in VERIF:
        Xl, yl = [], []
        for i, cid in enumerate(dfc.case_id.values):
            vt, pv, v0 = tp_c.get(cid, {}).get(lead, (None, None, None))
            if vt is None or pv is None:
                continue
            Xl.append(Zc[i]); yl.append(vt - pv)
        int_models[lead] = CB.fit_gbt_reg(np.array(Xl), np.array(yl), depth=2, n=150)
        # independent shrinkage for the *clipped* GBT residual, same protocol
        # as the office: fit a,b on calib to minimize |truth-(prior+a*clip+b)|
        preds, priors, truths = [], [], []
        for i, cid in enumerate(dfc.case_id.values):
            vt, pv, v0 = tp_c.get(cid, {}).get(lead, (None, None, None))
            if vt is None or pv is None:
                continue
            d = float(np.clip(int_models[lead].predict(Zc[i][None])[0], -25, 25))
            preds.append(d); priors.append(pv); truths.append(vt)
        preds, priors, truths = map(np.array, (preds, priors, truths))
        A = np.vstack([preds, np.ones_like(preds)]).T
        ab, *_ = np.linalg.lstsq(A, truths - priors, rcond=None)
        shrink[lead] = (float(ab[0]), float(ab[1]))
    summary["gbt_intensity_shrinkage"] = {str(l): shrink[l] for l in VERIF}
    print("  GBT intensity shrinkage (a,b):", summary["gbt_intensity_shrinkage"])

    cidx = {c: i for i, c in enumerate(dft.case_id.values)}
    unbounded, contract = {}, {}
    n_cross = {l: 0 for l in VERIF}; n_tot = {l: 0 for l in VERIF}
    for _, r in t_cases.iterrows():
        cid = case_id(r)
        if cid not in cidx:
            continue
        tt = tp_t.get(cid, {})
        gt = gbt_case.get(cid) or static_track.get(cid) or {}
        fc_u, fc_c = {}, {}
        for lead in VERIF:
            vt, pv, v0 = tt.get(lead, (None, None, None))
            trk_e = (bc_test.get(cid) or {}).get("forecast", {}).get(str(lead))
            if pv is None:
                continue
            raw = float(int_models[lead].predict(Zt[cidx[cid]][None])[0])
            n_tot[lead] += 1
            if abs(raw) > 25:
                n_cross[lead] += 1
            clipped = float(np.clip(raw, -25, 25))
            a, b = shrink[lead]
            v_contract = pv + a * clipped + b
            # unbounded uses the unbounded track gate; contract uses the
            # bounded track gate -> "GBT fully within office contract"
            eu = gt.get(str(lead)) or (bc_test.get(cid) or {}).get("forecast", {}).get(str(lead))
            eb = (gbt_case_bnd.get(cid) or {}).get(str(lead)) or eu
            if not eu:
                continue
            fc_u[str(lead)] = dict(lat=eu["lat"], lon=eu["lon"], vmax=round(pv + raw, 1))
            fc_c[str(lead)] = dict(lat=eb["lat"], lon=eb["lon"], vmax=round(v_contract, 1))
        if fc_c:
            init = dict(lat=r["lat"], lon=r["lon"], vmax=r["vmax"])
            diag = (t_feats.get(cid, {}) or {}).get("diag") or {}
            contract[cid] = apply_hard_caps(fc_c, init, diag)
            unbounded[cid] = fc_u
    write(args.split, "learned_gbt2", unbounded)
    write(args.split, "learned_gbt_contract", contract)
    summary["intensity_cap_crossing_frac"] = {
        str(l): round(n_cross[l] / max(n_tot[l], 1), 3) for l in VERIF}
    print("  fraction of GBT intensity residuals exceeding +-25 kt:",
          summary["intensity_cap_crossing_frac"])

    with open(os.path.join(work_dir("results"), f"{args.split}_decomp_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\nwrote decomp summary")


if __name__ == "__main__":
    main()
