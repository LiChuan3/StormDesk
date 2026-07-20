"""Data-driven / conformal justification of the contract envelope (R2/Q2).

The hand-set bounds are trust factor tau in [0.25,4] (a multiplicative weight
correction) and intensity delta in +-25 kt. We show they are not arbitrary:
each is a conformal-style coverage interval of the corresponding *ideal*
correction on the calibration season, and we sweep the width to expose the
safety-vs-headroom trade.

Trust: the ideal per-member weight is the calibration-optimal gate weight; the
ideal trust factor is (gate weight / skill-prior weight). We report the
fraction of these ratios already inside [0.25,4] and the 90/95% conformal
quantiles.
Delta: the ideal correction is (truth - bias-corrected prior); we report the
fraction inside +-25 kt and the conformal quantiles, per lead.
Width sweep: for a family of symmetric (log-)bounds we report coverage of the
ideal correction (headroom reachable) and the worst-case forced deviation
(safety).
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
from stormdesk.baselines import Calibration
from stormdesk.config import work_dir
from stormdesk.evaluate import case_id, load_forecasts
from stormdesk.guidance.merge import load_guidance, load_features
from stormdesk.agents.office import prior_weights

VERIF = [24, 48, 72]

_spec = importlib.util.spec_from_file_location(
    "ri12", os.path.join(os.path.dirname(os.path.abspath(__file__)), "12_ri_baselines.py"))
_ri12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ri12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-split", default="calib")
    args = ap.parse_args()
    calib = Calibration.load()
    with open(os.path.join(work_dir("cases"), f"{args.calib_split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    feats = load_features(args.calib_split)
    guid = load_guidance(args.calib_split)

    # ---- trust: ideal factor = gate weight / prior weight ----
    tr = CB.build_track_table(cases, feats, guid, calib, CB.PRIMARY)
    keys = CB.CASE_KEYS + CB.MEM_KEYS
    ratios = {l: [] for l in VERIF}
    for lead in VERIF:
        sub = tr[tr.lead == lead]
        X, med = CB.enc(sub, keys, CB.PRIMARY)
        y = np.log(np.clip(sub["err"].values, 1.0, None))
        model = CB.fit_gbt_reg(X, y)
        # leave-in gate weights per case, ratio to prior weight
        sub = sub.assign(gw=1.0 / np.exp(model.predict(X)) ** 2)
        for cid, grp in sub.groupby("case_id"):
            gw = grp["gw"].values
            gw = gw / gw.sum()
            g = guid.get(cid)
            if not g:
                continue
            pw = prior_weights(g, calib, lead)
            for m, wi in zip(grp["member"].values, gw):
                p = pw.get(m)
                if p and p > 1e-6:
                    ratios[lead].append(wi / p)
    print("=== trust factor: ideal = gate/prior weight ===")
    for lead in VERIF:
        a = np.array(ratios[lead])
        cov = float(np.mean((a >= 0.25) & (a <= 4.0)))
        q = np.percentile(a, [2.5, 5, 95, 97.5])
        print(f"  {lead}h: n={len(a)} coverage of [0.25,4]={cov:.1%}; "
              f"conformal 90%=[{q[1]:.2f},{q[2]:.2f}] 95%=[{q[0]:.2f},{q[3]:.2f}]")

    # ---- delta: ideal = truth - bias-corrected prior ----
    bc = load_forecasts(os.path.join(work_dir("forecasts"), f"{args.calib_split}_cons_bc.jsonl"))
    resid = {l: [] for l in VERIF}
    for _, r in cases.iterrows():
        cid = case_id(r)
        d = bc.get(cid)
        fc = d.get("forecast") if d else None
        if not fc:
            continue
        for lead in VERIF:
            vt = r.get(f"vmax_{lead}")
            e = fc.get(str(lead))
            if e and e.get("vmax") is not None and np.isfinite(vt):
                resid[lead].append(float(vt) - float(e["vmax"]))
    print("\n=== intensity delta: ideal = truth - bias-corrected prior ===")
    for lead in VERIF:
        a = np.array(resid[lead])
        cov = float(np.mean(np.abs(a) <= 25))
        q = np.percentile(a, [2.5, 5, 95, 97.5])
        print(f"  {lead}h: n={len(a)} coverage of +-25kt={cov:.1%}; "
              f"conformal 90%=[{q[1]:.1f},{q[2]:.1f}] 95%=[{q[0]:.1f},{q[3]:.1f}]")

    # ---- width sweep: headroom (coverage) vs safety (worst forced move) ----
    print("\n=== delta width sweep (24h): coverage vs worst-case forced move ===")
    a = np.array(resid[24])
    for b in (10, 15, 20, 25, 30, 40):
        cov = float(np.mean(np.abs(a) <= b))
        print(f"  +-{b} kt: reaches {cov:.1%} of ideal corrections; worst forced move {b} kt")
    out = dict(trust_ratio_coverage={str(l): float(np.mean((np.array(ratios[l]) >= 0.25) &
                                                           (np.array(ratios[l]) <= 4)))
                                     for l in VERIF},
               delta_coverage={str(l): float(np.mean(np.abs(np.array(resid[l])) <= 25))
                               for l in VERIF})
    with open(os.path.join(work_dir("results"), "bounds_analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote bounds_analysis.json")


if __name__ == "__main__":
    main()
