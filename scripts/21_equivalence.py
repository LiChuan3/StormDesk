"""Formal equivalence testing (TOST) and paired-difference CIs for the
point-forecast comparisons the paper leans on (review P0-2).

For each (target, reference) pair, on the homogeneous sample, we storm-cluster
bootstrap the paired mean-difference (target - reference) in track error (km)
and Vmax MAE (kt), report the 90% and 95% CIs, and give a TOST verdict against
pre-registered equivalence margins (track: +-10/15/20 km per lead; intensity:
+-1 kt). Equivalence within margin m holds iff the 90% CI lies within [-m, +m].
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.evaluate import case_id, load_forecasts, _truth, _fc
from stormdesk.geo import gc_distance_km

VERIF = [24, 48, 72]
TRACK_MARGINS = [10.0, 15.0, 20.0]
VMAX_MARGIN = 1.0


def paired_arrays(cases, A, B, lead, allowed=None):
    sids, dt, dv = [], [], []
    for _, r in cases.iterrows():
        la, lo, v = _truth(r, lead)
        if not (np.isfinite(la) and np.isfinite(lo)):
            continue
        cid = case_id(r)
        if allowed is not None and cid not in allowed:
            continue
        ea, eb = _fc(A, cid, lead), _fc(B, cid, lead)
        if ea is None or eb is None:
            continue
        sids.append(r["sid"])
        dt.append(gc_distance_km(la, lo, ea["lat"], ea["lon"])
                  - gc_distance_km(la, lo, eb["lat"], eb["lon"]))
        if (ea.get("vmax") is not None and eb.get("vmax") is not None and np.isfinite(v)):
            dv.append(abs(ea["vmax"] - v) - abs(eb["vmax"] - v))
        else:
            dv.append(np.nan)
    return np.array(sids), np.array(dt), np.array(dv)


def boot_ci(sids, d, n_boot, rng):
    d = d[np.isfinite(d)]
    sids = sids[:len(d)] if len(sids) != len(d) else sids
    uniq = np.array(sorted(set(sids)))
    idx = {s: np.where(sids == s)[0] for s in uniq}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[s] for s in pick])
        means.append(float(np.mean(d[ii])))
    means = np.array(means)
    return (float(np.mean(d)),
            float(np.percentile(means, 5)), float(np.percentile(means, 95)),
            float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)),
            float(2 * min((means > 0).mean(), (means < 0).mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--pairs", default=(
        "agent_full_qwen14b:hybrid_static,"
        "agent_full_qwen14b_anon:agent_full_qwen14b,"
        "learned_gbt2:hybrid_static,learned_gbt_contract:hybrid_static,"
        "agent_mini_qwen14b:hybrid_static"))
    ap.add_argument("--case-list", default=None,
                    help="analysis-manifest JSON restricting the sample per lead")
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    allowed = None
    if args.case_list:
        with open(args.case_list) as f:
            mf = json.load(f)
        allowed = {int(l): set(e["case_ids"]) for l, e in mf["leads"].items()}
    fdir = work_dir("forecasts")
    rng = np.random.default_rng(0)
    out = []
    for pair in args.pairs.split(","):
        tgt, ref = pair.split(":")
        A = load_forecasts(os.path.join(fdir, f"{args.split}_{tgt}.jsonl"))
        B = load_forecasts(os.path.join(fdir, f"{args.split}_{ref}.jsonl"))
        for lead, tmar in zip(VERIF, TRACK_MARGINS):
            sids, dt, dv = paired_arrays(cases, A, B, lead,
                                         allowed.get(lead) if allowed else None)
            if len(dt) < 30:
                continue
            tm, tlo, thi, tlo95, thi95, tp = boot_ci(sids, dt, args.n_boot, rng)
            vm, vlo, vhi, vlo95, vhi95, vp = boot_ci(sids, dv, args.n_boot, rng)
            teq = bool(-tmar < tlo and thi < tmar)
            veq = bool(-VMAX_MARGIN < vlo and vhi < VMAX_MARGIN)
            out.append(dict(target=tgt, ref=ref, lead=lead, n=len(dt),
                            track_diff=round(tm, 2), track_ci90=[round(tlo, 2), round(thi, 2)],
                            track_p=round(tp, 4), track_margin=tmar, track_equiv=teq,
                            vmax_diff=round(vm, 3), vmax_ci90=[round(vlo, 3), round(vhi, 3)],
                            vmax_p=round(vp, 4), vmax_equiv=veq))
    import pandas as pd
    df = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    with open(os.path.join(work_dir("results"), f"{args.split}_equivalence.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote equivalence.json")


if __name__ == "__main__":
    main()
