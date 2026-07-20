"""Stratified verification for the supplementary: by basin, by initial
intensity class, and by proximity to land. Homogeneous within each stratum.

Usage:
  python scripts/13_breakdowns.py --split test \
      --methods agent_full_qwen14b,hybrid_static,cons_aiwp,cons_bc,cliper
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.evaluate import evaluate_methods, load_forecasts, ri_scores


def load(split, names):
    fdir = work_dir("forecasts")
    out = {}
    for n in names:
        p = os.path.join(fdir, f"{split}_{n}.jsonl")
        if os.path.exists(p):
            fc = load_forecasts(p)
            out[n] = {k: v for k, v in fc.items() if v.get("forecast")}
    return out


def strata(cases, split):
    v = cases["vmax"]
    yield "basin", cases["basin"]
    yield "intensity", pd.cut(v, [0, 63, 95, np.inf],
                              labels=["TS(34-63kt)", "Cat1-2(64-95kt)", "Cat3+(>=96kt)"])
    d2l = cases.get("dist2land")
    if d2l is not None:
        yield "land", pd.Series(np.where(d2l < 300, "near-land(<300km)", "open-ocean"),
                                index=cases.index)
    # verified RI events: the cycles that matter most for intensity
    dv24 = cases["vmax_24"] - cases["vmax"]
    yield "ri", pd.Series(np.where(dv24 >= 30, "RI-event", "no-RI"), index=cases.index)
    # guidance conflict: cross-member 24-h track spread terciles
    from stormdesk.evaluate import case_id as _cid
    from stormdesk.geo import gc_distance_km
    from stormdesk.guidance.merge import load_guidance
    guidance = load_guidance(split)
    sp = []
    for _, r in cases.iterrows():
        g = guidance.get(_cid(r)) or {}
        ms = [e.get("24") or e.get(24) for e in g.values() if e]
        ms = [e for e in ms if e]
        if len(ms) >= 2:
            lat_c = float(np.mean([e["lat"] for e in ms]))
            lon_c = float(np.mean([e["lon"] for e in ms]))
            sp.append(float(np.mean([gc_distance_km(e["lat"], e["lon"], lat_c, lon_c)
                                     for e in ms])))
        else:
            sp.append(np.nan)
    sp = pd.Series(sp, index=cases.index)
    q1, q2 = sp.quantile([1 / 3, 2 / 3])
    lab = pd.Series(np.where(sp <= q1, "spread-low",
                             np.where(sp <= q2, "spread-mid", "spread-high")),
                    index=cases.index)
    lab[sp.isna()] = np.nan
    yield "spread24", lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods",
                    default="agent_full_qwen14b,hybrid_static,cons_aiwp,cons_bc,cliper")
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    methods = load(args.split, args.methods.split(","))
    print("methods:", {m: len(v) for m, v in methods.items()})

    all_rows = []
    for sname, series in strata(cases, args.split):
        for val in series.dropna().unique():
            sub = cases[series == val]
            if len(sub) < 30:
                continue
            t = evaluate_methods(sub, methods)
            t.insert(0, "stratum", f"{sname}={val}")
            all_rows.append(t)
            ri = ri_scores(sub, {m: v for m, v in methods.items()
                                 if m.startswith("agent")}, use_prob=True)
            ri.insert(0, "stratum", f"{sname}={val}")
            ri.to_csv(os.path.join(work_dir("results"),
                                   f"{args.split}_breakdown_ri.csv"),
                      mode="a", header=not os.path.exists(
                          os.path.join(work_dir("results"),
                                       f"{args.split}_breakdown_ri.csv")), index=False)
    out = pd.concat(all_rows, ignore_index=True)
    path = os.path.join(work_dir("results"), f"{args.split}_breakdowns.csv")
    out.to_csv(path, index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
