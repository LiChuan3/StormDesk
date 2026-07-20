"""Run-to-run sampling variance of the office (temperature 0.3, unseeded).

Compares repeated full-office runs on a common case subset: per-run mean
verification metrics, per-case forecast spread, and RI-probability spread.

Usage:
  python scripts/14_variance.py --split test \
      --runs agent_full_qwen14b,agent_full_qwen14b_r2,agent_full_qwen14b_r3
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.evaluate import case_id, evaluate_methods, load_forecasts, ri_scores
from stormdesk.geo import gc_distance_km


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", required=True)
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    names = args.runs.split(",")
    fdir = work_dir("forecasts")
    runs = {}
    for n in names:
        fc = load_forecasts(os.path.join(fdir, f"{args.split}_{n}.jsonl"))
        runs[n] = {k: v for k, v in fc.items() if v.get("forecast")}
    common = set.intersection(*(set(v) for v in runs.values()))
    print(f"common cases across {len(names)} runs: {len(common)}")
    sub = cases[[case_id(r) in common for _, r in cases.iterrows()]]

    tab = evaluate_methods(sub, runs)
    pd.set_option("display.width", 200)
    print(tab[["method", "lead", "n", "track_km", "vmax_mae_kt", "vmax_bias_kt"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nper-lead across-run range of mean metrics:")
    for lead in (24, 48, 72):
        t = tab[tab.lead == lead]
        print(f"  {lead}h track {t.track_km.min():.1f}-{t.track_km.max():.1f} km "
              f"(range {t.track_km.max()-t.track_km.min():.2f}); "
              f"vmax {t.vmax_mae_kt.min():.2f}-{t.vmax_mae_kt.max():.2f} kt "
              f"(range {t.vmax_mae_kt.max()-t.vmax_mae_kt.min():.2f})")

    # per-case spread across runs
    dpos, dv, dp = [], [], []
    for cid in common:
        e = [runs[n][cid]["forecast"].get("24") for n in names]
        if all(x for x in e):
            lat0, lon0 = e[0]["lat"], e[0]["lon"]
            dpos.append(max(gc_distance_km(lat0, lon0, x["lat"], x["lon"]) for x in e[1:]))
            vs = [x.get("vmax") for x in e if x.get("vmax") is not None]
            if len(vs) == len(names):
                dv.append(max(vs) - min(vs))
        ps = [runs[n][cid].get("ri24_prob") for n in names]
        ps = [p for p in ps if p is not None]
        if len(ps) == len(names):
            dp.append(max(ps) - min(ps))
    print(f"\nper-case 24h spread across runs: track median {np.median(dpos):.1f} km "
          f"(p90 {np.percentile(dpos, 90):.1f}); vmax median {np.median(dv):.1f} kt "
          f"(p90 {np.percentile(dv, 90):.1f}); ri_prob median {np.median(dp):.2f} "
          f"(p90 {np.percentile(dp, 90):.2f})")

    ri = ri_scores(sub, runs, use_prob=True)
    print("\nRI on common subset:")
    print(ri.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
