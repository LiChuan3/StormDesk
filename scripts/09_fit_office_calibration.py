"""Fit the office post-calibration (per-lead affine shrinkage of the intensity
delta) from an agent run on the CALIBRATION seasons, and write
<work>/models/office_calibration.json.

v_final = prior + a*delta + b, (a,b) by OLS of (truth - prior) on delta.
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
from stormdesk.evaluate import case_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="agent_full_qwen14b")
    ap.add_argument("--split", default="calib")
    ap.add_argument("--out", default=None,
                    help="output path (default models/office_calibration.json)")
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    truth = {case_id(r): r for _, r in cases.iterrows()}

    rows = []
    tr_path = os.path.join(work_dir("transcripts"), f"{args.split}_{args.tag}.jsonl")
    for line in open(tr_path):
        d = json.loads(line)
        t = d.get("transcript") or {}
        pv = t.get("prior_vmax") or {}
        it = (t.get("intensity") or t.get("single")) or {}
        dk = it.get("delta_kt") or {}
        r = truth.get(d["case_id"])
        if r is None:
            continue
        for l in ("24", "48", "72"):
            p, dd, vt = pv.get(l), dk.get(l), r.get(f"vmax_{l}")
            if p is None or dd is None or not np.isfinite(vt):
                continue
            try:
                dd = float(np.clip(float(dd), -25, 25))
            except (TypeError, ValueError):
                continue
            rows.append(dict(lead=l, prior=p, delta=dd, ideal=vt - p))
    df = pd.DataFrame(rows)
    coeffs = {}
    for l in ("24", "48", "72"):
        s = df[df.lead == l]
        if len(s) < 100:
            print(f"lead {l}: too few samples ({len(s)}), identity")
            continue
        A = np.vstack([s.delta.values, np.ones(len(s))]).T
        (a, b), *_ = np.linalg.lstsq(A, s.ideal.values, rcond=None)
        a = float(np.clip(a, 0.0, 1.0))
        b = float(np.clip(b, -5.0, 5.0))
        mae0 = s.ideal.abs().mean()
        mae1 = (s.ideal - s.delta).abs().mean()
        mae2 = (s.ideal - (a * s.delta + b)).abs().mean()
        coeffs[l] = dict(a=round(a, 3), b=round(b, 2), n=len(s))
        print(f"lead {l}: n={len(s)} a={a:.3f} b={b:+.2f} | MAE prior {mae0:.2f} "
              f"raw-delta {mae1:.2f} calibrated {mae2:.2f}")
    path = args.out or os.path.join(work_dir("models"), "office_calibration.json")
    with open(path, "w") as f:
        json.dump(coeffs, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
