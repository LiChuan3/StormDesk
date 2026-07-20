"""Calibrate the office's RI probability on the calibration seasons (Platt
scaling on logit of the raw probability) and pick the CSI-optimal threshold.
Writes <work>/models/ri_calibration.json {a, b, threshold}.

p_cal = sigmoid(a * logit(clip(p_raw)) + b)
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


def _logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_platt(p_raw, y, iters=500, lr=0.1):
    x = _logit(p_raw)
    a, b = 1.0, 0.0
    n = len(y)
    for _ in range(iters):
        z = _sigmoid(a * x + b)
        ga = float(((z - y) * x).mean())
        gb = float((z - y).mean())
        a -= lr * ga
        b -= lr * gb
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="agent_full_qwen14b")
    ap.add_argument("--split", default="calib")
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    truth = {case_id(r): r for _, r in cases.iterrows()}

    ps, ys = [], []
    fpath = os.path.join(work_dir("forecasts"), f"{args.split}_{args.tag}.jsonl")
    seen = set()
    for line in open(fpath):
        d = json.loads(line)
        cid = d["case_id"]
        if cid in seen or not d.get("forecast"):
            continue
        seen.add(cid)
        p = d.get("ri24_prob")
        r = truth.get(cid)
        if p is None or r is None or not np.isfinite(r.get("vmax_24", np.nan)):
            continue
        ps.append(float(p))
        ys.append(1.0 if (r["vmax_24"] - r["vmax"]) >= 30 else 0.0)
    ps, ys = np.array(ps), np.array(ys)
    print(f"calib RI samples: n={len(ps)}, events={int(ys.sum())}, "
          f"base={ys.mean():.3f}, mean_p={ps.mean():.3f}")

    a, b = fit_platt(ps, ys)
    pc = _sigmoid(a * _logit(ps) + b)
    brier_raw = float(np.mean((ps - ys) ** 2))
    brier_cal = float(np.mean((pc - ys) ** 2))
    clim = float(np.mean((ys.mean() - ys) ** 2))
    print(f"a={a:.3f} b={b:.3f} | Brier raw {brier_raw:.4f} -> cal {brier_cal:.4f} "
          f"(clim {clim:.4f}, BSS {1-brier_cal/clim:.3f})")

    best = (0.0, 0.5)
    for thr in np.arange(0.02, 0.61, 0.01):
        pred = pc >= thr
        hits = float(np.sum(pred & (ys == 1)))
        fa = float(np.sum(pred & (ys == 0)))
        miss = float(np.sum(~pred & (ys == 1)))
        csi = hits / (hits + fa + miss) if (hits + fa + miss) else 0.0
        if csi > best[0]:
            best = (csi, float(thr))
    print(f"CSI-optimal threshold on calib: {best[1]:.2f} (CSI {best[0]:.3f})")

    out = dict(a=round(a, 4), b=round(b, 4), threshold=round(best[1], 2),
               n=len(ps), events=int(ys.sum()))
    path = os.path.join(work_dir("models"), "ri_calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
