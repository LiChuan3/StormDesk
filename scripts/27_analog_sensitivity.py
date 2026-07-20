"""Analog library sensitivity (R5/Q5): effect of k and the feature/metric on
the analog-median RI signal (rate, and the RI FAR/CSI that a rate-threshold
rule yields) and on the analog-median intensity forecast.

Sweeps k in {6,12,20,30} and ablates the feature weighting (uniform vs the
tuned weights) and the environment-feature group, reporting on the test split:
analog RI-rate AUC/PR-AUC for RI, and analog-median intensity MAE at 24 h.
"""
import argparse
import importlib.util
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.analogs import AnalogLibrary, FEATURES, entry_features, summarize_analogs
from stormdesk.config import work_dir
from stormdesk.evaluate import case_id
from stormdesk.geo import motion_uv_kmh
from stormdesk.guidance.merge import load_features

VERIF = [24, 48, 72]


def eval_config(cases, feats, lib, k):
    """Return per-case (analog RI rate, analog median dv24, truth RI, truth v24)."""
    recs = []
    for _, r in cases.iterrows():
        cid = case_id(r)
        diag = (feats.get(cid, {}) or {}).get("diag") or {}
        vt = r.get("vmax_24")
        if not diag or not np.isfinite(vt):
            continue
        h = r["history"]
        mu, mv = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
        dv12 = (r["vmax"] - h[-3]["vmax"]) if h[-3]["vmax"] is not None else 0.0
        f = entry_features(r["lat"], r["lon"], r["vmax"], dv12, float(mu), float(mv), diag, r["init"])
        if f is None:
            continue
        s = summarize_analogs(lib.query(f, r["lat"], r["sid"], k=k), r["vmax"])
        recs.append((s.get("ri24_rate", 0.0), s.get("dv24_median"), int((vt - r["vmax"]) >= 30),
                     float(vt), float(r["vmax"])))
    return recs


def metrics(recs):
    from sklearn.metrics import roc_auc_score, average_precision_score
    ri_rate = np.array([x[0] for x in recs])
    y = np.array([x[2] for x in recs])
    med = np.array([x[1] if x[1] is not None else 0.0 for x in recs])
    v0 = np.array([x[4] for x in recs]); vt = np.array([x[3] for x in recs])
    mae = float(np.mean(np.abs((v0 + med) - vt)))
    try:
        auc = float(roc_auc_score(y, ri_rate)); pr = float(average_precision_score(y, ri_rate))
    except Exception:
        auc = pr = float("nan")
    # CSI of a rate>=0.3 rule
    pred = ri_rate >= 0.3
    h = int(np.sum(pred & (y == 1))); m = int(np.sum(~pred & (y == 1))); fa = int(np.sum(pred & (y == 0)))
    csi = h / (h + m + fa) if (h + m + fa) else 0.0
    far = fa / (h + fa) if (h + fa) else float("nan")
    return dict(n=len(recs), ri_auc=round(auc, 3), ri_prauc=round(pr, 3),
                rule_csi=round(csi, 3), rule_far=round(far, 3), vmax24_mae=round(mae, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    feats = load_features(args.split)
    base = AnalogLibrary.load()

    print("=== k sweep (tuned weights) ===")
    for k in (6, 12, 20, 30):
        print(f"k={k:2d}: {metrics(eval_config(cases, feats, base, k))}")

    # uniform feature weights
    uni = AnalogLibrary(base.records, base.stats)
    uni.w = np.ones_like(base.w)
    print("\n=== metric ablation (k=12) ===")
    print(f"tuned weights:   {metrics(eval_config(cases, feats, base, 12))}")
    print(f"uniform weights: {metrics(eval_config(cases, feats, uni, 12))}")

    # drop environment features (shear/sst/rh/steer) -> zero their weights
    env = AnalogLibrary(base.records, base.stats)
    names = list(FEATURES)
    envmask = np.array([0.0 if n in ("shear_kt", "sst_c", "rh_mid_pct", "steer_u", "steer_v")
                        else base.w[i] for i, n in enumerate(names)])
    env.w = envmask
    print(f"no-environment:  {metrics(eval_config(cases, feats, env, 12))}")


if __name__ == "__main__":
    main()
