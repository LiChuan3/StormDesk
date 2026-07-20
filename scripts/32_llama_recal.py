"""Independent second-family calibration analysis (Llama-3.1-8B).

Inputs: a calibration-season Llama office run (for its own anchors) and a test
run executed with those anchors (STORMDESK_OFFICE_CALIB pointing at the
Llama-fit shrinkage). This script:
  1. fits the Llama office's own RI Platt scaling + CSI-optimal threshold on
     the calibration run (storm-grouped CV not needed: the scaler has 2 params);
  2. scores the own-calibrated test run: point metrics vs the hybrid on the
     pairwise-homogeneous subset with paired tests, and RI at the own frozen
     threshold;
  3. prints the same numbers for the transferred-anchor run for comparison.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.evaluate import (case_id, load_forecasts, paired_test,
                                evaluate_methods)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt(p, y):
    """2-parameter logistic recalibration by Newton steps on logit(p)."""
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    x = np.log(p / (1 - p))
    a, b = 1.0, 0.0
    for _ in range(200):
        z = sigmoid(a * x + b)
        g = np.array([np.sum((z - y) * x), np.sum(z - y)])
        w = z * (1 - z)
        H = np.array([[np.sum(w * x * x) + 1e-6, np.sum(w * x)],
                      [np.sum(w * x), np.sum(w) + 1e-6]])
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def apply_platt(p, a, b):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return sigmoid(a * np.log(p / (1 - p)) + b)


def csi_threshold(p, y):
    best_t, best = 0.5, -1
    for t in np.arange(0.02, 0.9, 0.01):
        pred = p >= t
        h = np.sum(pred & (y == 1)); m = np.sum(~pred & (y == 1))
        fa = np.sum(pred & (y == 0))
        csi = h / max(h + m + fa, 1)
        if csi > best:
            best, best_t = csi, float(t)
    return best_t


def ri_arrays(cases_dict, run, coeffs=None):
    ps, ys = [], []
    for cid, d in run.items():
        r = cases_dict.get(cid)
        if r is None or d.get("ri24_prob") is None:
            continue
        vt = r.get("vmax_24")
        if not np.isfinite(vt):
            continue
        p = d["ri24_prob"]
        if isinstance(p, dict):  # some replies nest the probability per lead
            p = p.get("24", next(iter(p.values()), None))
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if coeffs:
            p = float(apply_platt([p], coeffs[0], coeffs[1])[0])
        ps.append(p)
        ys.append(1 if (vt - r["vmax"]) >= 30 else 0)
    return np.array(ps), np.array(ys)


def contingency(pred, y):
    h = int(np.sum(pred & (y == 1))); m = int(np.sum(~pred & (y == 1)))
    fa = int(np.sum(pred & (y == 0)))
    return dict(events=int(y.sum()),
                pod=round(h / max(h + m, 1), 3),
                far=round(fa / max(h + fa, 1), 3),
                csi=round(h / max(h + m + fa, 1), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-tag", default="agent_full_llama8b")
    ap.add_argument("--test-own", default="agent_full_llama8b_owncal")
    ap.add_argument("--test-transfer", default="agent_full_llama8b")
    args = ap.parse_args()

    fdir = work_dir("forecasts")
    with open(os.path.join(work_dir("cases"), "calib.pkl"), "rb") as f:
        ccases = pickle.load(f)
    with open(os.path.join(work_dir("cases"), "test.pkl"), "rb") as f:
        tcases = pickle.load(f)
    ccd = {case_id(r): r for _, r in ccases.iterrows()}
    tcd = {case_id(r): r for _, r in tcases.iterrows()}

    calib_run = load_forecasts(os.path.join(fdir, f"calib_{args.calib_tag}.jsonl"))
    print(f"calib run: {len(calib_run)} cycles")
    pc, yc = ri_arrays(ccd, calib_run)
    a, b = fit_platt(pc, yc)
    thr = csi_threshold(apply_platt(pc, a, b), yc)
    print(f"Llama own RI calibration: Platt a={a:.3f} b={b:.3f}, "
          f"frozen threshold {thr:.2f} (calib events {int(yc.sum())}/{len(yc)})")

    hybrid = load_forecasts(os.path.join(fdir, "test_hybrid_static.jsonl"))
    out = {}
    for name, tag in (("own-calibrated", args.test_own),
                      ("transferred", args.test_transfer)):
        path = os.path.join(fdir, f"test_{tag}.jsonl")
        if not os.path.exists(path):
            print(f"{name}: {path} missing, skip")
            continue
        run = load_forecasts(path)
        run = {k: v for k, v in run.items() if v.get("forecast")}
        # restrict comparison to the cycles this run covers
        tab = evaluate_methods(tcases, {"office": run, "hybrid": hybrid})
        rec = {}
        print(f"\n=== {name} ({tag}); n = "
              f"{sorted(set(tab[tab.method=='office'].n))} ===")
        for lead in (24, 48, 72):
            o = tab[(tab.method == "office") & (tab.lead == lead)].iloc[0]
            h = tab[(tab.method == "hybrid") & (tab.lead == lead)].iloc[0]
            pt = paired_test(tcases, run, hybrid, lead, "track")
            pv = paired_test(tcases, run, hybrid, lead, "vmax")
            rec[lead] = dict(track=round(float(o.track_km), 1),
                             hybrid_track=round(float(h.track_km), 1),
                             track_p=round(pt.get("p", np.nan), 4),
                             vmax=round(float(o.vmax_mae_kt), 2),
                             hybrid_vmax=round(float(h.vmax_mae_kt), 2),
                             vmax_p=round(pv.get("p", np.nan), 4))
            print(f"  {lead}h: track {rec[lead]['track']} vs hybrid "
                  f"{rec[lead]['hybrid_track']} (p={rec[lead]['track_p']}); "
                  f"vmax {rec[lead]['vmax']} vs {rec[lead]['hybrid_vmax']} "
                  f"(p={rec[lead]['vmax_p']})")
        pt_, yt_ = ri_arrays(tcd, run, coeffs=(a, b))
        ri = contingency(pt_ >= thr, yt_)
        ri["n"] = int(len(yt_))
        print(f"  RI (own Platt, thr {thr:.2f}): {ri}")
        out[name] = dict(points=rec, ri=ri)

    out["platt"] = dict(a=round(a, 3), b=round(b, 3), threshold=thr)
    with open(os.path.join(work_dir("results"), "llama_recal.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote llama_recal.json")


if __name__ == "__main__":
    main()
