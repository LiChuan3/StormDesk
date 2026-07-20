"""Basin-wise RI and reliability (calibration) diagrams (R6/Q6).

Uses the calibration-frozen protocol: the office ri24_prob (Platt-calibrated)
and a logistic RI classifier (refit on calib features), scored on the same
homogeneous test sample. Reports RI contingency per basin and writes a
reliability diagram (predicted vs observed RI frequency by probability bin) for
office-14B vs logistic to figures/fig_ri_reliability.pdf.
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.evaluate import (case_id, load_forecasts, ri_prob_calibration,
                                calibrate_ri_prob)

_spec = importlib.util.spec_from_file_location(
    "ri12", os.path.join(os.path.dirname(os.path.abspath(__file__)), "12_ri_baselines.py"))
_ri12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ri12)


def contingency(pred, y):
    h = int(np.sum(pred & (y == 1))); m = int(np.sum(~pred & (y == 1)))
    fa = int(np.sum(pred & (y == 0)))
    return dict(events=int(y.sum()), pod=round(h / (h + m), 3) if (h + m) else np.nan,
                far=round(fa / (h + fa), 3) if (h + fa) else np.nan,
                csi=round(h / (h + m + fa), 3) if (h + m + fa) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    dfc = _ri12.build_matrix("calib"); dft = _ri12.build_matrix(args.split)
    Zc, stats = _ri12.impute_standardize(dfc); Zt, _ = _ri12.impute_standardize(dft, stats)
    yc, yt = dfc["y"].values, dft["y"].values

    # logistic (grouped CV threshold + Platt), refit on calib
    from sklearn.linear_model import LogisticRegression
    def fit(X, y): return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(X, y)
    def prob(m, X): return m.predict_proba(X)[:, 1]
    oof = _ri12.cv_oof_probs(fit, prob, dfc, Zc, yc)
    a, b = _ri12.fit_platt(oof, yc); thr = _ri12.csi_threshold(_ri12.apply_platt(oof, a, b), yc)
    logit = fit(Zc, yc)
    p_log = _ri12.apply_platt(prob(logit, Zt), a, b)

    # office probabilities
    office = ri_prob_calibration() or {}
    fc = load_forecasts(os.path.join(work_dir("forecasts"), f"{args.split}_agent_full_qwen14b.jsonl"))
    cidpos = {c: i for i, c in enumerate(dft.case_id.values)}
    p_off = np.full(len(dft), np.nan)
    for c, d in fc.items():
        if c in cidpos and d.get("ri24_prob") is not None:
            p_off[cidpos[c]] = calibrate_ri_prob(float(d["ri24_prob"]), office)
    off_thr = office.get("threshold", 0.14)

    mask = np.isfinite(p_off) & np.isfinite(p_log)
    y = yt[mask]; po = p_off[mask]; pl = p_log[mask]
    # basin per case
    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cs = pickle.load(f)
    basin = {case_id(r): r["basin"] for _, r in cs.iterrows()}
    cids = dft.case_id.values[mask]
    bas = np.array([basin.get(c, "??") for c in cids])

    print("=== basin-wise RI (office-14B / logistic) ===")
    rows = {}
    for bname in sorted(set(bas)):
        sel = bas == bname
        if sel.sum() < 20 or y[sel].sum() < 3:
            continue
        co = contingency(po[sel] >= off_thr, y[sel]); cl = contingency(pl[sel] >= thr, y[sel])
        rows[bname] = dict(n=int(sel.sum()), office=co, logistic=cl)
        print(f"  {bname} (n={sel.sum()}, ev={int(y[sel].sum())}): "
              f"office POD/FAR/CSI {co['pod']}/{co['far']}/{co['csi']} | "
              f"logit {cl['pod']}/{cl['far']}/{cl['csi']}")
    json.dump(rows, open(os.path.join(work_dir("results"), f"{args.split}_ri_basin.json"), "w"), indent=1)

    # reliability curve
    bins = np.linspace(0, 1, 11)
    def rel(p):
        out = []
        for i in range(10):
            sel = (p >= bins[i]) & (p < bins[i + 1] if i < 9 else p <= 1.0)
            if sel.sum() >= 5:
                out.append((float(p[sel].mean()), float(y[sel].mean()), int(sel.sum())))
        return out
    rel_off, rel_log = rel(po), rel(pl)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="perfect")
        ax.axhline(float(y.mean()), color="#888", lw=0.7, ls="--", label="climatology")
        for data, c, lab in [(rel_off, "#C03028", "StormDesk (14B)"), (rel_log, "#2E6FA7", "Logistic")]:
            if data:
                xs, ys, ns = zip(*data)
                ax.plot(xs, ys, "-o", color=c, ms=4, lw=1.6, label=lab)
        ax.set_xlabel("Forecast RI probability"); ax.set_ylabel("Observed RI frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7, frameon=False)
        ax.set_title("RI reliability (test)", fontsize=9, loc="left")
        fig.tight_layout()
        p = os.path.join(work_dir("figures"), "fig_ri_reliability.pdf")
        fig.savefig(p); print(f"\nwrote {p}")
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
