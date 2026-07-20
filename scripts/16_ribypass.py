"""RI-commit shrinkage bypass (transcript replay; no LLM calls).

The office's affine post-calibration (a≈0.5) halves the specialist delta, so
even a full +25 kt RI commit adds only ~12.5 kt deterministically -- the
office's deterministic channel cannot report RI by design. This script replays
the recorded deliberations with a bypass rule: when the office committed to RI
(final ri24_prob >= 0.5 and specialist delta24 >= +20), the 24-h intensity
uses the raw (unshrunk) delta instead. Quantifies the tradeoff between
deterministic RI detection and intensity MAE.

Writes <split>_<tag>_ribypass.jsonl and prints the comparison.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.agents.auditor import apply_hard_caps
from stormdesk.evaluate import case_id, evaluate_methods, load_forecasts, ri_scores
from stormdesk.guidance.merge import load_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="agent_full_qwen14b")
    ap.add_argument("--prob-min", type=float, default=0.5,
                    help="raw office RI probability required to bypass")
    ap.add_argument("--delta-min", type=float, default=20.0,
                    help="specialist raw delta24 required to bypass")
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    feats = load_features(args.split)
    fc_path = os.path.join(work_dir("forecasts"), f"{args.split}_{args.tag}.jsonl")
    tr_path = os.path.join(work_dir("transcripts"), f"{args.split}_{args.tag}.jsonl")
    fcs = load_forecasts(fc_path)
    trs = {}
    with open(tr_path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                trs[d["case_id"]] = d.get("transcript") or {}

    by_cid = {case_id(r): r for _, r in cases.iterrows()}
    n_bypass = 0
    rows = []
    for cid, d in fcs.items():
        fc = d.get("forecast")
        if not fc:
            continue
        out = {l: dict(e) for l, e in fc.items()}
        t = trs.get(cid) or {}
        it = t.get("intensity") or {}
        prior = (t.get("prior_vmax") or {}).get("24")
        p = d.get("ri24_prob")
        try:
            delta24 = float((it.get("delta_kt") or {}).get("24"))
        except (TypeError, ValueError):
            delta24 = None
        r = by_cid.get(cid)
        if (r is not None and prior is not None and delta24 is not None
                and p is not None and float(p) >= args.prob_min
                and delta24 >= args.delta_min and out.get("24", {}).get("vmax") is not None):
            raw = float(prior) + float(np.clip(delta24, -25, 25))
            if raw > out["24"]["vmax"]:
                out["24"]["vmax"] = round(raw, 1)
                diag = (feats.get(cid, {}) or {}).get("diag") or {}
                init = dict(lat=r["lat"], lon=r["lon"], vmax=r["vmax"])
                out = apply_hard_caps(out, init, diag)
                n_bypass += 1
        rows.append(dict(case_id=cid, method=f"{args.tag}_ribypass", forecast=out,
                         ri24_prob=p))

    out_path = os.path.join(work_dir("forecasts"), f"{args.split}_{args.tag}_ribypass.jsonl")
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"bypassed {n_bypass} cycles -> {out_path}")

    methods = {
        "office": fcs,
        "office_ribypass": load_forecasts(out_path),
        "hybrid_static": load_forecasts(os.path.join(work_dir("forecasts"),
                                                     f"{args.split}_hybrid_static.jsonl")),
    }
    tab = evaluate_methods(cases, methods)
    import pandas as pd
    pd.set_option("display.width", 200)
    print(tab[["method", "lead", "n", "track_km", "vmax_mae_kt", "vmax_bias_kt"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\ndeterministic RI (dV >= +30 kt):")
    print(ri_scores(cases, methods).to_string(index=False,
                                              float_format=lambda x: f"{x:.3f}"))
    print("\nprobabilistic RI (calibrated threshold):")
    print(ri_scores(cases, {k: v for k, v in methods.items() if k.startswith("office")},
                    use_prob=True).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
