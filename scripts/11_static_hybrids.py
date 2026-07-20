"""Static task-decoupled hybrid baselines (no LLM at any stage).

hybrid_static  track from the AIWP consensus, Vmax from the bias-corrected
               weighted consensus - the strongest static per-task combination
               available in the member pool, chosen once and frozen.
hybrid_rules   hybrid_static passed through the deterministic physics-rule
               corrector (the same rules the auditor prompt encodes), plus a
               rule-based RI probability - a complete LLM-free "office".

With --correct-run <name>, additionally applies the deterministic corrector to
an existing agent run (e.g. agent_no_auditor_qwen14b) to isolate the LLM
auditor against a mechanical rule executor: writes <name>_detaudit.

Usage:
  python scripts/11_static_hybrids.py --split test
  python scripts/11_static_hybrids.py --split test --correct-run agent_no_auditor_qwen14b
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.analogs import AnalogLibrary, entry_features, summarize_analogs
from stormdesk.corrector import deterministic_corrector, ri_rule_fires
from stormdesk.evaluate import case_id, load_forecasts
from stormdesk.geo import motion_uv_kmh
from stormdesk.guidance.merge import load_features

VERIF = [24, 48, 72]


def write(split, method, rows):
    path = os.path.join(work_dir("forecasts"), f"{split}_{method}.jsonl")
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{method}: {len(rows)} forecasts -> {path}")


def case_context(r, feats, lib):
    """diag + analog summary for a case (same retrieval as the agent runs)."""
    cid = case_id(r)
    ft = feats.get(cid, {})
    diag = ft.get("diag") or {}
    summary = {}
    if diag:
        h = r["history"]
        mu, mv = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
        dv12 = (r["vmax"] - h[-3]["vmax"]) if h[-3]["vmax"] is not None else 0.0
        f = entry_features(r["lat"], r["lon"], r["vmax"], dv12,
                           float(mu), float(mv), diag, r["init"])
        if f is not None:
            analogs = lib.query(f, r["lat"], r["sid"], k=12)
            summary = summarize_analogs(analogs, r["vmax"])
    return diag, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--correct-run", default=None,
                    help="existing forecast method name to pass through the "
                         "deterministic corrector (writes <name>_detaudit)")
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    fdir = work_dir("forecasts")
    feats = load_features(args.split)
    lib = AnalogLibrary.load()

    if args.correct_run:
        src = load_forecasts(os.path.join(fdir, f"{args.split}_{args.correct_run}.jsonl"))
        rows = []
        for _, r in cases.iterrows():
            cid = case_id(r)
            d = src.get(cid)
            if not d or not d.get("forecast"):
                continue
            diag, summary = case_context(r, feats, lib)
            init = dict(lat=r["lat"], lon=r["lon"], vmax=r["vmax"])
            fc = deterministic_corrector(d["forecast"], init, diag, summary)
            rows.append(dict(case_id=cid, method=f"{args.correct_run}_detaudit",
                             forecast=fc, ri24_prob=d.get("ri24_prob")))
        write(args.split, f"{args.correct_run}_detaudit", rows)
        return

    aiwp = load_forecasts(os.path.join(fdir, f"{args.split}_cons_aiwp.jsonl"))
    bc = load_forecasts(os.path.join(fdir, f"{args.split}_cons_bc.jsonl"))

    hybrid, hybrid_rules = [], []
    for _, r in cases.iterrows():
        cid = case_id(r)
        da, db = aiwp.get(cid), bc.get(cid)
        if not da or not da.get("forecast"):
            continue
        fa = da["forecast"]
        fb = db["forecast"] if (db and db.get("forecast")) else {}
        fc = {}
        for l in VERIF:
            ea = fa.get(str(l))
            if ea is None:
                continue
            eb = fb.get(str(l)) or {}
            fc[str(l)] = dict(lat=ea["lat"], lon=ea["lon"], vmax=eb.get("vmax"))
        if not fc:
            continue
        hybrid.append(dict(case_id=cid, method="hybrid_static", forecast=fc))

        diag, summary = case_context(r, feats, lib)
        init = dict(lat=r["lat"], lon=r["lon"], vmax=r["vmax"])
        fc2 = deterministic_corrector(fc, init, diag, summary)
        ri_p = 0.55 if ri_rule_fires(diag, summary) else 0.08
        hybrid_rules.append(dict(case_id=cid, method="hybrid_rules", forecast=fc2,
                                 ri24_prob=ri_p))
    write(args.split, "hybrid_static", hybrid)
    write(args.split, "hybrid_rules", hybrid_rules)


if __name__ == "__main__":
    main()
