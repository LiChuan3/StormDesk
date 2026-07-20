"""Nudge (bounded displacement) usage statistics from office transcripts.

Reports per-lead usage frequency and magnitude of the Track Specialist's
<=60 km displacement nudge, and its verified effect: track error with the
nudge (as issued) vs. the same trust-weighted consensus without it.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.agents.office import assemble_track
from stormdesk.baselines import Calibration
from stormdesk.evaluate import case_id
from stormdesk.geo import gc_distance_km
from stormdesk.guidance.merge import load_guidance

VERIF = [24, 48, 72]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="agent_full_qwen14b")
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    guidance = load_guidance(args.split)
    calib = Calibration.load()
    trs = {}
    with open(os.path.join(work_dir("transcripts"), f"{args.split}_{args.tag}.jsonl")) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                trs[d["case_id"]] = d.get("transcript") or {}

    by_cid = {case_id(r): r for _, r in cases.iterrows()}
    n_total = 0
    used = {l: [] for l in VERIF}       # magnitudes km
    effect = {l: [] for l in VERIF}     # err_with - err_without (negative = helped)
    for cid, t in trs.items():
        tr = t.get("track") or {}
        nudge = tr.get("nudge") or {}
        trust = tr.get("trust") or {}
        r = by_cid.get(cid)
        g = guidance.get(cid)
        if r is None or not g:
            continue
        n_total += 1
        any_used = False
        for l in VERIF:
            nd = nudge.get(str(l)) if isinstance(nudge, dict) else None
            km = None
            if isinstance(nd, dict):
                try:
                    km = float(nd.get("km") or 0)
                except (TypeError, ValueError):
                    km = 0.0
            if km and km > 0:
                used[l].append(min(km, 60.0))
                any_used = True
        if not any_used:
            continue
        la = {l: r.get(f"lat_{l}") for l in VERIF}
        lo = {l: r.get(f"lon_{l}") for l in VERIF}
        t_with = assemble_track(g, calib, trust, nudge)
        t_wo = assemble_track(g, calib, trust, None)
        for l in VERIF:
            nd = nudge.get(str(l)) if isinstance(nudge, dict) else None
            if not (isinstance(nd, dict) and nd.get("km")):
                continue
            if not (np.isfinite(la[l]) and np.isfinite(lo[l])):
                continue
            k = str(l)
            if k in t_with and k in t_wo:
                e1 = float(gc_distance_km(la[l], lo[l], t_with[k]["lat"], t_with[k]["lon"]))
                e0 = float(gc_distance_km(la[l], lo[l], t_wo[k]["lat"], t_wo[k]["lon"]))
                effect[l].append(e1 - e0)

    print(f"cycles with transcripts: {n_total}")
    for l in VERIF:
        u, e = used[l], np.array(effect[l])
        frac = len(u) / max(n_total, 1)
        line = f"lead {l}h: nudge used in {len(u)} cycles ({frac:.1%})"
        if u:
            line += f", median {np.median(u):.0f} km (p90 {np.percentile(u, 90):.0f})"
        if len(e):
            line += (f"; effect on error: mean {e.mean():+.1f} km, median {np.median(e):+.1f}, "
                     f"helped in {(e < 0).mean():.0%} of used cycles")
        print(line)


if __name__ == "__main__":
    main()
