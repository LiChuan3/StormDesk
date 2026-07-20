"""Produce non-agent forecasts for a split: persistence, CLIPER, DL
specialists, and (with --consensus, after calibration) the consensus family.

Writes <work>/forecasts/<split>_<method>.jsonl with rows
{case_id, forecast: {lead: {lat, lon, vmax}}}.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir, LEADS_H
from stormdesk.cliper import Cliper
from stormdesk.evaluate import case_id
from stormdesk.geo import motion_uv_kmh, destination, wrap_lon
from stormdesk.guidance.merge import load_guidance

VERIF = [24, 48, 72]


def persistence(row) -> dict:
    h = row["history"]
    u, v = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
    spd = float(np.hypot(u, v))
    brg = float((np.degrees(np.arctan2(u, v)) + 360) % 360)
    out = {}
    for l in LEADS_H:
        la, lo = destination(row["lat"], row["lon"], brg, spd * l)
        out[str(l)] = dict(lat=round(float(la), 2), lon=round(float(lo), 2),
                           vmax=row["vmax"])
    return out


def write(split, method, rows):
    path = os.path.join(work_dir("forecasts"), f"{split}_{method}.jsonl")
    with open(path, "w") as f:
        for cid, fc in rows:
            f.write(json.dumps(dict(case_id=cid, method=method, forecast=fc)) + "\n")
    print(f"{method}: {len(rows)} forecasts -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--methods", default="persistence,cliper,gru,transformer")
    ap.add_argument("--consensus", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    methods = args.methods.split(",") if args.methods else []

    if "persistence" in methods:
        write(args.split, "persistence",
              [(case_id(r), persistence(r)) for _, r in cases.iterrows()])

    if "cliper" in methods:
        cl = Cliper.load()
        rows = []
        for _, r in cases.iterrows():
            fc = cl.predict_case(r["history"], r["lat"], r["lon"], r["vmax"],
                                 r["basin"], r["init"])
            if fc:
                rows.append((case_id(r), {str(k): dict(lat=round(v["lat"], 2),
                                                       lon=round(v["lon"], 2),
                                                       vmax=round(v["vmax"], 1))
                                          for k, v in fc.items()}))
        write(args.split, "cliper", rows)

    dl_wanted = [m for m in ("gru", "transformer") if m in methods]
    if dl_wanted:
        import torch
        from stormdesk.guidance import dl_models as dm
        device = args.device if torch.cuda.is_available() else "cpu"
        for name in dl_wanted:
            model = dm.load_model(name, device)
            rows = []
            for _, r in cases.iterrows():
                fc = dm.predict_case(model, r["history"], r["lat"], r["lon"],
                                     r["vmax"], r["basin"], r["init"], device)
                if fc:
                    rows.append((case_id(r), {str(k): v for k, v in fc.items()}))
            write(args.split, name, rows)

    if args.consensus:
        from stormdesk.baselines import Calibration, consensus_equal, consensus_weighted
        calib = Calibration.load()
        guidance = load_guidance(args.split)
        eq, wt, bc, aiwp = [], [], [], []
        for _, r in cases.iterrows():
            cid = case_id(r)
            g = guidance.get(cid)
            if not g:
                continue
            eq.append((cid, consensus_equal(g)))
            wt.append((cid, consensus_weighted(g, calib, correct_bias=False)))
            bc.append((cid, consensus_weighted(g, calib, correct_bias=True)))
            g2 = {m: g[m] for m in ("pangu", "fengwu") if m in g}
            if g2:
                aiwp.append((cid, consensus_equal(g2)))
        write(args.split, "cons_equal", eq)
        write(args.split, "cons_weighted", wt)
        write(args.split, "cons_bc", bc)
        write(args.split, "cons_aiwp", aiwp)


if __name__ == "__main__":
    main()
