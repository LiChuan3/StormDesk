"""Dump per-case learned-gate member weights for a split, so the office can be
given the supervised combiner's suggestion as an extra briefing line (R1/Q3:
can the LLM close the gap when handed the gate's advice?).

Fits the same track gate as scripts/19 on the calibration season and writes
<work>/models/gate_weights_<split>.json: {case_id: {lead: {member: weight}}}.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk import combiner as CB
from stormdesk.baselines import Calibration
from stormdesk.config import work_dir
from stormdesk.guidance.merge import load_guidance, load_features

VERIF = [24, 48, 72]


def load_split(split):
    with open(os.path.join(work_dir("cases"), f"{split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    return cases, load_features(split), load_guidance(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--calib-split", default="calib")
    args = ap.parse_args()
    calib = Calibration.load()
    c_cases, c_feats, c_guid = load_split(args.calib_split)
    t_cases, t_feats, t_guid = load_split(args.split)

    tr = CB.build_track_table(c_cases, c_feats, c_guid, calib, CB.PRIMARY)
    keys = CB.CASE_KEYS + CB.MEM_KEYS
    models, meds = {}, {}
    for lead in VERIF:
        sub = tr[tr.lead == lead]
        X, med = CB.enc(sub, keys, CB.PRIMARY)
        y = np.log(np.clip(sub["err"].values, 1.0, None))
        models[lead] = CB.fit_gbt_reg(X, y)
        meds[lead] = med

    te = CB.build_track_table(t_cases, t_feats, t_guid, calib, CB.PRIMARY, with_truth=False)
    out = {}
    for lead in VERIF:
        sub = te[te.lead == lead]
        if sub.empty:
            continue
        X, _ = CB.enc(sub, keys, CB.PRIMARY, meds[lead])
        pred = models[lead].predict(X)
        sub = sub.assign(w=1.0 / np.exp(pred) ** 2)
        for cid, grp in sub.groupby("case_id"):
            w = grp["w"].values
            w = w / w.sum()
            out.setdefault(cid, {})[str(lead)] = {
                m: round(float(wi), 3) for m, wi in zip(grp["member"].values, w)}
    path = os.path.join(work_dir("models"), f"gate_weights_{args.split}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"gate weights for {len(out)} cases -> {path}")


if __name__ == "__main__":
    main()
