"""Analog point-intensity baselines (review other-issue #8).

Tests whether the LLM's intensity delta is merely a restatement of the analog
median. Produces, using the same analog retrieval the office briefing uses:
  analog_median   vmax[lead] = init_vmax + analog median dV[lead]
  analog_linear   per-lead linear calibration truth-init ~ a*analog_median + b
                  fit on the calibration season (a kNN residual regressor)
Both carry the AIWP-consensus track so intensity MAE is directly comparable.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.analogs import AnalogLibrary, entry_features, summarize_analogs
from stormdesk.config import work_dir
from stormdesk.evaluate import case_id, load_forecasts
from stormdesk.geo import motion_uv_kmh
from stormdesk.guidance.merge import load_features

VERIF = [24, 48, 72]


def analog_medians(split, lib):
    with open(os.path.join(work_dir("cases"), f"{split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    feats = load_features(split)
    out = {}
    for _, r in cases.iterrows():
        cid = case_id(r)
        diag = (feats.get(cid, {}) or {}).get("diag") or {}
        if not diag:
            continue
        h = r["history"]
        mu, mv = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
        dv12 = (r["vmax"] - h[-3]["vmax"]) if h[-3]["vmax"] is not None else 0.0
        f = entry_features(r["lat"], r["lon"], r["vmax"], dv12, float(mu), float(mv),
                           diag, r["init"])
        if f is None:
            continue
        s = summarize_analogs(lib.query(f, r["lat"], r["sid"], k=12), r["vmax"])
        med = {l: s.get(f"dv{l}_median") for l in VERIF}
        out[cid] = dict(v0=float(r["vmax"]), med=med,
                        truth={l: (float(r.get(f"vmax_{l}"))
                                   if np.isfinite(r.get(f"vmax_{l}")) else None)
                               for l in VERIF})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--calib-split", default="calib")
    args = ap.parse_args()
    lib = AnalogLibrary.load()

    cal = analog_medians(args.calib_split, lib)
    tst = analog_medians(args.split, lib)

    # fit per-lead linear calibration on calib: (truth - v0) ~ a*med + b
    coef = {}
    for lead in VERIF:
        X, y = [], []
        for d in cal.values():
            m = d["med"].get(lead); t = d["truth"].get(lead)
            if m is None or t is None:
                continue
            X.append(m); y.append(t - d["v0"])
        X, y = np.array(X), np.array(y)
        A = np.vstack([X, np.ones_like(X)]).T
        ab, *_ = np.linalg.lstsq(A, y, rcond=None)
        coef[lead] = (float(ab[0]), float(ab[1]))
    print("analog_linear coefs (a,b):", {l: tuple(round(x, 3) for x in coef[l]) for l in VERIF})

    aiwp = load_forecasts(os.path.join(work_dir("forecasts"), f"{args.split}_cons_aiwp.jsonl"))
    med_rows, lin_rows = {}, {}
    for cid, d in tst.items():
        trk = (aiwp.get(cid) or {}).get("forecast", {})
        fm, fl = {}, {}
        for lead in VERIF:
            m = d["med"].get(lead)
            e = trk.get(str(lead))
            if m is None or not e:
                continue
            a, b = coef[lead]
            fm[str(lead)] = dict(lat=e["lat"], lon=e["lon"],
                                 vmax=round(float(np.clip(d["v0"] + m, 15, 185)), 1))
            fl[str(lead)] = dict(lat=e["lat"], lon=e["lon"],
                                 vmax=round(float(np.clip(d["v0"] + a * m + b, 15, 185)), 1))
        if fm:
            med_rows[cid] = fm; lin_rows[cid] = fl
    for name, rows in (("analog_median", med_rows), ("analog_linear", lin_rows)):
        path = os.path.join(work_dir("forecasts"), f"{args.split}_{name}.jsonl")
        with open(path, "w") as f:
            for cid, fc in rows.items():
                f.write(json.dumps(dict(case_id=cid, method=name, forecast=fc)) + "\n")
        print(f"{name}: {len(rows)} -> {path}")


if __name__ == "__main__":
    main()
