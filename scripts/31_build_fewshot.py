"""Build worked examples for the track specialist (featmatch_fs mode).

Picks calibration-season cases and pairs a compact briefing extract (guidance,
spread, member skill, prior weights, member track diagnostics -- exactly what
the track specialist sees) with the supervised gate's recommended trust factors
(gate weight / skill-prior weight, clipped to the office contract [0.25,4]).
Selection (calibration data only): two high-signal cases where the gate's
reweighting strongly beat the prior-weight consensus, from different basins,
plus one near-uniform case (trust ~1 everywhere) so the examples also teach
when NOT to move. Writes <work>/models/fewshot_track.json.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk import combiner as CB
from stormdesk.agents.office import build_briefing, prior_weights
from stormdesk.baselines import Calibration, _members_at
from stormdesk.config import work_dir
from stormdesk.evaluate import case_id
from stormdesk.geo import gc_distance_km
from stormdesk.guidance.merge import load_guidance, load_features

VERIF = [24, 48, 72]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-split", default="calib")
    ap.add_argument("--n-examples", type=int, default=3)
    args = ap.parse_args()

    calib = Calibration.load()
    with open(os.path.join(work_dir("cases"), f"{args.calib_split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    feats = load_features(args.calib_split)
    guidance = load_guidance(args.calib_split)

    print("fitting the GBT gate on the calibration seasons ...")
    tr = CB.build_track_table(cases, feats, guidance, calib, CB.PRIMARY)
    keys_full = CB.CASE_KEYS + CB.MEM_KEYS
    models, meds = {}, {}
    for lead in VERIF:
        sub = tr[tr.lead == lead]
        y = np.log(np.clip(sub["err"].values, 1.0, None))
        X, med = CB.enc(sub, keys_full, CB.PRIMARY)
        models[lead] = CB.fit_gbt_reg(X, y)
        meds[lead] = med

    te = CB.build_track_table(cases, feats, guidance, calib, CB.PRIMARY,
                              with_truth=True)
    cand = []
    for _, r in cases.iterrows():
        cid = case_id(r)
        g = guidance.get(cid)
        if not g:
            continue
        trust = {}
        gain24, spread24 = None, None
        ok = True
        for lead in VERIF:
            sub = te[(te.lead == lead) & (te.case_id == cid)]
            if len(sub) < 2:
                ok = False
                break
            X, _ = CB.enc(sub, keys_full, CB.PRIMARY, meds[lead])
            pred = models[lead].predict(X)
            w = 1.0 / np.exp(pred) ** 2
            w = w / w.sum()
            pw = prior_weights({m: g[m] for m in sub["member"].values if m in g},
                               calib, lead)
            t_l = {}
            for m, wi in zip(sub["member"].values, w):
                p = pw.get(m, 0.0)
                ratio = float(np.clip(wi / max(p, 1e-6), 0.25, 4.0))
                t_l[m] = round(ratio, 2)
            trust[str(lead)] = t_l
            if lead == 24:
                # realized: gate-weighted vs prior-weighted consensus error
                names = sub["member"].values
                glat, glon = CB.assemble_from_weights(
                    sub["lat"].values, sub["lon"].values.tolist(), w)
                wp = np.array([pw.get(m, 0.0) for m in names])
                if wp.sum() <= 1e-9:
                    wp = np.ones(len(names))
                plat, plon = CB.assemble_from_weights(
                    sub["lat"].values, sub["lon"].values.tolist(), wp)
                tla, tlo = sub["tlat"].iloc[0], sub["tlon"].iloc[0]
                gain24 = float(gc_distance_km(tla, tlo, plat, plon)
                               - gc_distance_km(tla, tlo, glat, glon))
                spread24 = float(np.std(list(t_l.values())))
        if not ok or gain24 is None:
            continue
        cand.append(dict(cid=cid, r=r, trust=trust, gain24=gain24,
                         spread24=spread24, basin=r["basin"]))

    cand.sort(key=lambda c: -c["gain24"])
    chosen, basins = [], set()
    for c in cand:  # two decisive examples, different basins
        if c["spread24"] > 0.5 and c["basin"] not in basins:
            chosen.append(c)
            basins.add(c["basin"])
        if len(chosen) == args.n_examples - 1:
            break
    picked = {c["cid"] for c in chosen}
    uniform = min((c for c in cand if c["cid"] not in picked),
                  key=lambda c: c["spread24"] + abs(c["gain24"]) / 50.0)
    chosen.append(uniform)

    out = []
    for c in chosen:
        r = c["r"]
        cid = c["cid"]
        g = guidance[cid]
        brief = build_briefing(dict(r), {}, None, g, calib, [], {},
                               include_diag=False, include_analogs=False,
                               prior_vmax=None, feature_block=True)
        # keep members present in the trust dict consistent with the briefing
        out.append(dict(case_id=cid, briefing=brief, trust=c["trust"],
                        gain24_km=round(c["gain24"], 1)))
        print(f"  example {cid} ({c['basin']}): gate gain at 24h "
              f"{c['gain24']:+.1f} km, trust spread {c['spread24']:.2f}")

    path = os.path.join(work_dir("models"), "fewshot_track.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(out)} examples -> {path}")


if __name__ == "__main__":
    main()
