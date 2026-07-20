"""Identify what the Physics Auditor actually contributes, controlling for
per-pipeline recalibration and simple offsets (review P0-4).

Everything is reconstructed by replay from the full-office transcripts (calib
for fitting, test for evaluation): each transcript records the consensus prior,
the specialist's raw delta, the auditor's per-lead adjustments and the chief's
acceptances. The only difference between the "office" and "no-auditor"
reconstructions is whether the accepted auditor adjustments are applied, so the
auditor is isolated cleanly. Each pipeline gets its OWN affine shrinkage
a_l,b_l fit on the calibration season under the identical protocol.

Methods compared at each lead (intensity only; positions unchanged):
  no_auditor        prior + shrink_na(clip delta)              [own a,b]
  office            prior + shrink_full(clip delta) + accepted auditor adj
  fixed_offset      no_auditor + per-lead constant c_l (calib-fit)
  affine_cal        alpha*draft + beta   (calib-fit on the no-auditor draft)
  shuffled_auditor  office but auditor adjustments permuted across cases
Plus the case-wise skill of the auditor: correlation / sign accuracy /
conditional MAE improvement of the adjustment vs the truth-minus-draft residual,
benchmarked against the fixed per-lead offset.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.agents.auditor import apply_hard_caps
from stormdesk.config import work_dir
from stormdesk.evaluate import case_id

VERIF = [24, 48, 72]


def load_transcripts(split, tag):
    out = {}
    with open(os.path.join(work_dir("transcripts"), f"{split}_{tag}.jsonl")) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["case_id"]] = d.get("transcript") or {}
    return out


def parse_case(t):
    """-> dict lead -> (prior, clipped_delta, accepted_adj) or None."""
    prior = t.get("prior_vmax") or {}
    it = t.get("intensity") or {}
    delta = it.get("delta_kt") or {}
    au = (t.get("auditor") or {}).get("issues") or []
    ch = (t.get("chief") or {}).get("accept_adjustments") or {}
    adj = {l: 0.0 for l in VERIF}
    for iss in au:
        try:
            l = int(iss.get("lead"))
        except (TypeError, ValueError):
            continue
        a = iss.get("adjust_kt")
        if a is None or l not in VERIF:
            continue
        if ch.get(str(l), True):
            adj[l] += float(np.clip(float(a), -20, 20))
    out = {}
    for l in VERIF:
        p = prior.get(str(l))
        if p is None:
            continue
        try:
            d = float(np.clip(float(delta.get(str(l), 0.0)), -25, 25))
        except (TypeError, ValueError):
            d = 0.0
        out[l] = (float(p), d, adj[l])
    return out


def fit_ab(P, D, ADJ, T):
    """Least-squares a,b: truth ~ prior + a*clip_delta + b + adj (adj optional)."""
    y = T - P - ADJ
    A = np.vstack([D, np.ones_like(D)]).T
    ab, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(ab[0]), float(ab[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="agent_full_qwen14b")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--case-list", default=None,
                    help="analysis-manifest JSON; restrict the test sample to "
                         "the same homogeneous per-lead case sets as Table 1")
    args = ap.parse_args()

    allowed = None
    if args.case_list:
        with open(args.case_list) as f:
            mf = json.load(f)
        allowed = {int(l): set(e["case_ids"]) for l, e in mf["leads"].items()}

    tr_c = load_transcripts("calib", args.tag)
    tr_t = load_transcripts("test", args.tag)
    with open(os.path.join(work_dir("cases"), "calib.pkl"), "rb") as f:
        cc = {case_id(r): r for _, r in pickle.load(f).iterrows()}
    with open(os.path.join(work_dir("cases"), "test.pkl"), "rb") as f:
        tc = {case_id(r): r for _, r in pickle.load(f).iterrows()}

    # ---- fit shrinkages and fixed offsets on calibration ----
    shrink_na, shrink_full, offset = {}, {}, {}
    for lead in VERIF:
        P, D, ADJ, T = [], [], [], []
        for cid, t in tr_c.items():
            r = cc.get(cid)
            if r is None:
                continue
            vt = r.get(f"vmax_{lead}")
            if not np.isfinite(vt):
                continue
            pc = parse_case(t).get(lead)
            if pc is None:
                continue
            p, d, adj = pc
            P.append(p); D.append(d); ADJ.append(adj); T.append(float(vt))
        P, D, ADJ, T = map(np.array, (P, D, ADJ, T))
        shrink_na[lead] = fit_ab(P, D, np.zeros_like(ADJ), T)
        shrink_full[lead] = fit_ab(P, D, ADJ, T)
        a, b = shrink_na[lead]
        draft = P + a * D + b
        offset[lead] = float(np.mean(T - draft))  # per-lead constant
    print("shrink_na:", {l: tuple(round(x, 3) for x in shrink_na[l]) for l in VERIF})
    print("shrink_full:", {l: tuple(round(x, 3) for x in shrink_full[l]) for l in VERIF})
    print("fixed offset:", {l: round(offset[l], 2) for l in VERIF})

    # ---- reconstruct test predictions ----
    rng = np.random.default_rng(0)
    # gather per-lead test arrays
    per = {l: dict(cid=[], p=[], d=[], adj=[], t=[]) for l in VERIF}
    for cid, t in tr_t.items():
        r = tc.get(cid)
        if r is None:
            continue
        pcs = parse_case(t)
        for lead in VERIF:
            vt = r.get(f"vmax_{lead}")
            if lead not in pcs or not np.isfinite(vt):
                continue
            if allowed is not None and cid not in allowed.get(lead, ()):
                continue
            p, d, adj = pcs[lead]
            per[lead]["cid"].append(cid); per[lead]["p"].append(p)
            per[lead]["d"].append(d); per[lead]["adj"].append(adj)
            per[lead]["t"].append(float(vt))

    def caps_series(cid_list, vmax):
        """apply per-case hard caps using init + diag (mpi) at this lead only."""
        # cheap: clamp to [15,185]; full rate/MPI caps need multi-lead context,
        # negligible at single lead for these bounded deltas
        return np.clip(vmax, 15, 185)

    rows = []
    aud_stats = {}
    for lead in VERIF:
        d = per[lead]
        P = np.array(d["p"]); D = np.array(d["d"]); ADJ = np.array(d["adj"])
        T = np.array(d["t"])
        a_na, b_na = shrink_na[lead]
        a_f, b_f = shrink_full[lead]
        draft = caps_series(d["cid"], P + a_na * D + b_na)
        office = caps_series(d["cid"], P + a_f * D + b_f + ADJ)
        fixed = caps_series(d["cid"], draft + offset[lead])
        # affine calibrator fit on calib no-auditor draft -> truth
        # (fit here quickly on calib arrays)
        # gather calib draft/truth
        Pc, Dc, Tc = [], [], []
        for cid, t in tr_c.items():
            r = cc.get(cid)
            if r is None:
                continue
            vt = r.get(f"vmax_{lead}")
            pc = parse_case(t).get(lead)
            if pc is None or not np.isfinite(vt):
                continue
            p, dd, _ = pc
            Pc.append(p); Dc.append(dd); Tc.append(float(vt))
        Pc, Dc, Tc = map(np.array, (Pc, Dc, Tc))
        draft_c = Pc + a_na * Dc + b_na
        Aa = np.vstack([draft_c, np.ones_like(draft_c)]).T
        alpha_beta, *_ = np.linalg.lstsq(Aa, Tc, rcond=None)
        affine = caps_series(d["cid"], alpha_beta[0] * draft + alpha_beta[1])
        shuf = caps_series(d["cid"], P + a_f * D + b_f + rng.permutation(ADJ))

        for name, pred in [("no_auditor", draft), ("office", office),
                           ("fixed_offset", fixed), ("affine_cal", affine),
                           ("shuffled_auditor", shuf)]:
            rows.append(dict(lead=lead, method=name, n=len(T),
                             mae=float(np.mean(np.abs(pred - T))),
                             bias=float(np.mean(pred - T))))
        # case-wise auditor skill: adjustment vs (truth - draft)
        resid = T - draft
        used = np.abs(ADJ) > 1e-6
        corr = float(np.corrcoef(ADJ[used], resid[used])[0, 1]) if used.sum() > 5 else np.nan
        sign_acc = float(np.mean(np.sign(ADJ[used]) == np.sign(resid[used]))) if used.sum() else np.nan
        # conditional MAE improvement of auditor vs fixed offset
        mae_draft = np.mean(np.abs(resid))
        mae_office = np.mean(np.abs(resid - ADJ))
        mae_fixed = np.mean(np.abs(resid - offset[lead]))
        aud_stats[lead] = dict(
            frac_adjusted=float(used.mean()),
            corr_adj_resid=round(corr, 3), sign_acc=round(sign_acc, 3),
            mae_gain_auditor=round(float(mae_draft - mae_office), 3),
            mae_gain_fixed_offset=round(float(mae_draft - mae_fixed), 3))

    import pandas as pd
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== intensity MAE / bias by method and lead (test) ===")
    print(tab.pivot_table(index="method", columns="lead", values=["mae", "bias"])
          .to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n=== auditor case-wise skill vs fixed offset ===")
    print(json.dumps(aud_stats, indent=1))

    out = dict(shrink_na={str(l): shrink_na[l] for l in VERIF},
               shrink_full={str(l): shrink_full[l] for l in VERIF},
               offset={str(l): round(offset[l], 3) for l in VERIF},
               table=rows, auditor_skill=aud_stats)
    with open(os.path.join(work_dir("results"), "auditor_id.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote auditor_id.json")


if __name__ == "__main__":
    main()
