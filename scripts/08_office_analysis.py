"""Analyze office deliberation transcripts (v2 contract): trust-factor
behavior and its relation to per-case member skill, delta statistics, audit
rates, RI-probability reliability, and cost accounting.

Usage: python scripts/08_office_analysis.py --split test --tag agent_full_qwen14b
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.evaluate import case_id
from stormdesk.geo import gc_distance_km
from stormdesk.guidance.merge import load_guidance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="agent_full_qwen14b")
    args = ap.parse_args()

    tr_path = os.path.join(work_dir("transcripts"), f"{args.split}_{args.tag}.jsonl")
    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    truth = {case_id(r): r for _, r in cases.iterrows()}
    guidance = load_guidance(args.split)

    trust_rows = []
    delta_rows = []
    audits = dict(n=0, cases_with_code_issue=0, auditor_revise=0,
                  adj_proposed=0, adj_accepted=0, final_violations=0)
    ri_probs, ri_obs = [], []
    seen = set()
    n = 0
    for line in open(tr_path):
        if not line.strip():
            continue
        d = json.loads(line)
        cid = d["case_id"]
        if cid in seen:
            continue
        seen.add(cid)
        t = d.get("transcript") or {}
        r = truth.get(cid)
        g = guidance.get(cid, {})
        if r is None:
            continue
        n += 1
        # ----- trust factors vs member case error -----
        tr = (t.get("track") or t.get("single")) or {}
        for lead, tf in (tr.get("trust") or {}).items():
            la, lo = r.get(f"lat_{lead}"), r.get(f"lon_{lead}")
            if not (isinstance(tf, dict) and np.isfinite(la)):
                continue
            for m, f in tf.items():
                e = (g.get(m) or {}).get(lead)
                if e is None:
                    continue
                try:
                    f = float(f)
                except (TypeError, ValueError):
                    continue
                err = float(gc_distance_km(la, lo, e["lat"], e["lon"]))
                trust_rows.append(dict(member=m, lead=int(lead),
                                       trust=float(np.clip(f, 0.25, 4)), err=err))
        # ----- deltas -----
        it = (t.get("intensity") or t.get("single")) or {}
        pv = t.get("prior_vmax") or {}
        for lead, dd in (it.get("delta_kt") or {}).items():
            vt = r.get(f"vmax_{lead}")
            p = pv.get(lead)
            if p is None or not np.isfinite(vt):
                continue
            try:
                dd = float(dd)
            except (TypeError, ValueError):
                continue
            delta_rows.append(dict(lead=int(lead), delta=dd, ideal=vt - p))
        # ----- audit -----
        audits["n"] += 1
        if t.get("code_audit"):
            audits["cases_with_code_issue"] += 1
        au = t.get("auditor") or {}
        if au.get("verdict") == "revise":
            audits["auditor_revise"] += 1
        ch = t.get("chief") or {}
        acc = ch.get("accept_adjustments") or {}
        for iss in (au.get("issues") or []):
            if iss.get("adjust_kt") is not None:
                audits["adj_proposed"] += 1
                if acc.get(str(iss.get("lead")), True):
                    audits["adj_accepted"] += 1
        if t.get("final_audit"):
            audits["final_violations"] += 1
        # ----- RI reliability -----
        p = d.get("ri24_prob", ch.get("ri24_prob", it.get("ri24_prob")))
        if p is not None and np.isfinite(r.get("vmax_24", np.nan)):
            ri_probs.append(float(p))
            ri_obs.append(1.0 if (r["vmax_24"] - r["vmax"]) >= 30 else 0.0)

    print(f"transcripts: {n} unique cases")

    tdf = pd.DataFrame(trust_rows)
    if len(tdf):
        print("\ntrust factor stats by member (lead 24/48/72 pooled):")
        gstats = tdf.groupby("member").agg(mean_trust=("trust", "mean"),
                                           frac_discount=("trust", lambda s: float((s < 0.9).mean())),
                                           frac_boost=("trust", lambda s: float((s > 1.1).mean())),
                                           n=("trust", "size"))
        print(gstats.round(3).to_string())
        # adjudication quality: within-case rank corr between trust and error
        qual = []
        for (lead,), sub in tdf.groupby(["lead"]):
            c = sub[["trust", "err"]].corr(method="spearman").iloc[0, 1]
            qual.append(dict(lead=lead, spearman_trust_vs_err=round(float(c), 3),
                             n=len(sub)))
        print(pd.DataFrame(qual).to_string(index=False))
        tdf.to_csv(os.path.join(work_dir("results"), f"{args.tag}_trust.csv"), index=False)

    ddf = pd.DataFrame(delta_rows)
    if len(ddf):
        print("\ndelta stats:")
        for l in (24, 48, 72):
            s = ddf[ddf.lead == l]
            if not len(s):
                continue
            corr = s[["delta", "ideal"]].corr().iloc[0, 1]
            print(f"  lead {l}: n={len(s)} mean={s.delta.mean():+.1f} "
                  f"corr(delta, truth-prior)={corr:.3f}")

    print("\naudit stats:", audits,
          f"| revise rate={audits['auditor_revise']/max(audits['n'],1):.1%}",
          f"accept rate={audits['adj_accepted']/max(audits['adj_proposed'],1):.1%}",
          f"final-violation rate={audits['final_violations']/max(audits['n'],1):.1%}")

    if ri_probs:
        from stormdesk.evaluate import ri_prob_calibration, calibrate_ri_prob
        coeffs = ri_prob_calibration()
        p = np.array(ri_probs)
        if coeffs:
            p = np.array([calibrate_ri_prob(x, coeffs) for x in p])
            print(f"\n(applying Platt calibration a={coeffs['a']}, b={coeffs['b']})")
        o = np.array(ri_obs)
        brier = float(np.mean((p - o) ** 2))
        clim = float(np.mean((o.mean() - o) ** 2))
        print(f"\nRI probability: n={len(p)} base={o.mean():.3f} mean_p={p.mean():.3f} "
              f"Brier={brier:.4f} clim={clim:.4f} BSS={1 - brier/clim:.3f}")
        bins = [0, .05, .15, .3, .5, .7, 1.01]
        rel = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() >= 5:
                rel.append(dict(bin=f"{lo:.2f}-{hi:.2f}", n=int(m.sum()),
                                mean_p=round(float(p[m].mean()), 3),
                                obs_freq=round(float(o[m].mean()), 3)))
        rdf = pd.DataFrame(rel)
        print(rdf.to_string(index=False))
        rdf.to_csv(os.path.join(work_dir("results"), f"{args.tag}_ri_reliability.csv"),
                   index=False)


if __name__ == "__main__":
    main()
