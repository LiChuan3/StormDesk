"""Verification tables: homogeneous comparison, skill, RI scores, significance."""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.evaluate import (evaluate_methods, load_forecasts, ri_scores,
                                skill_vs, paired_test, bootstrap_table, holm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods", default=None,
                    help="comma list; default = all forecast files for the split")
    ap.add_argument("--baseline", default="cliper")
    ap.add_argument("--sig-against", default=None,
                    help="comma list of reference methods for paired tests")
    ap.add_argument("--sig-targets", default=None,
                    help="comma list of methods to test (default: agent_full*, "
                         "hybrid_static)")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="storm-level bootstrap resamples (0 = off)")
    ap.add_argument("--bootstrap-ref", default=None,
                    help="reference method for bootstrap paired differences")
    ap.add_argument("--non-homogeneous", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest-out", default=None,
                    help="write the realized homogeneous sample (per-lead case "
                         "ids, storm counts, md5 hash, method availability) as "
                         "the analysis manifest JSON")
    ap.add_argument("--case-list", default=None,
                    help="path to a manifest JSON; restrict every computation "
                         "(metrics, paired tests, bootstrap) to its per-lead "
                         "case sets")
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)

    fdir = work_dir("forecasts")
    if args.methods:
        names = args.methods.split(",")
    else:
        names = sorted({os.path.basename(p)[len(args.split) + 1:-6]
                        for p in glob.glob(os.path.join(fdir, f"{args.split}_*.jsonl"))})
    methods = {}
    for n in names:
        p = os.path.join(fdir, f"{args.split}_{n}.jsonl")
        if os.path.exists(p):
            fc = load_forecasts(p)
            fc = {k: v for k, v in fc.items() if v.get("forecast")}
            if fc:
                methods[n] = fc
    # AIWP members live as shard files in the guidance dir
    from stormdesk.guidance.merge import load_guidance
    missing_aiwp = [n for n in names if n not in methods and n in ("pangu", "fengwu")]
    if missing_aiwp:
        g = load_guidance(args.split, members=missing_aiwp)
        for n in missing_aiwp:
            fc = {cid: dict(case_id=cid, forecast=members[n])
                  for cid, members in g.items() if n in members}
            if fc:
                methods[n] = fc
    print("methods:", {m: len(v) for m, v in methods.items()})

    case_filter = None
    if args.case_list:
        with open(args.case_list) as f:
            mf_in = json.load(f)
        case_filter = {int(l): set(e["case_ids"])
                       for l, e in mf_in["leads"].items()}
        print("case-list restriction:", {l: len(s) for l, s in case_filter.items()})

    sample_out = {}
    table = evaluate_methods(cases, methods, homogeneous=not args.non_homogeneous,
                             case_filter=case_filter, sample_out=sample_out)

    if args.manifest_out:
        import hashlib
        mf = {"split": args.split, "methods": sorted(methods),
              "availability": {m: len(v) for m, v in methods.items()},
              "leads": {}}
        for lead, pairs in sample_out.items():
            cids = sorted(c for c, _ in pairs)
            mf["leads"][str(lead)] = dict(
                n=len(cids), n_storms=len({s for _, s in pairs}),
                md5=hashlib.md5(",".join(cids).encode()).hexdigest(),
                case_ids=cids)
        with open(args.manifest_out, "w") as f:
            json.dump(mf, f, indent=1)
        print("manifest:", {l: (e["n"], e["n_storms"], e["md5"][:8])
                            for l, e in mf["leads"].items()},
              "->", args.manifest_out)
    table = skill_vs(args.baseline, table) if args.baseline in methods else table
    pd.set_option("display.width", 200)
    for lead in (24, 48, 72):
        sub = table[table.lead == lead].sort_values("track_km")
        print(f"\n=== lead {lead} h (n={sub.n.max()}) ===")
        cols = ["method", "n", "track_km", "vmax_mae_kt", "vmax_bias_kt"]
        if "track_skill_pct" in sub:
            cols += ["track_skill_pct", "vmax_skill_pct"]
        print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    ri = ri_scores(cases, methods)
    print("\n=== RI (>= +30 kt / 24 h), deterministic dV ===")
    print(ri.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if any(m.startswith("agent") for m in methods):
        rip = ri_scores(cases, {m: v for m, v in methods.items() if m.startswith("agent")},
                        use_prob=True)
        print("\n=== RI via explicit probability (p >= 0.5), agents ===")
        print(rip.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.sig_against:
        refs = [m for m in args.sig_against.split(",") if m in methods]
        if args.sig_targets:
            tgt = [m for m in args.sig_targets.split(",") if m in methods]
        else:
            tgt = [m for m in methods
                   if m.startswith("agent_full") or m == "hybrid_static"]
        recs = []
        for t in tgt:
            for ref in refs:
                if ref == t:
                    continue
                for lead in (24, 48, 72):
                    for metric in ("track", "vmax"):
                        r = paired_test(cases, methods[t], methods[ref], lead, metric,
                                        allowed=case_filter.get(lead) if case_filter else None)
                        recs.append(dict(target=t, ref=ref, lead=lead, metric=metric,
                                         n=r["n"], mean_diff=r.get("mean_diff", np.nan),
                                         p=r.get("p", np.nan)))
        # Holm adjustment per target method (family = its 6*len(refs) tests)
        sig = pd.DataFrame(recs)
        sig["p_holm"] = np.nan
        for t in sig.target.unique():
            sel = sig.target == t
            ps = sig.loc[sel, "p"].fillna(1.0).tolist()
            sig.loc[sel, "p_holm"] = holm(ps)
        print("\n=== paired tests (serial-correlation-adjusted t; Holm per target) ===")
        print(sig.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        sig.to_csv(os.path.join(work_dir("results"),
                                f"{args.split}_significance.csv"), index=False)

    if args.bootstrap:
        print(f"\nstorm-level bootstrap ({args.bootstrap} resamples) ...")
        ci = bootstrap_table(cases, methods, n_boot=args.bootstrap,
                             ref=args.bootstrap_ref, case_filter=case_filter)
        print(ci.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        ci.to_csv(os.path.join(work_dir("results"),
                               f"{args.split}_metrics_ci.csv"), index=False)

    out = args.out or os.path.join(work_dir("results"), f"{args.split}_metrics.csv")
    table.to_csv(out, index=False)
    ri.to_csv(out.replace(".csv", "_ri.csv"), index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
