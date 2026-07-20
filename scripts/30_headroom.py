"""Headroom utilization U of each LLM policy, with storm-cluster bootstrap CIs.

  U = (L_static - L_policy) / (L_static - L_learned)

on the analysis-manifest homogeneous sample, where L is the mean loss (track
great-circle error, or Vmax MAE), L_static is the static reference
(hybrid_static), and L_learned is the supervised gate operating in the same
bounded action space (learned_gbt_contract by default; learned_gbt2 reported as
a variant). U = 1 means the policy realizes all of the case-adaptive headroom
the supervised policy proves exists; U = 0 means none; U < 0 means the policy
is worse than the static reference.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.evaluate import case_id, load_forecasts, _truth, _fc
from stormdesk.geo import gc_distance_km

VERIF = [24, 48, 72]


def loss_arrays(cases, fdicts, lead, allowed):
    """aligned per-case losses for every method in fdicts; skip cases where
    any method is missing (they share the manifest, so this is ~none)."""
    names = list(fdicts)
    sids, L = [], {m: {"t": [], "v": []} for m in names}
    for _, r in cases.iterrows():
        la, lo, v = _truth(r, lead)
        if not (np.isfinite(la) and np.isfinite(lo)):
            continue
        cid = case_id(r)
        if allowed is not None and cid not in allowed:
            continue
        es = {m: _fc(fdicts[m], cid, lead) for m in names}
        if any(e is None for e in es.values()):
            continue
        has_v = np.isfinite(v) and all(e.get("vmax") is not None for e in es.values())
        sids.append(r["sid"])
        for m, e in es.items():
            L[m]["t"].append(float(gc_distance_km(la, lo, e["lat"], e["lon"])))
            L[m]["v"].append(abs(e["vmax"] - v) if has_v else np.nan)
    return (np.array(sids),
            {m: np.array(L[m]["t"]) for m in names},
            {m: np.array(L[m]["v"]) for m in names})


def u_ci(sids, l_static, l_policy, l_learned, n_boot, rng):
    ok = np.isfinite(l_static) & np.isfinite(l_policy) & np.isfinite(l_learned)
    s, a, p, g = sids[ok], l_static[ok], l_policy[ok], l_learned[ok]
    if len(a) < 30:
        return None
    den = float(np.mean(a - g))
    num = float(np.mean(a - p))
    if abs(den) < 1e-9:
        return None
    uniq = np.array(sorted(set(s)))
    idx = {x: np.where(s == x)[0] for x in uniq}
    us = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[x] for x in pick])
        d = float(np.mean(a[ii] - g[ii]))
        if abs(d) < 1e-9:
            continue
        us.append(float(np.mean(a[ii] - p[ii])) / d)
    us = np.array(us)
    return dict(n=int(len(a)), U=round(num / den, 3),
                ci90=[round(float(np.percentile(us, 5)), 3),
                      round(float(np.percentile(us, 95)), 3)],
                headroom=round(den, 3), used=round(num, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--case-list", required=True)
    ap.add_argument("--static-ref", default="hybrid_static")
    ap.add_argument("--learned-ref", default="learned_gbt_contract")
    ap.add_argument("--policies", default=(
        "agent_full_qwen14b,agent_full_qwen14b_gateadv,agent_mini_qwen14b,"
        "agent_single_refine_qwen14b,agent_full_qwen72b"))
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    with open(args.case_list) as f:
        mf = json.load(f)
    allowed = {int(l): set(e["case_ids"]) for l, e in mf["leads"].items()}

    fdir = work_dir("forecasts")

    def load(name):
        p = os.path.join(fdir, f"{args.split}_{name}.jsonl")
        return load_forecasts(p) if os.path.exists(p) else None

    fixed = {args.static_ref: load(args.static_ref),
             args.learned_ref: load(args.learned_ref)}
    assert all(fixed.values()), "static/learned reference forecasts missing"
    policies = {}
    for name in args.policies.split(","):
        fc = load(name)
        if fc:
            policies[name] = fc
        else:
            print(f"  (skip {name}: no forecast file)")

    rng = np.random.default_rng(0)
    out = {}
    for lead in VERIF:
        fdicts = dict(fixed)
        fdicts.update(policies)
        sids, T, V = loss_arrays(cases, fdicts, lead, allowed.get(lead))
        for pol in policies:
            for metric, L in (("track", T), ("vmax", V)):
                r = u_ci(sids, L[args.static_ref], L[pol], L[args.learned_ref],
                         args.n_boot, rng)
                if r:
                    out.setdefault(pol, {}).setdefault(metric, {})[str(lead)] = r

    print(f"headroom utilization U vs static={args.static_ref}, "
          f"learned={args.learned_ref}")
    for pol, d in out.items():
        for metric in ("track", "vmax"):
            if metric not in d:
                continue
            s = "  ".join(f"{l}h {e['U']:+.2f} [{e['ci90'][0]:+.2f},{e['ci90'][1]:+.2f}]"
                          for l, e in sorted(d[metric].items(), key=lambda kv: int(kv[0])))
            print(f"  {pol:36s} {metric:5s} {s}")

    res = dict(static_ref=args.static_ref, learned_ref=args.learned_ref,
               manifest=os.path.basename(args.case_list), policies=out)
    with open(os.path.join(work_dir("results"),
                           f"{args.split}_headroom.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote headroom.json")


if __name__ == "__main__":
    main()
