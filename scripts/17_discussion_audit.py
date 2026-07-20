"""Automated factual-consistency audit of the office's written discussions.

NOT a human forecaster evaluation (stated as such in the paper). Two layers:
1. LLM judge (a larger backbone than the office's) receives the briefing, the
   final forecast numbers and the discussion, and itemizes: numeric claims
   that contradict the briefing/forecast, statements inconsistent with the
   issued forecast (e.g. RI wording vs probability), and facts not grounded
   in the briefing.
2. Deterministic numeric cross-check: every number in the discussion is
   searched for in the briefing + forecast (with tolerance); reports the
   fraction of grounded numeric tokens.

Usage:
  python scripts/17_discussion_audit.py --split test --tag agent_full_qwen14b \
      --n 200 --llm-url http://192.168.100.5:8500/v1 --llm-model qwen2.5-72b
"""
import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.agents.llm import LLMClient, ask_json

JUDGE_SYSTEM = """You are a strict meteorological fact-checker. You receive the BRIEFING an
LLM forecast office saw, the FINAL FORECAST it issued (positions, intensities, RI
probability), and the DISCUSSION text it wrote. Audit the discussion:
1. NUMERIC CLAIMS: list every numeric claim (positions, winds, shear, SST, RH, probabilities,
   analog statistics) and whether it matches the briefing/forecast (tolerance: 1 unit, or
   0.2 degrees for coordinates). Count claims and mismatches.
2. FORECAST CONSISTENCY: does the narrative match the issued numbers (stated motion vs track,
   stated intensification vs intensity trend, stated RI risk vs ri24_prob)?
3. UNGROUNDED FACTS: statements of fact that appear in neither the briefing nor the forecast
   (do not count generic meteorological reasoning or hedged judgement).
Respond with ONLY a JSON object:
{"n_numeric_claims": <int>, "n_numeric_mismatches": <int>,
 "mismatches": ["<claim>: <briefing value>", ...],
 "consistent_with_forecast": true|false,
 "inconsistencies": ["..."],
 "n_ungrounded_facts": <int>, "ungrounded": ["..."],
 "verdict": "clean" | "minor_issues" | "major_issues"}"""


def numeric_crosscheck(discussion: str, briefing: str, forecast: dict, ri_prob):
    """Fraction of numeric tokens in the discussion that are grounded in the
    briefing text or the forecast values (|diff| <= 1.0, coords <= 0.25)."""
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", discussion)]
    nums = [x for x in nums if abs(x) > 1e-9]
    ref = [float(x) for x in re.findall(r"-?\d+\.?\d*", briefing)]
    for e in (forecast or {}).values():
        if isinstance(e, dict):
            for v in e.values():
                if isinstance(v, (int, float)):
                    ref.append(float(v))
    if ri_prob is not None:
        ref += [float(ri_prob), round(float(ri_prob) * 100, 0)]
    ref = np.array(sorted(set(ref)))
    if not nums or not len(ref):
        return None
    ok = 0
    for x in nums:
        tol = 0.25 if abs(x) <= 90 and abs(x - round(x, 1)) < 1e-9 and "." in f"{x}" else 1.0
        if np.min(np.abs(ref - x)) <= tol:
            ok += 1
    return ok / len(nums), len(nums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="agent_full_qwen14b")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--llm-url", required=True)
    ap.add_argument("--llm-model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = []
    with open(os.path.join(work_dir("transcripts"), f"{args.split}_{args.tag}.jsonl")) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d.get("discussion"):
                    recs.append(d)
    fcs = {}
    with open(os.path.join(work_dir("forecasts"), f"{args.split}_{args.tag}.jsonl")) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                fcs[d["case_id"]] = d
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(recs), size=min(args.n, len(recs)), replace=False)
    sample = [recs[i] for i in idx]
    print(f"{len(sample)} discussions sampled from {len(recs)}")

    llm = LLMClient(args.llm_url, args.llm_model, temperature=0.0, max_tokens=1200)
    lock = threading.Lock()
    results = []

    def one(d):
        cid = d["case_id"]
        t = d.get("transcript") or {}
        fc = fcs.get(cid) or {}
        user = ("BRIEFING:\n" + (t.get("briefing") or "")
                + "\n\nFINAL FORECAST: " + json.dumps(fc.get("forecast"))
                + "\nri24_prob: " + str(fc.get("ri24_prob"))
                + "\n\nDISCUSSION:\n" + d["discussion"])
        js = ask_json(llm, JUDGE_SYSTEM, user)
        cc = numeric_crosscheck(d["discussion"], t.get("briefing") or "",
                                fc.get("forecast"), fc.get("ri24_prob"))
        return dict(case_id=cid, judge=js,
                    crosscheck_frac=None if cc is None else round(cc[0], 3),
                    crosscheck_n=None if cc is None else cc[1])

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one, d) for d in sample]
        for k, fut in enumerate(as_completed(futs)):
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                r = dict(error=str(e)[:200])
            with lock:
                results.append(r)
            if (k + 1) % 25 == 0:
                print(f"{k+1}/{len(sample)}")

    out_path = os.path.join(work_dir("results"), f"{args.split}_discussion_audit.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in results if "judge" in r]
    ncl = np.array([r["judge"].get("n_numeric_claims") or 0 for r in ok], float)
    nmm = np.array([r["judge"].get("n_numeric_mismatches") or 0 for r in ok], float)
    nug = np.array([r["judge"].get("n_ungrounded_facts") or 0 for r in ok], float)
    cons = np.array([bool(r["judge"].get("consistent_with_forecast")) for r in ok])
    verd = [str(r["judge"].get("verdict")) for r in ok]
    cc = np.array([r["crosscheck_frac"] for r in ok if r.get("crosscheck_frac") is not None])
    print(f"\naudited {len(ok)}/{len(results)} (judge parse failures: {len(results)-len(ok)})")
    print(f"numeric claims/discussion: mean {ncl.mean():.1f}; "
          f"claim-level mismatch rate: {nmm.sum()/max(ncl.sum(),1):.1%}")
    print(f"discussions with zero numeric mismatches: {(nmm == 0).mean():.1%}")
    print(f"consistent with issued forecast: {cons.mean():.1%}")
    print(f"ungrounded facts/discussion: mean {nug.mean():.2f}; "
          f"zero-ungrounded: {(nug == 0).mean():.1%}")
    from collections import Counter
    print("verdicts:", dict(Counter(verd)))
    if len(cc):
        print(f"deterministic numeric grounding: mean {cc.mean():.1%} of numbers "
              f"found in briefing/forecast")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
