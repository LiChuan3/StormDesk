"""Run AIWP guidance (FengWu cached+ONNX continuation, or Pangu chained) for a
case split. Shardable across GPUs/nodes; writes JSONL per member+shard.

Example:
  python scripts/02_run_aiwp.py --split test --member fengwu --gpu 1 --shard 0 --nshards 4
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.evaluate import case_id
from stormdesk.geo import motion_uv_kmh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--member", choices=["fengwu", "pangu"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-lead", type=int, default=72)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse", action="store_true",
                    help="process the shard from the end (helper worker that "
                         "meets the forward worker in the middle)")
    ap.add_argument("--suffix", default="",
                    help="output-file suffix so helpers do not share a file")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    from stormdesk.guidance.aiwp import FengWuRunner, PanguRunner
    runner = FengWuRunner(0) if args.member == "fengwu" else PanguRunner(0)

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    cases = cases.iloc[args.shard::args.nshards]
    if args.reverse:
        cases = cases.iloc[::-1]
    if args.limit:
        cases = cases.iloc[:args.limit]

    out_path = os.path.join(work_dir("guidance"),
                            f"{args.split}_{args.member}_shard{args.shard}{args.suffix}.jsonl")

    # a case is done if ANY worker file for this member/split has it non-null;
    # rescanned periodically so forward and reverse workers meet cleanly
    import glob as _glob

    def scan_done() -> set:
        s = set()
        for p in _glob.glob(os.path.join(work_dir("guidance"),
                                         f"{args.split}_{args.member}_shard*.jsonl")):
            with open(p) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if d.get("forecast"):  # null rows (errors) are retried
                            s.add(d["case_id"])
                    except Exception:
                        pass
        return s

    done = scan_done()
    print(f"{args.member} shard {args.shard}/{args.nshards}: {len(cases)} cases, "
          f"{len(done)} already done", flush=True)

    t0 = time.time()
    with open(out_path, "a") as out:
        for k, (_, r) in enumerate(cases.iterrows()):
            if k and k % 50 == 0:
                done |= scan_done()
            cid = case_id(r)
            if cid in done:
                continue
            h = r["history"]
            m = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
            try:
                fc = runner.forecast(r["init"], r["lat"], r["lon"],
                                     max_lead=args.max_lead,
                                     motion0=(float(m[0]), float(m[1])))
            except Exception as e:  # noqa: BLE001
                print(f"ERR {cid}: {e}", flush=True)
                fc = None
            out.write(json.dumps(dict(case_id=cid, member=args.member,
                                      forecast=fc)) + "\n")
            out.flush()
            if (k + 1) % 20 == 0:
                dt = (time.time() - t0) / (k + 1)
                print(f"{k+1}/{len(cases)} avg {dt:.1f}s/case", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
