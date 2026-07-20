"""Run the StormDesk agent office over a case split (threaded LLM calls).

Example:
  python scripts/06_run_agent.py --split test --mode full --workers 12 \
      --llm-url http://192.168.100.5:8500/v1 --llm-model qwen2.5-14b --tag qwen14b
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.agents.llm import LLMClient
from stormdesk.agents.office import run_office
from stormdesk.analogs import AnalogLibrary, entry_features, summarize_analogs
from stormdesk.baselines import Calibration
from stormdesk.diagnostics import load_crop
from stormdesk.evaluate import case_id
from stormdesk.geo import motion_uv_kmh
from stormdesk.guidance.merge import load_guidance, load_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="full",
                    choices=["full", "no_analogs", "no_auditor", "no_diagnostics",
                             "single", "single_refine", "free", "free_schema",
                             "free_delta", "mini", "featmatch", "featmatch_fs"])
    ap.add_argument("--anonymize", action="store_true",
                    help="strip storm name and shift the year in the briefing "
                         "(LLM memorization / contamination control)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--llm-url", default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--members", default=None, help="comma list to restrict guidance")
    ap.add_argument("--gate-advice", default=None,
                    help="path to gate_weights JSON; inject the learned combiner's "
                         "suggested member weights into the briefing (R1/Q3)")
    ap.add_argument("--perturb", default=None,
                    help="degraded-init test (R7/Q10): 'center=0.5,vmax=7.5' shifts "
                         "the initial center by 0.5 deg (random bearing per case) and "
                         "adds 7.5 kt to the initial vmax before running")
    ap.add_argument("--fewshot", default=None,
                    help="path to fewshot_track.json (worked examples for the "
                         "track specialist); default models/fewshot_track.json "
                         "when mode is featmatch_fs")
    ap.add_argument("--prior-file", default=None,
                    help="forecast JSONL whose vmax replaces the bias-corrected "
                         "consensus as the office's intensity prior (stronger-"
                         "prior test); cases without an entry are skipped")
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    guidance_all = load_guidance(args.split)
    feats = load_features(args.split)
    calib = Calibration.load()
    lib = AnalogLibrary.load()
    gate_advice_all = {}
    if args.gate_advice:
        with open(args.gate_advice) as f:
            gate_advice_all = json.load(f)
    perturb = {}
    if args.perturb:
        for kv in args.perturb.split(","):
            k, v = kv.split("=")
            perturb[k.strip()] = float(v)
    fewshot = None
    fs_path = args.fewshot or (os.path.join(work_dir("models"), "fewshot_track.json")
                               if args.mode == "featmatch_fs" else None)
    if fs_path:
        with open(fs_path) as f:
            fewshot = json.load(f)
        print(f"few-shot examples: {len(fewshot)} from {fs_path}")
    prior_all = None
    if args.prior_file:
        from stormdesk.evaluate import load_forecasts
        pf = load_forecasts(args.prior_file)
        prior_all = {}
        for cid, d in pf.items():
            fc = d.get("forecast") or {}
            ent = {l: fc[l]["vmax"] for l in ("24", "48", "72")
                   if fc.get(l) and fc[l].get("vmax") is not None}
            if ent:
                prior_all[cid] = ent
        print(f"prior override for {len(prior_all)} cases from {args.prior_file}")
    print(f"{len(cases)} cases, guidance for {len(guidance_all)}, features {len(feats)}")

    name = f"agent_{args.mode}" + (f"_{args.tag}" if args.tag else "")
    out_path = os.path.join(work_dir("forecasts"), f"{args.split}_{name}.jsonl")
    tr_path = os.path.join(work_dir("transcripts"), f"{args.split}_{name}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for l in f:
                if l.strip():
                    d = json.loads(l)
                    if d.get("forecast"):
                        done.add(d["case_id"])
    rows = [r for _, r in cases.iterrows() if case_id(r) not in done]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} to run (skipping {len(done)} done)")

    llm = LLMClient(args.llm_url, args.llm_model, temperature=args.temperature)
    lock = threading.Lock()
    out_f = open(out_path, "a")
    tr_f = open(tr_path, "a")
    n_ok = [0]
    t0 = time.time()

    members = args.members.split(",") if args.members else None

    def one(r):
        cid = case_id(r)
        g = guidance_all.get(cid)
        if not g:
            return cid, None, "no guidance"
        if members:
            g = {m: v for m, v in g.items() if m in members}
        ft = feats.get(cid, {})
        diag = ft.get("diag") or {}
        sat = ft.get("sat")
        case = dict(r)
        # degraded-init test (R7): shift the office's perceived initial center
        # and intensity; truth used for scoring is unchanged (it stays in r).
        if perturb:
            from stormdesk.geo import destination
            seed = int(hashlib.md5(cid.encode()).hexdigest()[:8], 16)
            brg = (seed % 360)
            km = perturb.get("center", 0.0) * 111.19
            if km:
                la2, lo2 = destination(case["lat"], case["lon"], brg, km)
                case["lat"], case["lon"] = float(la2), float(lo2)
            if perturb.get("vmax"):
                case["vmax"] = float(case["vmax"]) + perturb["vmax"]
        # analogs (queried from the office's perceived state)
        analogs, summary = [], {}
        if args.mode != "no_analogs" and diag:
            h = r["history"]
            mu, mv = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
            dv12 = (case["vmax"] - h[-3]["vmax"]) if h[-3]["vmax"] is not None else 0.0
            f = entry_features(case["lat"], case["lon"], case["vmax"], dv12,
                               float(mu), float(mv), diag, r["init"])
            if f is not None:
                analogs = lib.query(f, case["lat"], r["sid"], k=12)
                summary = summarize_analogs(analogs, case["vmax"])
        sst_crop = None
        sup = load_crop(r["sid"], r["season"], r["init"], "SUPPLEMENT")
        if sup is not None:
            sst_crop = sup[0]
        prior_override = None
        if prior_all is not None:
            prior_override = prior_all.get(cid)
            if not prior_override:
                return cid, None, "no prior override"
        res = run_office(case, diag, sat, g, calib, analogs, summary, llm,
                         mode=args.mode, sst_crop=sst_crop,
                         anonymize=args.anonymize,
                         gate_advice=gate_advice_all.get(cid),
                         prior_override=prior_override, fewshot=fewshot)
        return cid, res, None

    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, r): case_id(r) for r in rows}
        for k, fut in enumerate(as_completed(futs)):
            cid = futs[fut]
            try:
                cid, res, err = fut.result()
            except Exception as e:  # noqa: BLE001
                res, err = None, str(e)[:300]
            with lock:
                if res is not None:
                    out_f.write(json.dumps(dict(
                        case_id=cid, method=name, forecast=res["final"],
                        ri24_prob=res.get("ri24_prob"),
                        confidence=res.get("confidence"))) + "\n")
                    tr_f.write(json.dumps(dict(
                        case_id=cid, discussion=res.get("discussion"),
                        transcript=res.get("transcript"))) + "\n")
                    n_ok[0] += 1
                else:
                    out_f.write(json.dumps(dict(case_id=cid, method=name,
                                                forecast=None, error=err)) + "\n")
                out_f.flush()
                tr_f.flush()
            if (k + 1) % 20 == 0:
                dt = time.time() - t0
                print(f"{k+1}/{len(rows)} ok={n_ok[0]} "
                      f"({dt/(k+1):.1f}s/case; llm calls {llm.n_calls}, "
                      f"tok {llm.n_prompt_tokens}+{llm.n_completion_tokens})", flush=True)
    out_f.close()
    tr_f.close()
    print(f"done: {n_ok[0]} forecasts -> {out_path}")


if __name__ == "__main__":
    main()
