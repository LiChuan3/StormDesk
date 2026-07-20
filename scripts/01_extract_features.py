"""Extract environmental + satellite diagnostics for case tables.

Reads the MSETCD 80x80 crops (ERA5/SUPPLEMENT/GRIDSAT) at each case init time.
Writes JSONL: {case_id, diag, sat}. Parallel over processes.
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.diagnostics import compute_diagnostics, compute_satellite, load_crop
from stormdesk.evaluate import case_id


def one(row) -> dict | None:
    era5 = load_crop(row["sid"], row["season"], row["init"], "ERA5")
    if era5 is None:
        return dict(case_id=case_id(row), diag={}, sat=None, missing=True)
    sup = load_crop(row["sid"], row["season"], row["init"], "SUPPLEMENT")
    sat_arr = load_crop(row["sid"], row["season"], row["init"], "GRIDSAT")
    diag = compute_diagnostics(era5, row["lat"], sup, row["vmax"])
    sat = compute_satellite(sat_arr) if sat_arr is not None else None
    return dict(case_id=case_id(row), diag=diag, sat=sat, missing=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    with open(os.path.join(work_dir("cases"), f"{args.split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    rows = [r for _, r in cases.iterrows()]
    out_path = os.path.join(work_dir("features"), f"{args.split}.jsonl")
    n_ok = 0
    with ProcessPoolExecutor(args.workers) as ex, open(out_path, "w") as out:
        for i, res in enumerate(ex.map(one, rows, chunksize=8)):
            if res is None:
                continue
            out.write(json.dumps(res) + "\n")
            n_ok += not res.get("missing")
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(rows)} done ({n_ok} with crops)", flush=True)
    print(f"wrote {out_path}: {len(rows)} rows, {n_ok} with crops")


if __name__ == "__main__":
    main()
