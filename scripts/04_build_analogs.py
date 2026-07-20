"""Build the historical analog library from 1980-2015 storm timesteps.

Each entry: state+environment features at a timestep and the observed
subsequent evolution (dV and displacement at 24/48/72 h). Requires the
diagnostics of each timestep -> parallel crop reads.
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

from stormdesk.config import work_dir, TRAIN_YEARS
from stormdesk.analogs import AnalogLibrary, entry_features
from stormdesk.diagnostics import compute_diagnostics, load_crop
from stormdesk.geo import motion_uv_kmh
from stormdesk.ibtracs import load_ibtracs


def one(payload):
    sid, name, basin, season, rows, i = payload
    r = rows[i]
    prev = rows[max(i - 2, 0)]
    if prev["vmax"] is None:
        return None
    era5 = load_crop(sid, season, r["time"], "ERA5")
    if era5 is None:
        return None
    sup = load_crop(sid, season, r["time"], "SUPPLEMENT")
    diag = compute_diagnostics(era5, r["lat"], sup, r["vmax"])
    mu, mv = motion_uv_kmh(prev["lat"], prev["lon"], r["lat"], r["lon"],
                           max(6.0 * (i - max(i - 2, 0)), 6.0))
    dv12 = r["vmax"] - prev["vmax"]
    f = entry_features(r["lat"], r["lon"], r["vmax"], dv12, float(mu), float(mv),
                       diag, r["time"])
    if f is None:
        return None
    outcome = {}
    for l in (24, 48, 72):
        j = i + l // 6
        if j < len(rows) and rows[j]["vmax"] is not None:
            outcome[f"dv_{l}"] = round(rows[j]["vmax"] - r["vmax"], 1)
            outcome[f"lat_{l}"] = rows[j]["lat"]
            outcome[f"lon_{l}"] = rows[j]["lon"]
    if "dv_24" not in outcome:
        return None
    return dict(sid=sid, name=name, basin=basin, time=str(r["time"]),
                lat=r["lat"], lon=r["lon"], vmax=r["vmax"], features=f,
                outcome=outcome)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2, help="use every Nth timestep")
    args = ap.parse_args()

    df = load_ibtracs()
    tasks = []
    for sid, g in df.groupby("SID"):
        season = int(g["SEASON"].iloc[0])
        if season < TRAIN_YEARS[0] or season > TRAIN_YEARS[1]:
            continue
        g = g.reset_index(drop=True)
        rows = [dict(time=r.ISO_TIME, lat=float(r.LAT), lon=float(r.LON),
                     vmax=None if pd.isna(r.VMAX) else float(r.VMAX))
                for _, r in g.iterrows()]
        name, basin = str(g["NAME"].iloc[0]), str(g["BASIN"].iloc[0])
        for i in range(0, len(rows), args.stride):
            if rows[i]["vmax"] is not None and rows[i]["vmax"] >= 34:
                tasks.append((sid, name, basin, season, rows, i))
    print(f"{len(tasks)} candidate timesteps")

    records = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, rec in enumerate(ex.map(one, tasks, chunksize=16)):
            if rec is not None:
                records.append(rec)
            if (k + 1) % 2000 == 0:
                print(f"{k+1}/{len(tasks)} scanned, {len(records)} kept", flush=True)
    lib = AnalogLibrary.build(records)
    lib.save()
    print(f"library: {len(records)} entries -> saved")


if __name__ == "__main__":
    main()
