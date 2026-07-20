"""Fit member skill profiles + intensity bias maps on the calibration split."""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.baselines import Calibration
from stormdesk.evaluate import case_id
from stormdesk.geo import gc_distance_km
from stormdesk.guidance.merge import load_guidance

VERIF = [24, 48, 72]


def main():
    with open(os.path.join(work_dir("cases"), "calib.pkl"), "rb") as f:
        cases = pickle.load(f)
    guidance = load_guidance("calib")
    # DL/cliper forecasts on calib written by 05_run_baselines --split calib
    for m in ("gru", "transformer", "cliper", "cons_aiwp"):
        path = os.path.join(work_dir("forecasts"), f"calib_{m}.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    guidance.setdefault(d["case_id"], {})[d["method"]] = d["forecast"]

    records = []
    for _, r in cases.iterrows():
        cid = case_id(r)
        g = guidance.get(cid)
        if not g:
            continue
        for lead in VERIF:
            la, lo, vt = r.get(f"lat_{lead}"), r.get(f"lon_{lead}"), r.get(f"vmax_{lead}")
            if not np.isfinite(la):
                continue
            for m, fc in g.items():
                e = fc.get(str(lead))
                if e is None:
                    continue
                vfc = e.get("vmax", e.get("vmax_kt"))
                records.append(dict(
                    member=m, lead=lead,
                    track_err_km=float(gc_distance_km(la, lo, e["lat"], e["lon"])),
                    vmax_fc=float(vfc) if vfc is not None else np.nan,
                    vmax_ob=vt if np.isfinite(vt) else np.nan))
    calib = Calibration.fit(records)
    calib.save()
    print(f"calibration fit on {len(records)} member-lead records")
    for m, t in calib.table.items():
        row = {l: (round(t.get(str(l), {}).get('track_rmse', float('nan')))) for l in VERIF}
        print(m, row, "v_bias24:", round(t.get("24", {}).get("v_bias", float("nan")), 1))


if __name__ == "__main__":
    main()
