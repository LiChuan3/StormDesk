"""Smoke test: one FengWu + one Pangu guidance case, checked against IBTrACS."""
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.evaluate import case_id
from stormdesk.geo import motion_uv_kmh, gc_distance_km


def show(name, fc, r):
    print(f"--- {name}")
    if fc is None:
        print("  no forecast (missing input)")
        return
    for l in ("24", "48", "72"):
        e = fc.get(l)
        if e is None:
            continue
        tl = (r[f"lat_{l}"], r[f"lon_{l}"], r[f"vmax_{l}"])
        err = float(gc_distance_km(tl[0], tl[1], e["lat"], e["lon"]))
        print(f"  lead {l}h: fc ({e['lat']:.1f},{e['lon']:.1f}) v={e['vmax_kt']:.0f}kt "
              f"mslp={e['mslp_hpa']:.0f} | ob ({tl[0]:.1f},{tl[1]:.1f}) v={tl[2]:.0f}kt "
              f"| track err {err:.0f} km")


def main():
    import onnxruntime as ort
    print("providers:", ort.get_available_providers())
    with open(os.path.join(work_dir("cases"), "test.pkl"), "rb") as f:
        cases = pickle.load(f)
    r = cases[(cases.basin == "WP") & (cases.vmax >= 90)].iloc[10]
    print("case:", case_id(r), r["name"], r["lat"], r["lon"], r["vmax"], "kt")
    h = r["history"]
    m = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
    m = (float(m[0]), float(m[1]))

    from stormdesk.guidance.aiwp import FengWuRunner, PanguRunner
    t0 = time.time()
    fw = FengWuRunner(0).forecast(r["init"], r["lat"], r["lon"], 72, m)
    print(f"fengwu: {time.time()-t0:.1f}s")
    show("fengwu", fw, r)

    t0 = time.time()
    pg = PanguRunner(0).forecast(r["init"], r["lat"], r["lon"], 72, m)
    print(f"pangu: {time.time()-t0:.1f}s")
    show("pangu", pg, r)


if __name__ == "__main__":
    main()
