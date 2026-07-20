"""Diagnose AIWP conventions:
A) tracker sanity on ERA5 analysis fields (init and +24h truth)
B) my FengWu ONNX step vs the known-good cache (+6h frame)
C) Pangu +24h field: vortex depth near the observed position
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir, CH_MSL, CH_U10, CH_V10, LEVMAJ_TO_VARMAJ, VARMAJ_TO_LEVMAJ
from stormdesk.evaluate import case_id
from stormdesk.geo import motion_uv_kmh
from stormdesk.tracker import track_step
from stormdesk.guidance.aiwp import era5_file, fengwu_cache_file, era5_to_pangu, pangu_to_era5
from stormdesk.config import get_paths


def main():
    with open(os.path.join(work_dir("cases"), "test.pkl"), "rb") as f:
        cases = pickle.load(f)
    r = cases[(cases.basin == "WP") & (cases.vmax >= 90)].iloc[10]
    print("case:", case_id(r), r["name"], "init", r["init"], r["lat"], r["lon"], r["vmax"], "kt")
    t0 = pd.Timestamp(r["init"])

    # A) tracker on analysis
    for dt, la, lo in ((0, r["lat"], r["lon"]), (24, r["lat_24"], r["lon_24"])):
        st = np.load(era5_file(t0 + pd.Timedelta(hours=dt)))
        fix = track_step(st, la, lo, 300.0)
        print(f"A) ERA5 t+{dt}: tracker {fix} | truth ({la},{lo}) v={r['vmax'] if dt==0 else r['vmax_24']}")
        del st

    # B) my onnx step vs cache
    p = get_paths()
    mean = np.load(p.fengwu_mean)[:, None, None].astype(np.float32)
    std = np.load(p.fengwu_std)[:, None, None].astype(np.float32)
    print("mean/std shapes:", mean.shape, std.shape)
    s_prev = np.load(era5_file(t0 - pd.Timedelta(hours=6))).astype(np.float32)
    s_now = np.load(era5_file(t0)).astype(np.float32)
    cache = np.load(fengwu_cache_file(t0))  # frames +6/+12/+18/+24
    import onnxruntime as ort
    opt = ort.SessionOptions()
    opt.enable_cpu_mem_arena = False
    opt.enable_mem_pattern = False
    sess = ort.InferenceSession(p.fengwu_onnx, sess_options=opt,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("onnx inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
    print("onnx outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
    x = np.concatenate([(s_prev[LEVMAJ_TO_VARMAJ] - mean) / std,
                        (s_now[LEVMAJ_TO_VARMAJ] - mean) / std], axis=0)[None]
    t1 = time.time()
    out = sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0]
    print(f"one step: {time.time()-t1:.1f}s, out shape {out.shape}")
    pred6 = (out[0, :69] * std + mean)[VARMAJ_TO_LEVMAJ]
    truth6 = np.load(era5_file(t0 + pd.Timedelta(hours=6))).astype(np.float32)
    for ch, name in ((CH_MSL, "msl"), (CH_U10, "u10")):
        d_cache = np.abs(pred6[ch] - cache[0][ch]).mean()
        d_truth = np.abs(pred6[ch] - truth6[ch]).mean()
        d_cache_truth = np.abs(np.asarray(cache[0][ch]) - truth6[ch]).mean()
        print(f"B) {name}: |mine-cache+6|={d_cache:.2f} |mine-era5+6|={d_truth:.2f} "
              f"|cache+6-era5+6|={d_cache_truth:.2f}")
    del sess, x, out, s_prev
    import gc as _gc
    _gc.collect()

    # C) pangu +24 field vortex depth
    sess2 = ort.InferenceSession(os.path.join(p.pangu_dir, "pangu_weather_24.onnx"),
                                 sess_options=opt,
                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("pangu inputs:", [(i.name, i.shape) for i in sess2.get_inputs()])
    up, sf = era5_to_pangu(s_now)
    up2, sf2 = sess2.run(None, {"input": up, "input_surface": sf})
    f24 = pangu_to_era5(up2, sf2)
    fix = track_step(f24, r["lat_24"], r["lon_24"], 400.0)
    print("C) pangu +24 tracked near truth:", fix)
    cachefix = track_step(cache[3], r["lat_24"], r["lon_24"], 400.0)
    print("C) fengwu cache +24 tracked near truth:", cachefix)
    # global min msl in a 10-deg box around truth for both
    for nm, fld in (("pangu", f24), ("cache_fw", cache[3])):
        from stormdesk.tracker import _window, LAT_AXIS
        ii, jj = _window(r["lat_24"], r["lon_24"], 10.0)
        sub = fld[CH_MSL][np.ix_(ii, jj)]
        print(f"C) {nm}: min msl in 10deg box = {sub.min()/100:.1f} hPa, "
              f"max10mwind = {np.hypot(fld[CH_U10][np.ix_(ii, jj)], fld[CH_V10][np.ix_(ii, jj)]).max()*1.944:.0f} kt")


if __name__ == "__main__":
    main()
