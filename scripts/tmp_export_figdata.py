"""Dump plotting data as JSON so the landscape/case figures can be redrawn
locally in the main-text (Comic Sans MS / Okabe-Ito) style."""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/yuxiaoning/projects/stormdesk")
from stormdesk.config import work_dir
from stormdesk.evaluate import case_id, load_forecasts, evaluate_methods

OUT = "/tmp/figdata"
os.makedirs(OUT, exist_ok=True)


def fnum(x):
    try:
        x = float(x)
        return round(x, 2) if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load_cases(split):
    with open(os.path.join(work_dir("cases"), f"{split}.pkl"), "rb") as f:
        return pickle.load(f)


def load_all_methods(split, names):
    fdir = work_dir("forecasts")
    out = {}
    for n in names:
        p = os.path.join(fdir, f"{split}_{n}.jsonl")
        if os.path.exists(p):
            fc = load_forecasts(p)
            fc = {k: v for k, v in fc.items() if v.get("forecast")}
            if fc:
                out[n] = fc
    missing_aiwp = [n for n in names if n not in out and n in ("pangu", "fengwu")]
    if missing_aiwp:
        from stormdesk.guidance.merge import load_guidance
        g = load_guidance(split, members=missing_aiwp)
        for n in missing_aiwp:
            fc = {cid: dict(case_id=cid, forecast=members[n])
                  for cid, members in g.items() if n in members}
            if fc:
                out[n] = fc
    return out


# ---------------- 1. test-season best tracks for the map ----------------
cases = load_cases("test")
from stormdesk.ibtracs import load_ibtracs
ib = load_ibtracs()
sids = set(cases["sid"])
tracks = ib[ib.SID.isin(sids)]
map_tracks = []
for sid, g in tracks.groupby("SID"):
    g = g.sort_values("ISO_TIME")
    lon = g["LON"].values.astype(float)
    lat = g["LAT"].values.astype(float)
    v = pd.to_numeric(g["VMAX"], errors="coerce").values
    map_tracks.append(dict(sid=str(sid),
                           lon=[fnum(x) for x in lon],
                           lat=[fnum(x) for x in lat],
                           vmax=[fnum(x) for x in v]))
json.dump(dict(n_storms=int(tracks.SID.nunique()), n_cycles=int(len(cases)),
               tracks=map_tracks),
          open(os.path.join(OUT, "map_tracks.json"), "w"))
print("map tracks:", tracks.SID.nunique(), "storms")

# land polygons (Natural Earth 110m via cartopy, exterior rings only)
try:
    import cartopy.io.shapereader as shpreader
    shp = shpreader.natural_earth(resolution="110m", category="physical", name="land")
    polys = []
    for geom in shpreader.Reader(shp).geometries():
        gs = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for p in gs:
            x, y = p.exterior.coords.xy
            polys.append([[round(float(a), 2) for a in x],
                          [round(float(b), 2) for b in y]])
    json.dump(polys, open(os.path.join(OUT, "land_polys.json"), "w"))
    print("land polys:", len(polys))
except Exception as e:
    print("no cartopy land:", repr(e))

# ---------------- 2. calibration-season motivation panels ----------------
ccases = load_cases("calib")
names = ["pangu", "fengwu", "gru", "transformer", "cliper",
         "cons_aiwp", "cons_weighted"]
methods = load_all_methods("calib", names)
tab = evaluate_methods(ccases, methods)
skill = {}
for m in names:
    if m not in methods:
        continue
    skill[m] = {str(l): fnum(tab[(tab.method == m) & (tab.lead == l)]["track_km"].iloc[0])
                for l in (24, 48, 72)}
bias = {}
for m in ["pangu", "fengwu", "gru", "transformer", "cliper"]:
    if m not in methods:
        continue
    errs = []
    for _, r in ccases.iterrows():
        e = methods[m].get(case_id(r))
        if not e:
            continue
        f = e["forecast"].get("48")
        vt = r.get("vmax_48")
        if f and f.get("vmax") is not None and np.isfinite(vt):
            errs.append(round(float(f["vmax"]) - float(vt), 1))
    bias[m] = errs
json.dump(dict(track_skill=skill, bias48=bias),
          open(os.path.join(OUT, "motivation.json"), "w"))
print("motivation done:", {m: len(v) for m, v in bias.items()})

# ---------------- 3. Noru case (2022265N18135_2022092400) ----------------
methods_t = load_all_methods("test", ["pangu", "fengwu", "gru", "transformer",
                                      "cliper", "cons_bc", "agent_full_qwen14b"])
row = None
for _, r in cases.iterrows():
    if case_id(r) == "2022265N18135_2022092400":
        row = r
        break
if row is None:  # fallback: biggest observed 24-h intensification, like fig6
    best = None
    for _, r in cases.iterrows():
        dv = (r.get("vmax_24") or np.nan) - r["vmax"]
        cid = case_id(r)
        if not all(cid in m for m in methods_t.values()):
            continue
        if best is None or (np.isfinite(dv) and dv > best[0]):
            best = (dv, r)
    row = best[1]
cid = case_id(row)
noru = dict(cid=cid, name=str(row["name"]),
            truth=dict(lat=[fnum(row["lat"])] + [fnum(row.get("lat_%d" % l)) for l in (24, 48, 72)],
                       lon=[fnum(row["lon"])] + [fnum(row.get("lon_%d" % l)) for l in (24, 48, 72)],
                       vmax=[fnum(row["vmax"])] + [fnum(row.get("vmax_%d" % l)) for l in (24, 48, 72)]),
            methods={})
for m, fc in methods_t.items():
    e = fc.get(cid)
    if not e:
        continue
    fobj = e["forecast"]
    noru["methods"][m] = {str(l): dict(lat=fnum(fobj.get(str(l), {}).get("lat")),
                                       lon=fnum(fobj.get(str(l), {}).get("lon")),
                                       vmax=fnum(fobj.get(str(l), {}).get("vmax")))
                          for l in (24, 48, 72)}
json.dump(noru, open(os.path.join(OUT, "noru_case.json"), "w"))
print("noru case:", cid, sorted(noru["methods"]))
