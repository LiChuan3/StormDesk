"""Publication figures for the StormDesk paper.

Usage: python scripts/07_figures.py <fig1a|fig1bc|fig3|fig4|fig5|fig6> [--out DIR]
Outputs PDF (vector) into <work>/figures by default.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stormdesk.config import work_dir
from stormdesk.evaluate import case_id, load_forecasts, evaluate_methods, ri_scores
from stormdesk.geo import gc_distance_km

# ---- fixed identity palette (validated categorical set, light mode) ----
C = dict(
    pangu="#2a78d6", fengwu="#1baf7a", gru="#eda100", transformer="#4a3aa7",
    cliper="#898781", persistence="#b8b6ae",
    cons_equal="#9fc7a4", cons_weighted="#66a76c", cons_bc="#008300",
    cons_aiwp="#2f7d3b",
    agent="#e34948", agent2="#e87ba4", accent="#eb6834",
)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

LABELS = dict(
    persistence="Persistence", cliper="CLIPER5-class", gru="GRU", transformer="Transformer",
    pangu="Pangu-Weather", fengwu="FengWu",
    cons_equal="Consensus (equal)", cons_weighted="Consensus (weighted)",
    cons_bc="Consensus (bias-corr.)", cons_aiwp="Consensus (AIWP)",
)


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 9.5,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "figure.dpi": 150, "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def load_cases(split="test"):
    with open(os.path.join(work_dir("cases"), f"{split}.pkl"), "rb") as f:
        return pickle.load(f)


def method_files(split="test"):
    fdir = work_dir("figures")
    return fdir


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


# ---------------------------------------------------------------- fig 1a map
def fig1a(out):
    cases = load_cases("test")
    df = pd.concat([pd.DataFrame(dict(sid=[r["sid"]], lat=[r["lat"]], lon=[r["lon"]],
                                      vmax=[r["vmax"]])) for _, r in cases.iterrows()])
    # full storm tracks from ibtracs for test seasons
    from stormdesk.ibtracs import load_ibtracs
    ib = load_ibtracs()
    sids = set(cases["sid"])
    tracks = ib[ib.SID.isin(sids)]

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree(central_longitude=180)
        fig = plt.figure(figsize=(7.2, 3.2))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_global()
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f0efec",
                       edgecolor="#c3c2b7", linewidth=0.3, zorder=0)
        ax.set_extent([-180, 180, -50, 55], crs=ccrs.PlateCarree(central_longitude=180))
        tf = ccrs.PlateCarree()
    except Exception:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.set_xlim(0, 360); ax.set_ylim(-50, 55)
        tf = None

    cmap = plt.get_cmap("YlOrRd")
    norm = matplotlib.colors.Normalize(vmin=20, vmax=145)
    for sid, g in tracks.groupby("SID"):
        g = g.sort_values("ISO_TIME")
        lon = g["LON"].values.astype(float)
        lon = np.where(lon < 0, lon + 360, lon)
        lat = g["LAT"].values.astype(float)
        v = pd.to_numeric(g["VMAX"], errors="coerce").values
        # break at dateline jumps
        brk = np.where(np.abs(np.diff(lon)) > 180)[0]
        segs = np.split(np.arange(len(lon)), brk + 1)
        for s in segs:
            if len(s) < 2:
                continue
            kwargs = dict(transform=tf) if tf else {}
            pts = ax.scatter(lon[s] if tf else lon[s], lat[s], c=v[s], cmap=cmap,
                             norm=norm, s=1.2, linewidths=0, zorder=2, **kwargs)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      shrink=0.75, pad=0.02, aspect=28)
    cb.set_label("Best-track intensity (kt)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    n_storms = tracks.SID.nunique()
    ax.set_title(f"2021–2022 test seasons: {n_storms} named storms, "
                 f"{len(cases)} forecast cycles", fontsize=9)
    fig.savefig(os.path.join(out, "fig1a_map.pdf"))
    print("fig1a done:", n_storms, "storms")


# ------------------------------------------------------- fig 1b/c motivation
def fig1bc(out):
    """b: heterogeneous member skill + static consensus (calib); c: intensity
    bias distributions (calib)."""
    cases = load_cases("calib")
    names = ["pangu", "fengwu", "gru", "transformer", "cliper",
             "cons_aiwp", "cons_weighted"]
    methods = load_all_methods("calib", names)
    tab = evaluate_methods(cases, methods)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    order = [n for n in ["cliper", "gru", "transformer", "pangu", "fengwu",
                         "cons_aiwp", "cons_weighted"] if n in methods]
    x = np.arange(len(order))
    for i, lead in enumerate([24, 48, 72]):
        vals = [tab[(tab.method == m) & (tab.lead == lead)]["track_km"].iloc[0] for m in order]
        ax.bar(x + (i - 1) * 0.27, vals, width=0.25,
               color=["#9ec5f4", "#5598e7", "#1c5cab"][i], label=f"{lead} h")
    ax.set_xticks(x)
    labs_b = {"cons_aiwp": "Cons.\n(AIWP)", "cons_weighted": "Cons.\n(wtd)",
              "cliper": "CLIPER"}
    ax.set_xticklabels([labs_b.get(m) or LABELS.get(m, m) for m in order],
                       fontsize=6.5, rotation=24, ha="right")
    ax.set_ylabel("Track error (km)")
    ax.set_title("b  Heterogeneous guidance; static consensus", loc="left", fontsize=8.5)
    ax.legend(ncol=1, loc="upper right", fontsize=7)

    # c: intensity bias distributions at 48 h
    ax = axes[1]
    data, labs, cols = [], [], []
    for m in ["pangu", "fengwu", "gru", "transformer", "cliper"]:
        if m not in methods:
            continue
        errs = []
        for _, r in cases.iterrows():
            e = methods[m].get(case_id(r))
            if not e:
                continue
            f = e["forecast"].get("48")
            vt = r.get("vmax_48")
            if f and f.get("vmax") is not None and np.isfinite(vt):
                errs.append(f["vmax"] - vt)
        data.append(errs)
        labs.append(LABELS[m])
        cols.append(C[m])
    bp = ax.violinplot(data, showmedians=True, widths=0.75)
    for body, c in zip(bp["bodies"], cols):
        body.set_facecolor(c); body.set_alpha(0.55)
    for k in ("cmedians", "cmins", "cmaxes", "cbars"):
        bp[k].set_color(MUTED); bp[k].set_linewidth(0.8)
    ax.axhline(0, color=INK, lw=0.8, ls="--")
    ax.set_xticks(np.arange(1, len(labs) + 1))
    ax.set_xticklabels(labs, fontsize=7, rotation=15)
    ax.set_ylabel("48-h intensity bias (kt)")
    ax.set_title("c  AIWP guidance is systematically too weak", loc="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1bc_motivation.pdf"))
    print("fig1bc done")


# ------------------------------------------------------------- fig 3 curves
def fig3(out, agent="agent_full_qwen14b"):
    cases = load_cases("test")
    names = ["persistence", "cliper", "gru", "transformer", "pangu", "fengwu",
             "cons_equal", "cons_weighted", "cons_bc", "cons_aiwp",
             "learned_gbt", agent]
    methods = load_all_methods("test", names)
    tab = evaluate_methods(cases, methods)
    tab.to_csv(os.path.join(out, "fig3_table.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    show = [n for n in ["cliper", "gru", "pangu", "fengwu",
                        "cons_aiwp", "cons_bc", "learned_gbt", agent]
            if n in methods]
    ls = {"cons_bc": "--", "cons_aiwp": "-.", "cliper": ":", "gru": ":"}
    labs3 = {"learned_gbt": "Learned combiner (GBT)"}
    for m in show:
        sub = tab[tab.method == m].sort_values("lead")
        is_agent = m.startswith("agent")
        kw = dict(color="#2F7D3B" if m == "learned_gbt" else C.get(m, C["agent"]),
                  lw=2.4 if is_agent else (2.0 if m == "learned_gbt" else 1.4),
                  marker="o", ms=4.5 if is_agent else 3, ls=ls.get(m, "-"),
                  label="StormDesk (ours)" if is_agent
                  else labs3.get(m) or LABELS.get(m, m),
                  zorder=5 if is_agent else (4 if m == "learned_gbt" else 3))
        axes[0].plot(sub.lead, sub.track_km, **kw)
        axes[1].plot(sub.lead, sub.vmax_mae_kt, **kw)
    axes[0].set_xlabel("Forecast lead (h)"); axes[0].set_ylabel("Track error (km)")
    axes[0].set_xticks([24, 48, 72])
    axes[0].set_title("a  Track", loc="left")
    axes[1].set_xlabel("Forecast lead (h)"); axes[1].set_ylabel("Intensity MAE (kt)")
    axes[1].set_xticks([24, 48, 72])
    axes[1].set_title("b  Intensity", loc="left")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_main.pdf"))
    print("fig3 done")


# ------------------------------------------------------------------ fig 4 RI
def fig4(out, agent="agent_full_qwen14b"):
    """a: calibrated reliability diagram; b: POD/FAR/CSI across office variants."""
    import json as _json
    cases = load_cases("test")
    variants = [agent, "agent_single_qwen14b", "agent_no_analogs_qwen14b",
                "agent_full_qwen7b"]
    methods = load_all_methods("test", variants)
    ri = ri_scores(cases, methods, use_prob=True)
    ri.to_csv(os.path.join(out, "fig4_ri.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                             gridspec_kw=dict(width_ratios=[1, 1.25]))
    # a: reliability of the calibrated office probability
    ax = axes[0]
    rel_path = os.path.join(work_dir("results"), f"{agent}_ri_reliability.csv")
    if os.path.exists(rel_path):
        rel = pd.read_csv(rel_path)
        ax.plot([0, 0.5], [0, 0.5], color=MUTED, lw=0.8, ls="--")
        ax.plot(rel.mean_p, rel.obs_freq, color=C["agent"], marker="o", ms=4, lw=1.6)
        for _, rr in rel.iterrows():
            ax.annotate(f"n={rr['n']}", (rr.mean_p, rr.obs_freq),
                        textcoords="offset points", xytext=(4, -9), fontsize=6, color=MUTED)
        ax.set_xlabel("Forecast RI probability (calibrated)")
        ax.set_ylabel("Observed RI frequency")
        ax.set_xlim(0, 0.45); ax.set_ylim(0, 0.45)
    ax.set_title("a  RI probability reliability", loc="left", fontsize=9)

    # b: categorical skill across office variants
    ax = axes[1]
    labs = {agent: "StormDesk\n(14B)", "agent_single_qwen14b": "single\nagent",
            "agent_no_analogs_qwen14b": "no\nanalogs", "agent_full_qwen7b": "7B"}
    show = [m for m in variants if m in set(ri.method)]
    x = np.arange(len(show))
    for i, met in enumerate(["pod", "far", "csi"]):
        vals = [float(ri[ri.method == m][met].iloc[0]) for m in show]
        bars = ax.bar(x + (i - 1) * 0.27, vals, width=0.25,
                      color=["#2f7d3b", "#e87ba4", "#2a78d6"][i], label=met.upper())
        if met == "csi":
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labs.get(m, m) for m in show], fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title("b  Categorical RI skill (all baselines: POD = 0)",
                 loc="left", fontsize=9)
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_ri.pdf"))
    print("fig4 done")


# ------------------------------------------------------------ fig 5 ablation
def fig5(out, tags=("qwen14b",)):
    """Component ablations + backbone scale: track@72, |vmax bias|@72, RI CSI."""
    cases = load_cases("test")
    variants = ["agent_full_qwen14b", "agent_no_analogs_qwen14b",
                "agent_no_auditor_qwen14b", "agent_no_diagnostics_qwen14b",
                "agent_single_qwen14b", "agent_single_refine_qwen14b",
                "agent_mini_qwen14b", "agent_full_qwen14b_anon",
                "agent_full_qwen7b", "agent_full_qwen72b"]
    methods = load_all_methods("test", variants + ["cons_bc"])
    tab = evaluate_methods(cases, methods)
    tab.to_csv(os.path.join(out, "fig5_ablation.csv"), index=False)
    ri = ri_scores(cases, {m: v for m, v in methods.items() if m.startswith("agent")},
                   use_prob=True)

    labs = {"agent_full_qwen14b": "full (14B)", "agent_no_analogs_qwen14b": "– analogs",
            "agent_no_auditor_qwen14b": "– auditor",
            "agent_no_diagnostics_qwen14b": "– diagnostics",
            "agent_single_qwen14b": "single agent",
            "agent_single_refine_qwen14b": "single, 5-call refine",
            "agent_mini_qwen14b": "mini office (3 calls)",
            "agent_full_qwen14b_anon": "full, anonymized",
            "agent_free_qwen14b": "free generation",
            "agent_full_qwen7b": "full (7B)", "agent_full_qwen72b": "full (72B)",
            "cons_bc": "consensus (bc)"}
    show = [m for m in variants if m in methods] + ["cons_bc"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    specs = [("track_km", 72, "Track error @72 h (km)", None),
             ("vmax_bias_kt", 72, "Intensity bias @72 h (kt)", None)]
    for ax, (col, lead, title, _) in zip(axes[:2], specs):
        vals = []
        for m in show:
            s = tab[(tab.method == m) & (tab.lead == lead)]
            vals.append(float(s[col].iloc[0]) if len(s) else np.nan)
        cols = [C["agent"] if m == "agent_full_qwen14b"
                else (C["cons_bc"] if m == "cons_bc" else "#9ec5f4") for m in show]
        y = np.arange(len(show))
        ax.barh(y, vals, color=cols, height=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels([labs[m] for m in show], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=8)
        if col == "vmax_bias_kt":
            ax.axvline(0, color=INK, lw=0.8)
    ax = axes[2]
    agents = [m for m in show if m.startswith("agent")]
    vals = [float(ri[ri.method == m]["csi"].iloc[0]) if len(ri[ri.method == m]) else np.nan
            for m in agents]
    y = np.arange(len(agents))
    ax.barh(y, vals, color=[C["agent"] if m == "agent_full_qwen14b" else "#9ec5f4"
                            for m in agents], height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([labs[m] for m in agents], fontsize=7)
    ax.invert_yaxis()
    ax.set_title("RI CSI (baselines: 0)", loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_ablation.pdf"))
    print("fig5 done")


# ---------------------------------------------------------- fig 6 case study
def fig6(out, sid=None, agent="agent_full_qwen14b"):
    """Guidance spaghetti + intensity panel for one storm; pick an RI case."""
    cases = load_cases("test")
    methods = load_all_methods("test", ["pangu", "fengwu", "gru", "transformer",
                                        "cliper", "cons_bc", agent])
    # choose case: biggest observed 24h intensification with all methods present
    best = None
    for _, r in cases.iterrows():
        if sid and r["sid"] != sid:
            continue
        dv = (r.get("vmax_24") or np.nan) - r["vmax"]
        cid = case_id(r)
        if not all(cid in m for m in methods.values()):
            continue
        if best is None or (np.isfinite(dv) and dv > best[0]):
            best = (dv, r)
    dv, r = best
    cid = case_id(r)
    print("case study:", cid, r["name"], f"dv24={dv:+.0f}kt")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    # truth track
    tl_lat = [r["lat"]] + [r.get(f"lat_{l}") for l in (24, 48, 72)]
    tl_lon = [r["lon"]] + [r.get(f"lon_{l}") for l in (24, 48, 72)]
    ax.plot(tl_lon, tl_lat, color=INK, lw=2, marker="s", ms=4, label="Best track", zorder=6)
    for m, fc in methods.items():
        e = fc.get(cid)
        if not e:
            continue
        lats = [r["lat"]] + [e["forecast"].get(str(l), {}).get("lat") for l in (24, 48, 72)]
        lons = [r["lon"]] + [e["forecast"].get(str(l), {}).get("lon") for l in (24, 48, 72)]
        is_agent = m.startswith("agent")
        ax.plot(lons, lats, color=C.get(m, C["agent"]),
                lw=2.2 if is_agent else 1.1, ls="-" if is_agent else "--",
                marker="o", ms=3, zorder=5 if is_agent else 3,
                label="StormDesk" if is_agent else LABELS.get(m, m))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"a  {r['name']} — track", loc="left")
    ax.legend(fontsize=6.5)

    ax = axes[1]
    leads = [0, 24, 48, 72]
    truth = [r["vmax"]] + [r.get(f"vmax_{l}") for l in (24, 48, 72)]
    ax.plot(leads, truth, color=INK, lw=2, marker="s", ms=4, label="Best track", zorder=6)
    for m, fc in methods.items():
        e = fc.get(cid)
        if not e:
            continue
        vs = [r["vmax"]] + [e["forecast"].get(str(l), {}).get("vmax") for l in (24, 48, 72)]
        is_agent = m.startswith("agent")
        ax.plot(leads, vs, color=C.get(m, C["agent"]),
                lw=2.2 if is_agent else 1.1, ls="-" if is_agent else "--",
                marker="o", ms=3, zorder=5 if is_agent else 3)
    ax.set_xlabel("Lead (h)"); ax.set_ylabel("Vmax (kt)")
    ax.set_xticks(leads)
    ax.set_title("b  Intensity", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig6_case_{r['name']}.pdf"))
    print("fig6 done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fig", choices=["fig1a", "fig1bc", "fig3", "fig4", "fig5", "fig6"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--sid", default=None)
    ap.add_argument("--agent", default="agent_full_qwen14b")
    args = ap.parse_args()
    out = args.out or work_dir("figures")
    style()
    if args.fig == "fig1a":
        fig1a(out)
    elif args.fig == "fig1bc":
        fig1bc(out)
    elif args.fig == "fig3":
        fig3(out, args.agent)
    elif args.fig == "fig4":
        fig4(out, args.agent)
    elif args.fig == "fig5":
        fig5(out)
    elif args.fig == "fig6":
        fig6(out, args.sid, args.agent)


if __name__ == "__main__":
    main()
