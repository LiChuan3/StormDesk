"""Publication figures for the optimized paper (优化版).

Style: hand-drawn top-venue look (Comic Sans MS, thin marks, direct labels,
no chartjunk), Okabe-Ito colorblind-safe palette.  All numbers are read from
results_v7/ (the frozen analysis manifest run); nothing is typed by hand
except the free-generation safety numbers, which come from the supplementary
free-generation table and are annotated with their source.
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results_v7")

# ---------------------------------------------------------------- style
from matplotlib import font_manager
CS = "Comic Sans MS"
assert any("Comic Sans" in f.name for f in font_manager.fontManager.ttflist), \
    "Comic Sans MS not visible to matplotlib"

mpl.rcParams.update({
    "font.family": CS,
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "figure.dpi": 200,
})

# Okabe-Ito, fixed assignment by entity (never by rank)
C_PRIOR = "#818181"      # static hybrid prior
C_BASIN = "#56B4E9"      # regime-conditioned static
C_LEARN = "#009E73"      # supervised policies
C_LLM = "#E69F00"        # zero-shot LLM office
C_LLM2 = "#F2CE85"       # single-agent refine (same family, lighter)
C_ADV = "#0072B2"        # office + gate advice
C_FEAT = "#D55E00"       # office + explicit features
C_FS = "#9E3D00"         # + worked examples (darker of same family)

LEADS = (24, 48, 72)

ci = pd.read_csv(os.path.join(RES, "test_metrics_ci.csv")).set_index(
    ["method", "lead"])


def m(method, lead, col="track_km"):
    return float(ci.loc[(method, lead), col])


def savefig(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out)


# ================================================================ headroom U
# U with 90% storm-bootstrap CIs (static = hybrid prior, learned = gate+stack),
# from the manifest headroom run (headroom_v7_featmatch.txt / test_headroom.json)
U = {  # policy -> [(U, lo, hi) at 24/48/72 h], track
    "office + gate advice": [(0.20, -0.87, 0.43), (0.14, -0.85, 0.39), (0.25, 0.06, 0.35)],
    "office (conservative)": [(-0.26, -2.71, 0.19), (-0.41, -3.23, 0.06), (0.10, -0.18, 0.23)],
    "single agent, self-refine": [(-0.21, -2.06, 0.09), (-0.27, -2.10, 0.04), (0.03, -0.18, 0.12)],
    "office + explicit features": [(-2.67, -15.06, -0.55), (-3.08, -17.98, -0.58), (-0.49, -1.88, -0.07)],
    "  + worked examples": [(-5.84, -27.76, -2.02), (-9.22, -48.71, -2.85), (-1.81, -5.00, -0.79)],
}
U_COLOR = {
    "office + gate advice": C_ADV,
    "office (conservative)": C_LLM,
    "single agent, self-refine": C_LLM2,
    "office + explicit features": C_FEAT,
    "  + worked examples": C_FS,
}


def fig_headroom():
    # forest-plot layout: bars and CI whiskers live in the panel, every
    # number lives in a value rail right of the U=1 line, so no text ever
    # sits on a bar, a whisker or an arrow
    fig, ax = plt.subplots(figsize=(3.35, 3.05))
    XMIN = -11.0
    X_VAL, X_NOTE = 1.50, 3.15   # value rail / annotation rail
    policies = list(U)
    n = len(policies)
    bar_h = 0.24
    yticks, ylabels = [], []
    for i, pol in enumerate(policies):
        y0 = (n - 1 - i)            # group center, top group first
        yticks.append(y0)
        ylabels.append(pol)
        for j, lead in enumerate(LEADS):
            u, lo, hi = U[pol][j]
            y = y0 + (1 - j) * bar_h  # 24h on top
            color = U_COLOR[pol]
            alpha = (1.0, 0.72, 0.45)[j]
            ax.barh(y, max(u, XMIN), height=bar_h * 0.88, color=color,
                    alpha=alpha, zorder=3)
            # CI whisker, clipped at the left edge with an arrow
            lo_c = max(lo, XMIN + 0.15)
            ax.plot([lo_c, min(hi, 1.30)], [y, y], color="#333333",
                    lw=0.7, zorder=4)
            if lo < XMIN:
                ax.annotate("", xy=(XMIN + 0.02, y), xytext=(lo_c, y),
                            arrowprops=dict(arrowstyle="-|>", lw=0.7,
                                            color="#333333"), zorder=4)
            # value rail (always in empty space)
            ax.text(X_VAL, y, f"{u:+.2f}".replace("-", "−"),
                    fontsize=6.2, va="center", ha="left", color="#222222")
            # annotation rail: lead key on the first group, clipped CI bounds
            # on the arrowed rows
            if i == 0:
                note = f"{lead} h"
            elif lo < XMIN:
                note = f"CI {lo:.0f}".replace("-", "−")
            else:
                note = ""
            if note:
                ax.text(X_NOTE, y, note, fontsize=6.2, va="center",
                        ha="left", color="#888888")

    ax.axvline(0, color="#818181", lw=0.9, zorder=2)
    ax.axvline(1, color=C_LEARN, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.text(-0.18, n - 0.28, "static prior\nU = 0", fontsize=6.8, ha="right",
            va="bottom", color="#555555", linespacing=0.95)
    ax.text(1.12, n - 0.28, "supervised\ngate U = 1", fontsize=6.8,
            ha="left", va="bottom", color=C_LEARN, linespacing=0.95)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xlim(XMIN, 5.2)
    ax.set_ylim(-0.55, n + 0.55)
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 1])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("headroom utilization U (track)", fontsize=8)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(XMIN, 1.55)
    ax.tick_params(axis="y", length=0)
    savefig(fig, "fig_headroom.pdf")


# ================================================================ ladder
def fig_ladder():
    stages = ["static_convex", "static_basin", "learned_gbt2"]
    names = ["prior", "+global\nstatic", "+basin\nstatic", "+case\ngate"]
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.05))
    for ax, lead in zip(axes, (24, 72)):
        vals = [m("hybrid_static", lead)] + [m(s, lead) for s in stages]
        office = m("agent_full_qwen14b", lead)
        x = np.arange(4)
        # descending step ladder
        ax.plot(x, vals, "-", color="#BBBBBB", lw=1.0, zorder=2)
        cols = [C_PRIOR, C_PRIOR, C_BASIN, C_LEARN]
        ax.scatter(x, vals, s=26, c=cols, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + (vals[0] - vals[-1]) * 0.08, f"{v:.0f}",
                    fontsize=6.2, ha="center", va="bottom", color="#222222")
        # office reference line
        ax.axhline(office, color=C_LLM, lw=1.0, ls=(0, (4, 2)), zorder=1)
        ax.text(3.42, office, "LLM\noffice", fontsize=6.0, color=C_LLM,
                va="center", ha="left", linespacing=0.95)
        # share of headroom captured by the basin table
        share = (vals[0] - vals[2]) / (vals[0] - vals[3])
        ax.annotate(f"basin table:\n{share:.0%} of the gain",
                    xy=(2.0, vals[2]), xytext=(0.06, 0.10),
                    textcoords="axes fraction", fontsize=6.2,
                    color="#1F82BC", ha="left", linespacing=0.95,
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#888888",
                                    connectionstyle="arc3,rad=0.25"))
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=5.6, linespacing=0.9)
        ax.set_title(f"{lead} h", fontsize=8, pad=2)
        ax.set_xlim(-0.45, 3.9)
        lo, hi = min(vals[-1], office), max(vals[0], office)
        ax.set_ylim(lo - (hi - lo) * 0.42, hi + (hi - lo) * 0.30)
    axes[0].set_ylabel("track error (km)")
    fig.subplots_adjust(wspace=0.40)
    savefig(fig, "fig_ladder.pdf")


# ================================================================ RI
# numbers from the frozen RI verification table (results_v7; identical to the
# v6 main-paper Table 2): homogeneous probabilistic sample, n=1287, 87 events
RI = [  # full name, short name, POD, FAR, CSI, PR-AUC, color, marker, filled
    ("analog frequency", "analog", 0.48, 0.78, 0.177, 0.218, C_PRIOR, "D", True),
    ("logistic regression", "logistic", 0.53, 0.75, 0.204, 0.306, C_LEARN, "s", True),
    ("gradient boosting", "GBT", 0.52, 0.75, 0.202, 0.266, "#0F7B5C", "s", False),
    ("office rule, literal", "office rule", 0.28, 0.68, 0.174, 0.137, C_PRIOR, "^", False),
    ("office (14B)", "office 14B", 0.69, 0.82, 0.169, 0.160, C_LLM, "o", True),
    ("office (72B)", "72B", 0.62, 0.79, 0.188, 0.208, C_FS, "o", True),
    ("mini office", "mini office", 0.67, 0.80, 0.180, 0.183, "#C88A1E", "o", True),
    ("single (1 call)", "single call", 0.48, 0.79, 0.174, 0.183, C_LLM2, "o", True),
    ("no analogs", "no analogs", 0.87, 0.89, 0.110, 0.119, C_FEAT, "o", True),
]
def fig_ri():
    # three aligned bar columns sharing one row order: a visual version of
    # the RI table (references on top, LLM policies below, as in Table 1),
    # so no two labels can ever collide
    refs, llms = RI[:4], RI[4:]
    rows = refs + llms
    ys, y = [], 0.0
    for k, r in enumerate(rows):
        ys.append(y)
        y -= 1.0
        if k == len(refs) - 1:
            y -= 0.7          # visual gap between the two groups
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.90), sharey=True,
                             gridspec_kw=dict(wspace=0.34))
    metrics = [("POD", 2, 1.0, (0, 0.5, 1.0), ("0", ".5", "1")),
               ("FAR", 3, 1.0, (0, 0.5, 1.0), ("0", ".5", "1")),
               ("PR-AUC", 5, 0.40, (0, 0.3), ("0", ".3"))]
    for ax, (title, idx, xmax, ticks, ticklab) in zip(axes, metrics):
        for yy, r in zip(ys, rows):
            v = r[idx]
            if r[8]:
                ax.barh(yy, v, height=0.62, color=r[6], zorder=3)
            else:
                ax.barh(yy, v, height=0.62, facecolor="white",
                        edgecolor=r[6], lw=0.9, zorder=3)
            ax.text(v + xmax * 0.035, yy, f"{v:.2f}".lstrip("0"),
                    fontsize=6.2, va="center", ha="left", color="#333333")
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xlim(0, xmax)
        ax.set_xticks(list(ticks))
        ax.set_xticklabels(ticklab)
        ax.tick_params(axis="x", labelsize=6.8)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=7)
    axes[0].set_ylim(min(ys) - 0.75, 0.75)
    savefig(fig, "fig_ri.pdf")


# ================================================================ audit
# numbers from the audit-stage transcript replay (results_v7; identical to
# the v6 main-paper Table 3): intensity MAE in kt at 24/48/72 h
AUDIT = [  # rows top to bottom, worst 72-h MAE first
    ("LLM auditor + chief", C_LLM, (13.3, 17.7, 18.8)),
    ("no audit (draft)", C_PRIOR, (12.0, 16.8, 18.2)),
    ("affine recal. of draft", C_BASIN, (11.7, 16.1, 17.0)),
    ("learned (GBT) auditor", C_LEARN, (11.3, 15.0, 15.9)),
]


def fig_audit():
    fig, ax = plt.subplots(figsize=(3.35, 1.58))
    nr = len(AUDIT)
    for i, (name, col, vals) in enumerate(AUDIT):
        y = nr - 1 - i
        ax.plot([vals[0], vals[2]], [y, y], color=col, lw=1.0, alpha=0.35,
                zorder=2)
        for j, v in enumerate(vals):
            ax.scatter(v, y, s=18, color=col, alpha=(1.0, 0.72, 0.45)[j],
                       zorder=3)
            # nudge the 48 h / 72 h labels apart where the points sit close
            dx = 0.0
            if vals[2] - vals[1] < 1.15:
                dx = {1: -0.18, 2: 0.18}.get(j, 0.0)
            ax.text(v + dx, y - 0.24, f"{v:.1f}", fontsize=6.2, ha="center",
                    va="top", color="#333333")
            if i == 0:  # lead key above the top row only
                ax.text(v + dx, y + 0.26, f"{LEADS[j]} h", fontsize=6.2,
                        ha="center", va="bottom", color="#888888")
    ax.set_yticks(range(nr))
    ax.set_yticklabels([a[0] for a in AUDIT][::-1])
    ax.set_xlim(10.4, 19.9)
    ax.set_ylim(-0.70, nr - 0.22)
    ax.set_xticks([11, 13, 15, 17, 19])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("intensity MAE (kt), transcript replay", fontsize=7.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    savefig(fig, "fig_audit.pdf")


# ================================================================ landscape
# data exported from the server run (scripts/tmp_export_figdata.py): test-set
# best tracks from IBTrACS, Natural Earth 110m land, calibration-season member
# evaluation (same numbers as supplementary fig1bc_motivation)
DATA = os.path.join(HERE, "data")

MEM_LABEL = dict(cliper="CLIPER", gru="GRU", transformer="Transf.",
                 pangu="Pangu", fengwu="FengWu",
                 cons_aiwp="AIWP\ncons.", cons_weighted="wtd.\ncons.")


def _load_land():
    with open(os.path.join(DATA, "land_polys.json")) as f:
        return json.load(f)


def _draw_land(ax, polys, wrap360=False, window=None):
    for x, y in polys:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if window is not None:
            x0, x1, y0, y1 = window
            if x.max() < x0 or x.min() > x1 or y.max() < y0 or y.min() > y1:
                continue
        offs = (0.0, 360.0) if wrap360 else (0.0,)
        for off in offs:
            ax.fill(x + off, y, facecolor="#EFEDE6", edgecolor="#CCCAC0",
                    lw=0.3, zorder=1)


def fig_landscape():
    with open(os.path.join(DATA, "map_tracks.json")) as f:
        mt = json.load(f)
    with open(os.path.join(DATA, "motivation.json")) as f:
        mot = json.load(f)
    land = _load_land()

    fig = plt.figure(figsize=(7.0, 3.45))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.24, 1.0],
                          hspace=0.42, wspace=0.28,
                          left=0.065, right=0.985, top=0.93, bottom=0.10)
    axm = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1])

    # ---- (a) map of test-season best tracks, colored by intensity ----
    _draw_land(axm, land, wrap360=True)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "warm", ["#F2E2C4", "#E69F00", "#D55E00", "#7A1F00"])
    norm = mpl.colors.Normalize(vmin=20, vmax=145)
    for tr in mt["tracks"]:
        lon = np.array([np.nan if v is None else v for v in tr["lon"]], float)
        lat = np.array([np.nan if v is None else v for v in tr["lat"]], float)
        v = np.array([np.nan if x is None else x for x in tr["vmax"]], float)
        lon = np.where(lon < 0, lon + 360, lon)
        brk = np.where(np.abs(np.diff(lon)) > 180)[0]
        for s in np.split(np.arange(len(lon)), brk + 1):
            if len(s) < 2:
                continue
            axm.scatter(lon[s], lat[s], c=v[s], cmap=cmap, norm=norm,
                        s=1.1, linewidths=0, zorder=3)
    axm.set_xlim(0, 360)
    axm.set_ylim(-48, 52)
    axm.set_xticks([])
    axm.set_yticks([])
    for sp in ("left", "bottom"):
        axm.spines[sp].set_visible(False)
    axm.set_title(f"a   {mt['n_cycles']:,} forecast cycles over "
                  f"{mt['n_storms']} storms, 2021–2022 test seasons",
                  loc="left", fontsize=8, pad=3)
    # colorbar in the storm-free southeast Pacific
    cax = axm.inset_axes([0.615, 0.10, 0.20, 0.055])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([35, 85, 135])
    cb.ax.tick_params(labelsize=6.2, length=1.8, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cax.set_title("best-track intensity (kt)", fontsize=6.2, pad=2,
                  color="#555555")

    # ---- (b) member track error, calibration seasons ----
    order = ["cliper", "gru", "transformer", "pangu", "fengwu",
             "cons_aiwp", "cons_weighted"]
    skill = mot["track_skill"]
    x = np.arange(len(order), dtype=float)
    for j, lead in enumerate(LEADS):
        vals = [skill[mm][str(lead)] for mm in order]
        cols = ["#818181" if mm.startswith("cons") else "#0072B2"
                for mm in order]
        axb.bar(x + (j - 1) * 0.27, vals, width=0.25, color=cols,
                alpha=(1.0, 0.68, 0.40)[j], zorder=3)
    axb.set_xticks(x)
    axb.set_xticklabels([MEM_LABEL[mm] for mm in order], fontsize=6.2,
                        linespacing=0.9)
    axb.set_ylabel("track error (km)", fontsize=7.5)
    axb.tick_params(axis="y", labelsize=6.8)
    axb.set_title("b   member track skill is heterogeneous", loc="left",
                  fontsize=8, pad=3)
    handles = [mpl.patches.Patch(facecolor="#0072B2", alpha=a)
               for a in (1.0, 0.68, 0.40)]
    axb.legend(handles, ["24 h", "48 h", "72 h"], fontsize=6.2,
               loc="upper right", handlelength=1.0, handleheight=0.9,
               borderaxespad=0.1, labelspacing=0.3)

    # ---- (c) 48-h intensity bias, calibration seasons ----
    fams = [("gru", "#818181"), ("transformer", "#818181"),
            ("cliper", "#818181"), ("pangu", "#0072B2"),
            ("fengwu", "#0072B2")]
    pos = [0.0, 1.0, 2.0, 3.4, 4.4]
    vp = axc.violinplot([mot["bias48"][mm] for mm, _ in fams], positions=pos,
                        showmedians=True, widths=0.8)
    for body, (mm, col) in zip(vp["bodies"], fams):
        body.set_facecolor(col)
        body.set_alpha(0.45)
        body.set_edgecolor("none")
    vp["cmedians"].set_color("#555555")
    vp["cmedians"].set_linewidth(0.8)
    for k in ("cmins", "cmaxes", "cbars"):
        vp[k].set_visible(False)
    axc.axhline(0, color="#333333", lw=0.7, ls=(0, (4, 2)), zorder=2)
    axc.set_xticks(pos)
    axc.set_xticklabels([MEM_LABEL[mm].replace("\n", " ") for mm, _ in fams],
                        fontsize=6.2)
    axc.set_ylabel("48-h intensity bias (kt)", fontsize=7.5)
    axc.tick_params(axis="y", labelsize=6.8)
    axc.set_yticks([-80, -40, 0, 40])
    axc.set_ylim(-100, 80)
    axc.set_title("c   raw AIWP intensity is far too weak", loc="left",
                  fontsize=8, pad=3)
    # family labels live in the empty strip above the violins
    axc.text(1.0, 66, "statistical: unbiased, noisy", fontsize=6.2,
             ha="center", color="#555555")
    axc.text(3.9, 30, "tracker-derived\nAIWP: −27 kt", fontsize=6.2,
             ha="center", color="#0072B2", linespacing=0.95)
    savefig(fig, "fig_landscape.pdf")


# ================================================================ case study
# Noru (2022), initialized 2022-09-24 00 UTC: +87 kt/24 h RI, Luzon crossing,
# re-intensification.  Data from the recorded test-run forecasts (server
# export); the office line is the deployed 14B office.
CASE_STYLE = [  # key, label, color, lw, ls, zorder
    ("truth", "best track", "#111111", 1.6, "-", 7),
    ("agent_full_qwen14b", "StormDesk office", C_LLM, 1.7, "-", 6),
    ("cons_bc", "static prior", C_PRIOR, 1.2, "-", 5),
    ("pangu", "Pangu-Weather", "#0072B2", 0.9, (0, (4, 2)), 4),
    ("fengwu", "FengWu", "#56B4E9", 0.9, (0, (4, 2)), 4),
    ("gru", "GRU", "#009E73", 0.9, (0, (4, 2)), 3),
    ("transformer", "Transformer", "#CC79A7", 0.9, (0, (4, 2)), 3),
    ("cliper", "CLIPER", "#D55E00", 0.9, (0, (1.5, 1.5)), 3),
]


def fig_case():
    with open(os.path.join(DATA, "noru_case.json")) as f:
        case = json.load(f)
    land = _load_land()
    truth = case["truth"]
    leads = ("24", "48", "72")

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(3.35, 2.02),
        gridspec_kw=dict(width_ratios=[1.12, 1.0], wspace=0.42,
                         left=0.105, right=0.985, top=0.895, bottom=0.345))

    def series(key, field):
        if key == "truth":
            return truth[field]
        mth = case["methods"][key]
        return [truth[field][0]] + [mth[l][field] for l in leads]

    handles = []
    for key, label, col, lw, ls, z in CASE_STYLE:
        lons, lats = series(key, "lon"), series(key, "lat")
        marker = "s" if key == "truth" else "o"
        ln, = axa.plot(lons, lats, color=col, lw=lw, ls=ls, marker=marker,
                       ms=2.6 if key == "truth" else 2.0, zorder=z,
                       label=label)
        handles.append(ln)
        axb.plot([0, 24, 48, 72], series(key, "vmax"), color=col, lw=lw,
                 ls=ls, marker=marker, ms=2.6 if key == "truth" else 2.0,
                 zorder=z)

    _draw_land(axa, land, window=(105, 135, 8, 24))
    axa.set_xlim(109.5, 130.5)
    axa.set_ylim(12.6, 18.4)
    axa.set_xticks([112, 118, 124, 130])
    axa.set_yticks([13, 15, 17])
    axa.tick_params(labelsize=6.2)
    axa.set_xlabel("longitude (°E)", fontsize=6.8, labelpad=1.5)
    axa.set_ylabel("latitude (°N)", fontsize=6.8, labelpad=1.5)
    axa.set_title("a   track", loc="left", fontsize=7.5, pad=2)

    axb.set_xticks([0, 24, 48, 72])
    axb.set_yticks([40, 80, 120])
    axb.tick_params(labelsize=6.2)
    axb.set_xlabel("lead time (h)", fontsize=6.8, labelpad=1.5)
    axb.set_ylabel("intensity (kt)", fontsize=6.8, labelpad=1.5)
    axb.set_ylim(18, 150)
    axb.set_title("b   intensity", loc="left", fontsize=7.5, pad=2)
    axb.annotate("+87 kt / 24 h", xy=(20, 122), xytext=(34, 137),
                 fontsize=6.2, color="#555555",
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#888888"))

    # legend strip below both panels: guaranteed empty space, nothing to
    # collide with (the supplementary original drew it inside the track panel)
    fig.legend(handles, [c[1] for c in CASE_STYLE], loc="lower center",
               bbox_to_anchor=(0.5, -0.015), ncol=4, fontsize=5.8,
               frameon=False, handlelength=1.5, columnspacing=0.8,
               handletextpad=0.4, labelspacing=0.35)
    savefig(fig, "fig_case_noru.pdf")


# ================================================================ curves
def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9))
    track_pol = [
        ("hybrid_static", "static prior", C_PRIOR, "-"),
        ("static_basin", "basin static", C_BASIN, "-"),
        ("learned_gbt_contract", "supervised gate (in contract)", C_LEARN, "-"),
        ("agent_full_qwen14b", "LLM office", C_LLM, "-"),
        ("agent_featmatch_qwen14b", "office + features, decisive", C_FEAT, (0, (4, 2))),
    ]
    vmax_pol = [
        ("hybrid_static", "static prior", C_PRIOR, "-"),
        ("learned_gbt_contract", "supervised gate (in contract)", C_LEARN, "-"),
        ("aiwp_postproc", "AIWP post-processor", "#0F7B5C", (0, (4, 2))),
        ("agent_full_qwen14b", "LLM office", C_LLM, "-"),
    ]
    for ax, pols, col, unit in (
            (axes[0], track_pol, "track_km", "track error (km)"),
            (axes[1], vmax_pol, "vmax_mae_kt", "intensity MAE (kt)")):
        for meth, label, color, ls in pols:
            v = [m(meth, l, col) for l in LEADS]
            lo = [m(meth, l, col.replace("_km", "_lo").replace("_mae_kt", "_lo"))
                  for l in LEADS]
            hi = [m(meth, l, col.replace("_km", "_hi").replace("_mae_kt", "_hi"))
                  for l in LEADS]
            ax.fill_between(LEADS, lo, hi, color=color, alpha=0.10, lw=0)
            ax.plot(LEADS, v, linestyle=ls, color=color, lw=1.3, marker="o",
                    ms=2.6, label=label)
            ax.text(72.9, v[-1], label, fontsize=6.0, color=color,
                    va="center", ha="left")
        ax.set_xticks(LEADS)
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel(unit)
        ax.set_xlim(21, 90)
    savefig(fig, "fig_curves.pdf")


if __name__ == "__main__":
    fig_headroom()
    fig_ladder()
    fig_ri()
    fig_audit()
    fig_landscape()
    fig_case()
    fig_curves()
