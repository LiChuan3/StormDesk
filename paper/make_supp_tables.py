"""Generate supplementary LaTeX tables from the result CSVs (results_v7/,
the analysis-manifest run)."""
import json
import os

import pandas as pd

RES = os.path.join(os.path.dirname(__file__), "results_v7")
OUT = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(OUT, exist_ok=True)

NICE = {
    "persistence": "Persistence", "cliper": "CLIPER5-class", "gru": "GRU",
    "transformer": "Transformer", "pangu": "Pangu-Weather", "fengwu": "FengWu",
    "cons_equal": "Consensus (equal)", "cons_weighted": "Consensus (weighted)",
    "cons_bc": "Consensus (bias-corr.)", "cons_aiwp": "Consensus (AIWP)",
    "hybrid_static": "Hybrid (static)", "hybrid_rules": "Hybrid + rules",
    "learned_gbt": "Learned combiner (GBT)",
    "learned_gbt_bounded": "Learned (bounded gate)",
    "learned_gbt2": "Learned combiner (GBT)",
    "learned_gbt_contract": "Learned (within contract)",
    "static_convex": "Static convex (global)",
    "static_basin": "Static convex (basin$\\times$lead)",
    "static_icat": "Static convex (intensity$\\times$lead)",
    "gate_ridge": "Linear (ridge) gate",
    "aiwp_postproc": "AIWP intensity post-proc.",
    "agent_full_qwen14b_gateadv": "office + gate advice",
    "agent_featmatch_qwen14b": "office + explicit features",
    "agent_featmatch_fs_qwen14b": "office + features + examples",
    "agent_full_qwen14b_t0": "StormDesk (14B, temp.\\ 0)",
    "agent_full_qwen14b_strongprior": "office, post-proc.\\ prior",
    "agent_full_llama8b_owncal": "StormDesk (Llama-8B, own calib.)",
    "gbt_static": "GBT static gate", "gbt_case": "GBT case gate",
    "gbt_shuffled": "GBT case-shuffled",
    "analog_median": "Analog point (median)", "analog_linear": "Analog point (linear)",
    "agent_mini_qwen14b": "mini office (3 calls)",
    "agent_full_llama8b": "StormDesk (Llama-8B)",
    "agent_no_auditor_llama8b": "no auditor (Llama-8B)",
    "agent_free_llama8b": "free (Llama-8B)",
    "agent_full_qwen7b": "StormDesk (7B)", "agent_full_qwen14b": "StormDesk (14B)",
    "agent_full_qwen72b": "StormDesk (72B)",
    "agent_single_qwen14b": "single (1 call)",
    "agent_single_refine_qwen14b": "single-refine (5 calls)",
    "agent_no_analogs_qwen14b": "no analogs",
    "agent_no_auditor_qwen14b": "no auditor",
    "agent_no_auditor_qwen14b_detaudit": "no auditor + mech.\\ corrector",
    "agent_no_diagnostics_qwen14b": "no diagnostics",
    "agent_full_qwen14b_anon": "StormDesk (14B, anonymized)",
    "agent_free_qwen14b": "free generation (14B)",
    "clim": "Climatology", "ri_analog": "Analog RI frequency",
    "ri_rules": "Office rule, literal", "ri_rules_graded": "Office rule, graded",
    "ri_logit": "Logistic regression", "ri_gbdt": "Gradient boosting",
}
ORDER = list(NICE)


def esc(m):
    return NICE.get(m, m.replace("_", "\\_"))


def tab_ci():
    df = pd.read_csv(os.path.join(RES, "test_metrics_ci.csv"))
    df["o"] = df.method.map({m: i for i, m in enumerate(ORDER)})
    df = df.sort_values(["lead", "o"])
    rows = []
    for lead in (24, 48, 72):
        sub = df[df.lead == lead]
        for _, r in sub.iterrows():
            rows.append(
                f"{lead} & {esc(r.method)} & "
                f"{r.track_km:.1f} [{r.track_lo:.1f}, {r.track_hi:.1f}] & "
                f"{r.vmax_mae_kt:.1f} [{r.vmax_lo:.1f}, {r.vmax_hi:.1f}] \\\\")
    half = (len(rows) + 1) // 2
    header = ("\\begin{tabular}{llcc}\n\\toprule\n"
              "Lead & Method & Track km [95\\% CI] & Vmax MAE kt [95\\% CI] \\\\\n"
              "\\midrule\n")
    footer = "\n\\bottomrule\n\\end{tabular}"
    out = (header + "\n".join(rows[:half]) + footer + "\\hfill" +
           header + "\n".join(rows[half:]) + footer)
    open(os.path.join(OUT, "supp_ci.tex"), "w").write(out)


def tab_sig():
    df = pd.read_csv(os.path.join(RES, "test_significance.csv"))
    df = df[df.p.notna()].copy()
    # keep the informative reference per target: everything vs the hybrid;
    # LLM policies and learned references also vs the office; the static
    # ladder also vs the global convex stack; strongprior vs its own prior
    llm_or_learned = {m for m in df.target.unique()
                      if m.startswith("agent") or m.startswith("learned")
                      or m in ("static_basin", "aiwp_postproc")}
    ladder = {"static_basin", "static_icat", "gate_ridge", "gbt_case"}
    keep = (
        (df.ref == "hybrid_static")
        | ((df.ref == "agent_full_qwen14b") & df.target.isin(llm_or_learned))
        | ((df.ref == "static_convex") & df.target.isin(ladder))
        | (df.ref == "aiwp_postproc")
    ) & (df.target != df.ref)
    df = df[keep].copy()
    # Holm within each target-reference family (six tests), matching the
    # paper's significance protocol
    def _holm(ps):
        import numpy as np
        order = np.argsort(ps.values)
        adj = np.empty(len(ps))
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, (len(ps) - rank) * ps.values[i])
            adj[i] = min(run, 1.0)
        return pd.Series(adj, index=ps.index)
    df["p_holm"] = df.groupby(["target", "ref"])["p"].transform(_holm)
    # rows grouped into (target, reference) families of six tests, so every
    # split (across the two floats and across the side-by-side columns)
    # falls on a family boundary
    fams = []
    for _, g in df.groupby(["target", "ref"], sort=False):
        fam = []
        for _, r in g.iterrows():
            unit = "km" if r.metric == "track" else "kt"
            fam.append(
                f"{esc(r.target)} & {esc(r.ref)} & {r.lead} & {r.metric} & "
                f"{r.mean_diff:+.2f} {unit} & {r.p:.4f} & {r.p_holm:.4f} \\\\")
        fams.append(fam)

    def split_at_boundary(groups, target_rows):
        first, rest, acc = [], list(groups), 0
        while rest and acc + len(rest[0]) <= target_rows:
            g = rest.pop(0)
            first.append(g)
            acc += len(g)
        return first, rest

    header = ("\\begin{tabular}{llllrrr}\n\\toprule\n"
              "Target & Reference & Lead & Metric & Mean diff & $p$ & "
              "$p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n")
    footer = "\n\\bottomrule\n\\end{tabular}"

    def two_col(groups):
        rows = [r for g in groups for r in g]
        left, right = split_at_boundary(groups, (len(rows) + 1) // 2)
        lrows = [r for g in left for r in g]
        rrows = [r for g in right for r in g]
        return (header + "\n".join(lrows) + footer + "\\hfill" +
                header + "\n".join(rrows) + footer)

    n_all = sum(len(g) for g in fams)
    fa, fb = split_at_boundary(fams, (n_all + 1) // 2)
    open(os.path.join(OUT, "supp_sig_a.tex"), "w").write(two_col(fa))
    open(os.path.join(OUT, "supp_sig_b.tex"), "w").write(two_col(fb))
    stale = os.path.join(OUT, "supp_sig.tex")
    if os.path.exists(stale):
        os.remove(stale)


def tab_ri():
    df = pd.read_csv(os.path.join(RES, "test_ri_baselines.csv"))
    df["o"] = df.method.map({m: i for i, m in enumerate(ORDER)})
    df = df.sort_values("o")
    lines = [
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Method & Thr. & POD [CI] & FAR [CI] & CSI [CI] & BSS [CI] & $p_{\\mathrm{CSI}}$ \\\\",
        "\\midrule"]
    for _, r in df.iterrows():
        if r.method == "clim":
            lines.append("Climatology & --- & 0 & --- & 0 & 0 (ref.) & --- \\\\")
            continue

        def ci(v, lo, hi, d=2):
            if pd.isna(lo):
                return f"{v:.{d}f}"
            return f"{v:.{d}f} [{lo:.{d}f},{hi:.{d}f}]"
        p = "---" if pd.isna(r.get("csi_vs_office_p")) else f"{r.csi_vs_office_p:.3f}"
        lines.append(
            f"{esc(r.method)} & {r.threshold:.2f} & "
            f"{ci(r.pod, r.get('pod_lo'), r.get('pod_hi'))} & "
            f"{ci(r.far, r.get('far_lo'), r.get('far_hi'))} & "
            f"{ci(r.csi, r.get('csi_lo'), r.get('csi_hi'))} & "
            f"{ci(r.bss, r.get('bss_lo'), r.get('bss_hi'))} & {p} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "supp_ri.tex"), "w").write("\n".join(lines))


def tab_break():
    df = pd.read_csv(os.path.join(RES, "test_breakdowns.csv"))
    keep = ["agent_full_qwen14b", "hybrid_static", "cons_bc"]
    df = df[df.method.isin(keep)]
    lines = [
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "Stratum & Method & $n_{24}$ & Track km 24/48/72 & Vmax MAE 24/48/72 \\\\".replace(
            "{llrrrr}", "{llrll}"),
        "\\midrule"]
    lines[0] = "\\begin{tabular}{llrll}"
    for st in df.stratum.unique():
        sub = df[df.stratum == st]
        for m in keep:
            s = sub[sub.method == m]
            if s.empty:
                continue
            by = {int(r.lead): r for _, r in s.iterrows()}
            if 24 not in by:
                continue
            tr = "/".join(f"{by[l].track_km:.0f}" if l in by else "--" for l in (24, 48, 72))
            vm = "/".join(f"{by[l].vmax_mae_kt:.1f}" if l in by else "--" for l in (24, 48, 72))
            lines.append(f"{st.replace('_', ' ').replace('=', ': ')} & {esc(m)} & "
                         f"{int(by[24].n)} & {tr} & {vm} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(os.path.join(OUT, "supp_break.tex"), "w").write("\n".join(lines))


def tab_logit():
    with open(os.path.join(RES, "ri_logit_coefs.json")) as f:
        c = json.load(f)
    items = sorted(c.items(), key=lambda kv: -abs(kv[1]))
    lines = ["\\begin{tabular}{lr}", "\\toprule",
             "Feature (standardized) & Coefficient \\\\", "\\midrule"]
    for k, v in items:
        name = k.replace("_", "\\_")
        lines.append(f"{name} & {v:+.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "supp_logit.tex"), "w").write("\n".join(lines))


if __name__ == "__main__":
    tab_ci()
    tab_sig()
    tab_ri()
    tab_break()
    tab_logit()
    print("wrote tables to", OUT)
