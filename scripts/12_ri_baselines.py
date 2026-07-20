"""Same-information probabilistic RI baselines.

Answers the reviewer question "is the RI skill just the classifier + the
rules?": fits dedicated RI classifiers on the calibration seasons (2018-2020)
using exactly the evidence available in the office briefing (environment,
satellite IR, analog statistics, consensus-prior trend, guidance spread), and
scores them on the same homogeneous test sample with the identical protocol
used for the agent probability (Platt scaling + CSI-optimal threshold, both
frozen on calibration data).

Methods:
  clim         calibration-season base rate (BSS reference)
  ri_analog    analog RI frequency used directly as the probability
  ri_rules     the hand-coded rule from the prompt executed literally
               (analog rate >= 0.3, shear < 15 kt, SST >= 28.5 C, POT >= 40 kt)
  ri_logit     L2 logistic regression on the full feature vector
  ri_gbdt      gradient-boosted trees on the same features
  agents       any forecast file with an explicit ri24_prob (scored with the
               office Platt coefficients from models/ri_calibration.json)

Outputs: results/<split>_ri_baselines.csv (+ storm-level bootstrap CIs),
results/<split>_ri_curves.json (threshold sweeps for the supplementary).
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.analogs import AnalogLibrary, entry_features, summarize_analogs
from stormdesk.evaluate import case_id, load_forecasts, ri_prob_calibration, calibrate_ri_prob
from stormdesk.geo import motion_uv_kmh, gc_distance_km
from stormdesk.guidance.merge import load_features, load_guidance

FEATS = ["v0", "dv12", "dv24", "abslat", "mot_speed", "shear_kt", "sst_c",
         "rh_mid_pct", "pot_kt", "div200", "bt_min", "cold_frac", "quad_std",
         "analog_ri", "analog_dv24", "prior_dv24", "spread24", "vrange24"]


# ---------------------------------------------------------------------------
# feature matrix
# ---------------------------------------------------------------------------
def build_matrix(split):
    with open(os.path.join(work_dir("cases"), f"{split}.pkl"), "rb") as f:
        cases = pickle.load(f)
    feats = load_features(split)
    guidance = load_guidance(split)
    lib = AnalogLibrary.load()
    bc = load_forecasts(os.path.join(work_dir("forecasts"), f"{split}_cons_bc.jsonl"))

    rows = []
    for _, r in cases.iterrows():
        if not np.isfinite(r.get("vmax_24", np.nan)):
            continue
        cid = case_id(r)
        ft = feats.get(cid, {})
        diag = ft.get("diag") or {}
        sat = ft.get("sat") or {}
        if not diag:
            continue
        h = r["history"]
        mu, mv = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
        dv12 = (r["vmax"] - h[-3]["vmax"]) if h[-3]["vmax"] is not None else 0.0
        dv24 = (r["vmax"] - h[0]["vmax"]) if h[0]["vmax"] is not None else 0.0
        summary = {}
        f = entry_features(r["lat"], r["lon"], r["vmax"], dv12, float(mu), float(mv),
                           diag, r["init"])
        if f is not None:
            summary = summarize_analogs(lib.query(f, r["lat"], r["sid"], k=12), r["vmax"])
        prior_dv24 = np.nan
        d = bc.get(cid)
        if d and d.get("forecast") and d["forecast"].get("24") and \
                d["forecast"]["24"].get("vmax") is not None:
            prior_dv24 = float(d["forecast"]["24"]["vmax"]) - float(r["vmax"])
        spread24 = vrange24 = np.nan
        g = guidance.get(cid) or {}
        ms = [e.get("24") or e.get(24) for e in g.values() if e]
        ms = [e for e in ms if e]
        if len(ms) >= 2:
            lat_c = float(np.mean([e["lat"] for e in ms]))
            lon_c = float(np.mean([e["lon"] for e in ms]))
            spread24 = float(np.mean([gc_distance_km(e["lat"], e["lon"], lat_c, lon_c)
                                      for e in ms]))
            vs = [e.get("vmax", e.get("vmax_kt")) for e in ms]
            vs = [x for x in vs if x is not None]
            if len(vs) >= 2:
                vrange24 = float(max(vs) - min(vs))
        rows.append(dict(
            case_id=cid, sid=r["sid"],
            y=int((r["vmax_24"] - r["vmax"]) >= 30.0),
            v0=r["vmax"], dv12=dv12, dv24=dv24, abslat=abs(r["lat"]),
            mot_speed=float(np.hypot(mu, mv)),
            shear_kt=diag.get("shear_kt"), sst_c=diag.get("sst_c"),
            rh_mid_pct=diag.get("rh_mid_pct"), pot_kt=diag.get("pot_kt"),
            div200=diag.get("div200_1e7"),
            bt_min=sat.get("bt_min_k"), cold_frac=sat.get("cold_frac_208k"),
            quad_std=sat.get("quadrant_bt_std_k"),
            analog_ri=summary.get("ri24_rate"), analog_dv24=summary.get("dv24_median"),
            prior_dv24=prior_dv24, spread24=spread24, vrange24=vrange24,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# calibration protocol (identical to the agent treatment)
# ---------------------------------------------------------------------------
def fit_platt(p_raw, y, iters=200):
    """1-D logistic on x = logit(p): z = a*x + b (Newton)."""
    p = np.clip(np.asarray(p_raw, float), 0.01, 0.99)
    x = np.log(p / (1 - p))
    y = np.asarray(y, float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = a * x + b
        q = 1 / (1 + np.exp(-z))
        g = np.array([np.sum((q - y) * x), np.sum(q - y)])
        w = q * (1 - q) + 1e-9
        H = np.array([[np.sum(w * x * x), np.sum(w * x)],
                      [np.sum(w * x), np.sum(w)]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(a), float(b)


def apply_platt(p_raw, a, b):
    p = np.clip(np.asarray(p_raw, float), 0.01, 0.99)
    x = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-(a * x + b)))


def csi_threshold(p_cal, y):
    grid = np.unique(np.concatenate([np.linspace(0.02, 0.98, 97), np.unique(p_cal)]))
    best_t, best_csi = 0.5, -1.0
    for t in grid:
        pred = p_cal >= t
        h = int(np.sum(pred & (y == 1)))
        m = int(np.sum(~pred & (y == 1)))
        fa = int(np.sum(pred & (y == 0)))
        csi = h / (h + m + fa) if (h + m + fa) else 0.0
        if csi > best_csi + 1e-12 or (abs(csi - best_csi) <= 1e-12 and t > best_t):
            best_csi, best_t = csi, float(t)
    return best_t


def contingency(p, y, t):
    pred = p >= t
    h = int(np.sum(pred & (y == 1)))
    m = int(np.sum(~pred & (y == 1)))
    fa = int(np.sum(pred & (y == 0)))
    return dict(hits=h, misses=m, false_alarms=fa,
                pod=h / (h + m) if (h + m) else np.nan,
                far=fa / (h + fa) if (h + fa) else np.nan,
                csi=h / (h + m + fa) if (h + m + fa) else np.nan)


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def impute_standardize(df, stats=None):
    X = df[FEATS].astype(float).copy()
    if stats is None:
        med = X.median()
        stats = dict(median=med, mean=None, std=None)
        X = X.fillna(med)
        stats["mean"], stats["std"] = X.mean(), X.std().replace(0, 1.0)
    else:
        X = X.fillna(stats["median"])
    Z = (X - stats["mean"]) / stats["std"]
    return Z.values, stats


def np_logistic_fit(X, y, l2=1.0, lr=0.05, iters=4000):
    """Fallback numpy logistic regression with balanced class weights."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    cw = np.where(y == 1, n / (2 * max(y.sum(), 1)), n / (2 * max((1 - y).sum(), 1)))
    for _ in range(iters):
        z = Xb @ w
        q = 1 / (1 + np.exp(-z))
        g = Xb.T @ (cw * (q - y)) / n + l2 * np.r_[w[:-1], 0] / n
        w -= lr * g
    return w


def np_logistic_prob(w, X):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return 1 / (1 + np.exp(-(Xb @ w)))


def rules_score(df):
    cond = ((df["analog_ri"].fillna(0) >= 0.3) & (df["shear_kt"].fillna(99) < 15)
            & (df["sst_c"].fillna(0) >= 28.5) & (df["pot_kt"].fillna(0) >= 40))
    n_met = ((df["analog_ri"].fillna(0) >= 0.3).astype(int)
             + (df["shear_kt"].fillna(99) < 15).astype(int)
             + (df["sst_c"].fillna(0) >= 28.5).astype(int)
             + (df["pot_kt"].fillna(0) >= 40).astype(int))
    return cond.values.astype(bool), (n_met.values / 4.0)


def grouped_folds(sids, k=5, seed=0):
    uniq = np.array(sorted(set(sids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, k)
    sid_arr = np.asarray(sids)
    for f in folds:
        te = np.isin(sid_arr, f)
        yield ~te, te


def cv_oof_probs(fit_fn, prob_fn, df, Z, y, k=5):
    oof = np.full(len(y), np.nan)
    for tr, te in grouped_folds(df["sid"].values, k=k):
        model = fit_fn(Z[tr], y[tr])
        oof[te] = prob_fn(model, Z[te])
    return oof


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--calib-split", default="calib")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--agent-glob", default="agent_*,hybrid_rules",
                    help="comma list of forecast-method name globs with ri24_prob")
    args = ap.parse_args()

    print("building calib matrix ...")
    dfc = build_matrix(args.calib_split)
    print("building test matrix ...")
    dft = build_matrix(args.split)
    yc, yt = dfc["y"].values, dft["y"].values
    base_rate = float(yc.mean())
    print(f"calib n={len(dfc)} events={yc.sum()} ({base_rate:.3f}); "
          f"test n={len(dft)} events={yt.sum()} ({yt.mean():.3f})")

    Zc, stats = impute_standardize(dfc)
    Zt, _ = impute_standardize(dft, stats)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier

        def fit_logit(X, y):
            return LogisticRegression(C=1.0, class_weight="balanced",
                                      max_iter=5000).fit(X, y)

        def prob_logit(m, X):
            return m.predict_proba(X)[:, 1]

        def fit_gbdt(X, y):
            return GradientBoostingClassifier(
                n_estimators=150, max_depth=2, learning_rate=0.05,
                subsample=0.8, random_state=0).fit(X, y)

        def prob_gbdt(m, X):
            return m.predict_proba(X)[:, 1]
        have_sklearn = True
    except ImportError:
        print("sklearn unavailable -> numpy logistic only")
        fit_logit = lambda X, y: np_logistic_fit(X, y)  # noqa: E731
        prob_logit = lambda m, X: np_logistic_prob(m, X)  # noqa: E731
        have_sklearn = False

    methods = {}  # name -> (p_test_raw, platt(a,b) or None, threshold)

    # climatology
    methods["clim"] = dict(p=np.full(len(dft), base_rate), a=None, thr=0.5)

    # analog frequency (no fitting; Platt straight on calib values)
    p_ana_c = dfc["analog_ri"].fillna(base_rate).values
    a, b = fit_platt(p_ana_c, yc)
    thr = csi_threshold(apply_platt(p_ana_c, a, b), yc)
    methods["ri_analog"] = dict(p=dft["analog_ri"].fillna(base_rate).values,
                                a=(a, b), thr=thr)

    # literal rule from the prompt (binary; no calibration by construction)
    rule_bin_t, rule_grade_t = rules_score(dft)
    rule_bin_c, rule_grade_c = rules_score(dfc)
    methods["ri_rules"] = dict(p=rule_bin_t.astype(float), a=None, thr=0.5)
    a, b = fit_platt(np.clip(rule_grade_c, 0.01, 0.99), yc)
    thr = csi_threshold(apply_platt(np.clip(rule_grade_c, 0.01, 0.99), a, b), yc)
    methods["ri_rules_graded"] = dict(p=np.clip(rule_grade_t, 0.01, 0.99), a=(a, b), thr=thr)

    # fitted classifiers: grouped-CV OOF probs on calib -> Platt + threshold,
    # refit on full calib -> test probs
    for name, ff, pf in ([("ri_logit", fit_logit, prob_logit)]
                         + ([("ri_gbdt", fit_gbdt, prob_gbdt)] if have_sklearn else [])):
        oof = cv_oof_probs(ff, pf, dfc, Zc, yc)
        a, b = fit_platt(oof, yc)
        thr = csi_threshold(apply_platt(oof, a, b), yc)
        model = ff(Zc, yc)
        methods[name] = dict(p=pf(model, Zt), a=(a, b), thr=thr)
        if name == "ri_logit" and have_sklearn:
            coef = dict(zip(FEATS, np.round(model.coef_[0], 3).tolist()))
            with open(os.path.join(work_dir("results"), "ri_logit_coefs.json"), "w") as f:
                json.dump(coef, f, indent=1)

    # agent / office probabilities (office Platt from models/, frozen threshold)
    office = ri_prob_calibration() or {}
    cid_pos = {c: i for i, c in enumerate(dft["case_id"].values)}
    fdir = work_dir("forecasts")
    import fnmatch
    pats = args.agent_glob.split(",")
    files = sorted(glob.glob(os.path.join(fdir, f"{args.split}_*.jsonl")))
    for path in files:
        name = os.path.basename(path)[len(args.split) + 1:-6]
        if not any(fnmatch.fnmatch(name, p) for p in pats):
            continue
        fc = load_forecasts(path)
        p = np.full(len(dft), np.nan)
        for c, d in fc.items():
            if c in cid_pos and d.get("ri24_prob") is not None and d.get("forecast"):
                p[cid_pos[c]] = float(d["ri24_prob"])
        if np.isfinite(p).sum() < 50:
            continue
        if name == "hybrid_rules":
            methods[name] = dict(p=p, a=None, thr=0.5)
        else:
            methods[name] = dict(p=p, a=(office.get("a", 1.0), office.get("b", 0.0)),
                                 thr=office.get("threshold", 0.5))

    # homogeneous sample: rows where every method has a probability
    mask = np.ones(len(dft), bool)
    for name, m in methods.items():
        mask &= np.isfinite(m["p"])
    print(f"homogeneous RI sample: n={mask.sum()} events={yt[mask].sum()}")
    y = yt[mask]
    sids = dft["sid"].values[mask]
    test_rate = float(y.mean())
    bs_clim = brier(np.full(mask.sum(), base_rate), y)          # frozen (calib)
    bs_clim_test = brier(np.full(mask.sum(), test_rate), y)     # test climatology

    def extended(pc):
        """PR-AUC, ROC-AUC, log loss, calibration slope/intercept (on logit)."""
        out = {}
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
            out["roc_auc"] = round(float(roc_auc_score(y, pc)), 3)
            out["pr_auc"] = round(float(average_precision_score(y, pc)), 3)
            out["log_loss"] = round(float(log_loss(y, np.clip(pc, 1e-4, 1 - 1e-4))), 4)
        except Exception:  # noqa: BLE001
            pass
        x = np.log(np.clip(pc, 0.01, 0.99) / (1 - np.clip(pc, 0.01, 0.99)))
        # logistic recalibration slope/intercept: y ~ sigmoid(s*x + i); 1 Newton-free fit
        a2, b2 = fit_platt(pc, y)
        out["cal_slope"] = round(a2, 3)
        out["cal_intercept"] = round(b2, 3)
        return out

    rows, curves = [], {}
    pc_by_m = {}
    for name, m in methods.items():
        p_raw = m["p"][mask]
        pc = apply_platt(p_raw, *m["a"]) if m["a"] else p_raw
        pc_by_m[name] = pc
        c = contingency(pc, y, m["thr"])
        bs = brier(pc, y)
        rows.append(dict(method=name, n=int(mask.sum()), events=int(y.sum()),
                         threshold=round(m["thr"], 3), **c,
                         brier=round(bs, 4), bss=round(1 - bs / bs_clim, 3),
                         bss_test=round(1 - bs / bs_clim_test, 3), **extended(pc)))
        grid = np.linspace(0.01, 0.99, 99)
        sw = [contingency(pc, y, t) for t in grid]
        curves[name] = dict(thresholds=grid.tolist(),
                            pod=[s["pod"] for s in sw], far=[s["far"] for s in sw],
                            csi=[s["csi"] for s in sw])

    # storm-level bootstrap CIs
    rng = np.random.default_rng(0)
    uniq = np.array(sorted(set(sids)))
    sid_idx = {s: np.where(sids == s)[0] for s in uniq}
    boot = {name: {"pod": [], "far": [], "csi": [], "bss": []} for name in methods}
    ref = "agent_full_qwen14b"
    csi_diff = {name: [] for name in methods}
    bss_diff = {name: [] for name in methods}
    for _ in range(args.n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([sid_idx[s] for s in pick])
        yb = y[idx]
        if yb.sum() == 0:
            continue
        bsc = brier(np.full(len(idx), base_rate), yb)
        ref_csi = ref_bss = None
        for name, m in methods.items():
            pb = pc_by_m[name][idx]
            c = contingency(pb, yb, m["thr"])
            boot[name]["pod"].append(c["pod"])
            boot[name]["far"].append(c["far"])
            boot[name]["csi"].append(c["csi"])
            boot[name]["bss"].append(1 - brier(pb, yb) / bsc)
            if name == ref:
                ref_csi = c["csi"]
                ref_bss = boot[name]["bss"][-1]
        if ref_csi is not None:
            for name in methods:
                csi_diff[name].append(boot[name]["csi"][-1] - ref_csi)
                bss_diff[name].append(boot[name]["bss"][-1] - ref_bss)

    for r in rows:
        b = boot[r["method"]]
        for k in ("pod", "far", "csi", "bss"):
            v = np.array([x for x in b[k] if np.isfinite(x)])
            if len(v):
                r[f"{k}_lo"], r[f"{k}_hi"] = round(float(np.percentile(v, 2.5)), 3), \
                    round(float(np.percentile(v, 97.5)), 3)
        d = np.array([x for x in csi_diff[r["method"]] if np.isfinite(x)])
        if len(d) and r["method"] != ref:
            r["csi_vs_office_p"] = round(2 * min((d > 0).mean(), (d < 0).mean()), 3)
            # TOST-style equivalence via 90% CI of the paired CSI difference
            # within +-0.05 (equivalence margin)
            lo90, hi90 = np.percentile(d, 5), np.percentile(d, 95)
            r["csi_diff_lo90"], r["csi_diff_hi90"] = round(float(lo90), 3), \
                round(float(hi90), 3)
            r["csi_equiv_pm05"] = bool(lo90 > -0.05 and hi90 < 0.05)
        db = np.array([x for x in bss_diff[r["method"]] if np.isfinite(x)])
        if len(db) and r["method"] != ref:
            r["bss_vs_office_p"] = round(2 * min((db > 0).mean(), (db < 0).mean()), 3)
        # post-hoc sensitivity: test-set-optimal threshold (labelled as such)
        cv = curves[r["method"]]
        csis = [x if x is not None else -1 for x in cv["csi"]]
        j = int(np.argmax(csis))
        r["csi_testopt"] = round(csis[j], 3)
        r["thr_testopt"] = round(cv["thresholds"][j], 2)

    out = pd.DataFrame(rows).sort_values("csi", ascending=False)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    rdir = work_dir("results")
    out.to_csv(os.path.join(rdir, f"{args.split}_ri_baselines.csv"), index=False)
    with open(os.path.join(rdir, f"{args.split}_ri_curves.json"), "w") as f:
        json.dump(curves, f)
    print(f"wrote {rdir}/{args.split}_ri_baselines.csv and _ri_curves.json")


if __name__ == "__main__":
    main()
