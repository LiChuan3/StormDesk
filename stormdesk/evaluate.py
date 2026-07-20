"""NHC-style homogeneous verification of track and intensity forecasts.

Metrics: great-circle track error (km), Vmax MAE and bias (kt), skill relative
to the CLIPER5-class baseline, RI (>= +30 kt/24 h) contingency scores, and
paired significance tests with a serial-correlation-adjusted sample size
(following NHC verification practice).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from .geo import gc_distance_km

VERIF = [24, 48, 72]


def case_id(row) -> str:
    return f"{row['sid']}_{pd.Timestamp(row['init']).strftime('%Y%m%d%H')}"


def load_forecasts(path: str) -> dict:
    """JSONL of {case_id, forecast: {lead: {lat,lon,vmax}}} -> dict."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            fc = d.get("forecast")
            if fc:  # tracker rows store intensity as vmax_kt
                for e in fc.values():
                    if isinstance(e, dict) and e.get("vmax") is None \
                            and e.get("vmax_kt") is not None:
                        e["vmax"] = e["vmax_kt"]
            out[d["case_id"]] = d
    return out


def _truth(row, lead):
    la, lo, v = row.get(f"lat_{lead}"), row.get(f"lon_{lead}"), row.get(f"vmax_{lead}")
    return la, lo, v


def _fc(fdict, cid, lead):
    d = fdict.get(cid)
    if not d or not d.get("forecast"):
        return None
    return d["forecast"].get(str(lead))


def evaluate_methods(cases: pd.DataFrame, methods: dict[str, dict],
                     homogeneous: bool = True, case_filter: dict | None = None,
                     sample_out: dict | None = None) -> pd.DataFrame:
    """methods: name -> forecasts dict (from load_forecasts). Returns tidy
    DataFrame with per-method per-lead metrics on the homogeneous sample.
    case_filter: optional {lead: set(case_id)} restriction (analysis manifest);
    sample_out: if a dict is passed, it is filled with {lead: [(cid, sid)...]}
    for the realized sample (to freeze/emit the manifest)."""
    rows = []
    for lead in VERIF:
        # homogeneous mask: truth exists and every method has a forecast
        sample = []
        allowed = case_filter.get(lead) if case_filter else None
        for _, r in cases.iterrows():
            la, lo, v = _truth(r, lead)
            if not (np.isfinite(la) and np.isfinite(lo)):
                continue
            cid = case_id(r)
            if allowed is not None and cid not in allowed:
                continue
            entries = {m: _fc(fd, cid, lead) for m, fd in methods.items()}
            if homogeneous and any(e is None for e in entries.values()):
                continue
            sample.append((r, cid, entries))
        if sample_out is not None:
            sample_out[lead] = [(cid, r["sid"]) for r, cid, _ in sample]
        for m in methods:
            te, ve, vb, sids = [], [], [], []
            for r, cid, entries in sample:
                e = entries[m]
                if e is None:
                    continue
                la, lo, v = _truth(r, lead)
                te.append(float(gc_distance_km(la, lo, e["lat"], e["lon"])))
                sids.append(r["sid"])
                vm = e.get("vmax")
                if vm is not None and np.isfinite(v):
                    ve.append(abs(vm - v))
                    vb.append(vm - v)
            rows.append(dict(method=m, lead=lead, n=len(te),
                             track_km=float(np.mean(te)) if te else np.nan,
                             track_med_km=float(np.median(te)) if te else np.nan,
                             vmax_mae_kt=float(np.mean(ve)) if ve else np.nan,
                             vmax_bias_kt=float(np.mean(vb)) if vb else np.nan,
                             n_storms=len(set(sids))))
    return pd.DataFrame(rows)


def paired_test(cases: pd.DataFrame, m_a: dict, m_b: dict, lead: int,
                metric: str = "track", allowed: set | None = None) -> dict:
    """Paired t-test A vs B with effective sample size adjusted for serial
    correlation (forecasts <18 h apart on the same storm are dependent).
    allowed: optional case_id restriction (analysis manifest)."""
    diffs, keys = [], []
    for _, r in cases.iterrows():
        la, lo, v = _truth(r, lead)
        if not np.isfinite(la):
            continue
        cid = case_id(r)
        if allowed is not None and cid not in allowed:
            continue
        ea, eb = _fc(m_a, cid, lead), _fc(m_b, cid, lead)
        if ea is None or eb is None:
            continue
        if metric == "track":
            da = float(gc_distance_km(la, lo, ea["lat"], ea["lon"]))
            db = float(gc_distance_km(la, lo, eb["lat"], eb["lon"]))
        else:
            if ea.get("vmax") is None or eb.get("vmax") is None or not np.isfinite(v):
                continue
            da, db = abs(ea["vmax"] - v), abs(eb["vmax"] - v)
        diffs.append(da - db)
        keys.append((r["sid"], pd.Timestamp(r["init"])))
    d = np.array(diffs)
    if len(d) < 10:
        return dict(n=len(d), p=np.nan)
    # lag-1 autocorrelation within storms (12-h spacing)
    order = np.argsort([f"{s}_{t.isoformat()}" for s, t in keys])
    ds = d[order]
    ks = [keys[i] for i in order]
    pairs = [(ds[i], ds[i + 1]) for i in range(len(ds) - 1)
             if ks[i][0] == ks[i + 1][0] and (ks[i + 1][1] - ks[i][1]).total_seconds() <= 12 * 3600]
    if len(pairs) > 5:
        a = np.array(pairs)
        r1 = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        r1 = max(min(r1, 0.95), 0.0)
    else:
        r1 = 0.0
    n_eff = len(d) * (1 - r1) / (1 + r1)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(max(n_eff, 2)))
    from scipy import stats
    p = float(2 * stats.t.sf(abs(t), df=max(n_eff - 1, 2)))
    return dict(n=len(d), n_eff=float(n_eff), mean_diff=float(d.mean()), t=float(t), p=p)


def ri_prob_calibration():
    """Platt coefficients + CSI-optimal threshold fit on the calib seasons."""
    import os
    from .config import work_dir
    path = os.path.join(work_dir("models"), "ri_calibration.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def calibrate_ri_prob(p, coeffs) -> float:
    p = min(max(float(p), 0.01), 0.99)
    x = np.log(p / (1 - p))
    z = coeffs["a"] * x + coeffs["b"]
    return float(1 / (1 + np.exp(-z)))


def ri_scores(cases: pd.DataFrame, methods: dict[str, dict],
              threshold: float = 30.0, use_prob: bool = False,
              prob_threshold: float | None = None,
              calibrate: bool = True) -> pd.DataFrame:
    """RI contingency at 24 h: event = Vmax(+24) - Vmax(0) >= threshold.

    With use_prob, a method that stores an explicit `ri24_prob` per case is
    scored by its (Platt-calibrated, if available) probability against the
    CSI-optimal threshold chosen on the calibration seasons.
    """
    coeffs = ri_prob_calibration() if (use_prob and calibrate) else None
    if prob_threshold is None:
        prob_threshold = (coeffs or {}).get("threshold", 0.5)
    rows = []
    for m, fd in methods.items():
        hits = miss = fa = cn = 0
        for _, r in cases.iterrows():
            v0 = r["vmax"]
            vt = r.get("vmax_24")
            if not np.isfinite(vt):
                continue
            cid = case_id(r)
            e = _fc(fd, cid, 24)
            if e is None or e.get("vmax") is None:
                continue
            obs = (vt - v0) >= threshold
            p = fd.get(cid, {}).get("ri24_prob") if use_prob else None
            if p is not None:
                if coeffs:
                    p = calibrate_ri_prob(p, coeffs)
                pred = float(p) >= prob_threshold
            else:
                pred = (e["vmax"] - v0) >= threshold
            hits += obs and pred
            miss += obs and not pred
            fa += pred and not obs
            cn += (not obs) and (not pred)
        n_ev = hits + miss
        rows.append(dict(method=m, events=n_ev, hits=hits, misses=miss, false_alarms=fa,
                         pod=hits / n_ev if n_ev else np.nan,
                         far=fa / (hits + fa) if (hits + fa) else np.nan,
                         csi=hits / (hits + miss + fa) if (hits + miss + fa) else np.nan))
    return pd.DataFrame(rows)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment (family-wise error control)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def bootstrap_table(cases: pd.DataFrame, methods: dict[str, dict],
                    n_boot: int = 2000, seed: int = 0,
                    ref: str | None = None,
                    case_filter: dict | None = None) -> pd.DataFrame:
    """Storm-level (cluster) bootstrap CIs for mean track error and Vmax MAE
    on the homogeneous sample; optional paired differences vs a reference
    method with a bootstrap two-sided p-value."""
    rng = np.random.default_rng(seed)
    rows = []
    for lead in VERIF:
        allowed = case_filter.get(lead) if case_filter else None
        sids, errs = [], {m: {"t": [], "v": []} for m in methods}
        for _, r in cases.iterrows():
            la, lo, v = _truth(r, lead)
            if not (np.isfinite(la) and np.isfinite(lo)):
                continue
            cid = case_id(r)
            if allowed is not None and cid not in allowed:
                continue
            entries = {m: _fc(fd, cid, lead) for m, fd in methods.items()}
            if any(e is None for e in entries.values()):
                continue
            if not np.isfinite(v) or any(e.get("vmax") is None for e in entries.values()):
                vv = None
            else:
                vv = v
            sids.append(r["sid"])
            for m, e in entries.items():
                errs[m]["t"].append(float(gc_distance_km(la, lo, e["lat"], e["lon"])))
                errs[m]["v"].append(abs(e["vmax"] - vv) if vv is not None else np.nan)
        sids = np.array(sids)
        T = {m: np.array(errs[m]["t"]) for m in methods}
        V = {m: np.array(errs[m]["v"]) for m in methods}
        uniq = np.array(sorted(set(sids)))
        sid_idx = {s: np.where(sids == s)[0] for s in uniq}
        bt = {m: [] for m in methods}
        bv = {m: [] for m in methods}
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([sid_idx[s] for s in pick])
            for m in methods:
                bt[m].append(float(np.mean(T[m][idx])))
                bv[m].append(float(np.nanmean(V[m][idx])))
        for m in methods:
            at, av = np.array(bt[m]), np.array(bv[m])
            row = dict(method=m, lead=lead, n=len(sids), n_storms=len(uniq),
                       track_km=float(np.mean(T[m])),
                       track_lo=float(np.percentile(at, 2.5)),
                       track_hi=float(np.percentile(at, 97.5)),
                       vmax_mae_kt=float(np.nanmean(V[m])),
                       vmax_lo=float(np.percentile(av, 2.5)),
                       vmax_hi=float(np.percentile(av, 97.5)))
            if ref and ref in methods and m != ref:
                dt = at - np.array(bt[ref])
                dv = av - np.array(bv[ref])
                row.update(
                    dtrack=float(np.mean(T[m]) - np.mean(T[ref])),
                    dtrack_lo=float(np.percentile(dt, 2.5)),
                    dtrack_hi=float(np.percentile(dt, 97.5)),
                    dtrack_p=float(2 * min((dt > 0).mean(), (dt < 0).mean())),
                    dvmax=float(np.nanmean(V[m]) - np.nanmean(V[ref])),
                    dvmax_lo=float(np.percentile(dv, 2.5)),
                    dvmax_hi=float(np.percentile(dv, 97.5)),
                    dvmax_p=float(2 * min((dv > 0).mean(), (dv < 0).mean())))
            rows.append(row)
    return pd.DataFrame(rows)


def skill_vs(baseline: str, table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for lead in VERIF:
        base_t = table[(table.method == baseline) & (table.lead == lead)]["track_km"]
        base_v = table[(table.method == baseline) & (table.lead == lead)]["vmax_mae_kt"]
        if base_t.empty:
            continue
        bt, bv = float(base_t.iloc[0]), float(base_v.iloc[0])
        sel = out.lead == lead
        out.loc[sel, "track_skill_pct"] = 100 * (bt - out.loc[sel, "track_km"]) / bt
        out.loc[sel, "vmax_skill_pct"] = 100 * (bv - out.loc[sel, "vmax_mae_kt"]) / bv
    return out
