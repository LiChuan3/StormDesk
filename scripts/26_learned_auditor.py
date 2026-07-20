"""Learned feature-based auditor and post-processed AIWP intensity member
(R3/Q9 and R4/Q4), both fit on the calibration season.

R3: a small gradient-boosted regressor predicts the ideal correction to the
no-auditor intensity draft (truth - draft) from the same diagnostics the LLM
auditor sees (shear, SST, RH, POT, divergence, satellite, analog RI rate, the
draft, the intensity trend). Applied as a drop-in replacement for the LLM
auditor + chief, this isolates the value of the audit *checks/features* from the
LLM text. Compared against: no-auditor, LLM office, affine recalibrator.

R4: a gradient-boosted post-processor maps raw tracker-derived AIWP intensity +
environment -> truth (a proxy for modern AIWP intensity post-processors such as
TCBench-ANN / BaguanCyclone), added as an intensity baseline. It quantifies the
achievable non-agentic intensity skill that raw tracker winds understate.
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk import combiner as CB
from stormdesk.config import work_dir
from stormdesk.evaluate import case_id, load_forecasts
from stormdesk.guidance.merge import load_guidance, load_features

VERIF = [24, 48, 72]

_spec = importlib.util.spec_from_file_location(
    "ri12", os.path.join(os.path.dirname(os.path.abspath(__file__)), "12_ri_baselines.py"))
_ri12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ri12)

_spec2 = importlib.util.spec_from_file_location(
    "aud20", os.path.join(os.path.dirname(os.path.abspath(__file__)), "20_auditor_id.py"))
_aud = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_aud)


def cases_dict(split):
    with open(os.path.join(work_dir("cases"), f"{split}.pkl"), "rb") as f:
        cs = pickle.load(f)
    return {case_id(r): r for _, r in cs.iterrows()}, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-split", default="calib")
    ap.add_argument("--split", default="test")
    ap.add_argument("--case-list", default=None,
                    help="analysis-manifest JSON; restrict the test evaluation "
                         "to the same homogeneous per-lead case sets as Table 1")
    ap.add_argument("--emit-calib-cv", action="store_true",
                    help="also write <calib>_aiwp_postproc.jsonl with "
                         "storm-grouped 5-fold out-of-fold predictions (for use "
                         "as a stronger office prior on the calibration split)")
    args = ap.parse_args()

    allowed = None
    if args.case_list:
        with open(args.case_list) as f:
            mf = json.load(f)
        allowed = {int(l): set(e["case_ids"]) for l, e in mf["leads"].items()}

    # feature matrices shared with the RI classifiers (environment/sat/analog)
    dfc = _ri12.build_matrix(args.calib_split)
    dft = _ri12.build_matrix(args.split)
    Zc, stats = _ri12.impute_standardize(dfc)
    Zt, _ = _ri12.impute_standardize(dft, stats)

    # -------- R3: learned auditor on the no-auditor draft --------
    tr_c = _aud.load_transcripts(args.calib_split, "agent_full_qwen14b")
    tr_t = _aud.load_transcripts(args.split, "agent_full_qwen14b")
    ccd, _ = cases_dict(args.calib_split)
    tcd, tcases = cases_dict(args.split)
    from stormdesk.config import work_dir as _wd
    office_cal = json.load(open(os.path.join(_wd("models"), "office_calibration.json")))

    def draft(cid, tr, lead):
        t = tr.get(cid)
        if t is None:
            return None
        pc = _aud.parse_case(t).get(lead)
        if pc is None:
            return None
        p, d, adj = pc
        a = office_cal[str(lead)]["a"]; b = office_cal[str(lead)]["b"]
        return float(np.clip(p + a * d + b, 15, 185))

    cidx_c = {c: i for i, c in enumerate(dfc.case_id.values)}
    cidx_t = {c: i for i, c in enumerate(dft.case_id.values)}
    rows = {}
    for lead in VERIF:
        Xtr, ytr = [], []
        for cid in dfc.case_id.values:
            r = ccd.get(cid); dr = draft(cid, tr_c, lead)
            if r is None or dr is None:
                continue
            vt = r.get(f"vmax_{lead}")
            if not np.isfinite(vt):
                continue
            Xtr.append(np.concatenate([Zc[cidx_c[cid]], [dr]])); ytr.append(float(vt) - dr)
        model = CB.fit_gbt_reg(np.array(Xtr), np.array(ytr), depth=2, n=150)
        preds, truth, drafts = [], [], []
        for cid in dft.case_id.values:
            r = tcd.get(cid); dr = draft(cid, tr_t, lead)
            if r is None or dr is None:
                continue
            if allowed is not None and cid not in allowed.get(lead, ()):
                continue
            vt = r.get(f"vmax_{lead}")
            if not np.isfinite(vt):
                continue
            corr = float(np.clip(model.predict(np.concatenate([Zt[cidx_t[cid]], [dr]])[None])[0], -25, 25))
            preds.append(dr + corr); truth.append(float(vt)); drafts.append(dr)
        preds, truth, drafts = map(np.array, (preds, truth, drafts))
        rows[lead] = dict(learned_auditor_mae=float(np.mean(np.abs(preds - truth))),
                          learned_auditor_bias=float(np.mean(preds - truth)),
                          n=len(truth))
    print("=== R3 learned feature-based auditor (drop-in for LLM auditor+chief) ===")
    for lead in VERIF:
        print(f"  {lead}h: MAE {rows[lead]['learned_auditor_mae']:.2f} kt, "
              f"bias {rows[lead]['learned_auditor_bias']:+.2f} (n={rows[lead]['n']})")
    print("  (compare Table: no-auditor 12.1/17.0/18.6, LLM office 13.4/17.9/19.1, affine 11.9/16.4/17.4)")

    # -------- R4: post-processed AIWP intensity member --------
    aiwp_c = load_forecasts(os.path.join(work_dir("forecasts"), f"{args.calib_split}_cons_aiwp.jsonl"))
    aiwp_t = load_forecasts(os.path.join(work_dir("forecasts"), f"{args.split}_cons_aiwp.jsonl"))

    def aiwp_v(fc, cid, lead):
        d = fc.get(cid)
        if not d or not d.get("forecast"):
            return None
        e = d["forecast"].get(str(lead))
        return float(e["vmax"]) if e and e.get("vmax") is not None else None

    pp_rows = {}
    out_fc = {}
    calib_cv_fc = {}
    for lead in VERIF:
        Xtr, ytr = [], []
        for cid in dfc.case_id.values:
            r = ccd.get(cid); av = aiwp_v(aiwp_c, cid, lead)
            if r is None or av is None:
                continue
            vt = r.get(f"vmax_{lead}")
            if not np.isfinite(vt):
                continue
            Xtr.append(np.concatenate([Zc[cidx_c[cid]], [av]])); ytr.append(float(vt))
        model = CB.fit_gbt_reg(np.array(Xtr), np.array(ytr), depth=2, n=150)
        preds, truth = [], []
        for cid in dft.case_id.values:
            r = tcd.get(cid); av = aiwp_v(aiwp_t, cid, lead)
            if r is None or av is None:
                continue
            p = float(np.clip(model.predict(np.concatenate([Zt[cidx_t[cid]], [av]])[None])[0], 15, 185))
            e = (aiwp_t.get(cid) or {}).get("forecast", {}).get(str(lead))
            out_fc.setdefault(cid, {})[str(lead)] = dict(lat=e["lat"], lon=e["lon"], vmax=round(p, 1))
            vt = r.get(f"vmax_{lead}")
            if np.isfinite(vt) and (allowed is None or cid in allowed.get(lead, ())):
                preds.append(p); truth.append(float(vt))
        preds, truth = np.array(preds), np.array(truth)
        pp_rows[lead] = dict(mae=float(np.mean(np.abs(preds - truth))),
                             bias=float(np.mean(preds - truth)))

        # out-of-fold calibration-split predictions (stronger-prior experiment)
        if args.emit_calib_cv:
            sid_of = {cid: cid.split("_")[0] for cid in dfc.case_id.values}
            cids_l, X_l, sid_l = [], [], []
            for cid in dfc.case_id.values:
                r = ccd.get(cid); av = aiwp_v(aiwp_c, cid, lead)
                if r is None or av is None:
                    continue
                cids_l.append(cid)
                X_l.append(np.concatenate([Zc[cidx_c[cid]], [av]]))
                sid_l.append(sid_of[cid])
            X_l = np.array(X_l); sid_l = np.array(sid_l)
            uniq = sorted(set(sid_l))
            folds = {s: i % 5 for i, s in enumerate(uniq)}
            fold_of = np.array([folds[s] for s in sid_l])
            for k in range(5):
                tr_m = fold_of != k
                # training targets: truth for the training cids
                ytr_k = []
                keep = []
                for j in np.where(tr_m)[0]:
                    vt = ccd[cids_l[j]].get(f"vmax_{lead}")
                    if np.isfinite(vt):
                        keep.append(j); ytr_k.append(float(vt))
                mk = CB.fit_gbt_reg(X_l[keep], np.array(ytr_k), depth=2, n=150)
                for j in np.where(~tr_m)[0]:
                    p = float(np.clip(mk.predict(X_l[j][None])[0], 15, 185))
                    e = (aiwp_c.get(cids_l[j]) or {}).get("forecast", {}).get(str(lead))
                    if e:
                        calib_cv_fc.setdefault(cids_l[j], {})[str(lead)] = dict(
                            lat=e["lat"], lon=e["lon"], vmax=round(p, 1))
    path = os.path.join(work_dir("forecasts"), f"{args.split}_aiwp_postproc.jsonl")
    with open(path, "w") as f:
        for cid, fc in out_fc.items():
            f.write(json.dumps(dict(case_id=cid, method="aiwp_postproc", forecast=fc)) + "\n")
    if args.emit_calib_cv and calib_cv_fc:
        cpath = os.path.join(work_dir("forecasts"),
                             f"{args.calib_split}_aiwp_postproc.jsonl")
        with open(cpath, "w") as f:
            for cid, fc in calib_cv_fc.items():
                f.write(json.dumps(dict(case_id=cid, method="aiwp_postproc",
                                        forecast=fc)) + "\n")
        print(f"calib CV postproc prior: {len(calib_cv_fc)} -> {cpath}")
    print("\n=== R4 post-processed AIWP intensity (env-conditioned GBT on raw AIWP) ===")
    for lead in VERIF:
        print(f"  {lead}h: MAE {pp_rows[lead]['mae']:.2f} kt, bias {pp_rows[lead]['bias']:+.2f}")
    print(f"  (raw AIWP tracker: 26/27/27 kt MAE; bias-corr consensus: 12.0/16.5/17.9) -> {path}")

    json.dump(dict(learned_auditor=rows, aiwp_postproc=pp_rows),
              open(os.path.join(work_dir("results"), "learned_auditor_postproc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
