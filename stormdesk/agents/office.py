"""StormDesk meeting protocol (v2): brief -> specialists -> audit -> synthesis.

Adjudication contract: the office computes a skill-prior weighted consensus;
agents return bounded multiplicative trust factors (track) and bounded deltas
on the bias-corrected consensus prior (intensity). The Physics Auditor can
propose bounded intensity corrections; the Chief accepts/rejects them, sets
the RI probability and writes the discussion. Track is never free-generated.
"""
from __future__ import annotations

import json

import numpy as np

from ..geo import destination, gc_distance_km, wrap_lon, motion_uv_kmh
from ..baselines import consensus_weighted, _members_at, _mean_position
from .auditor import audit_forecast, apply_hard_caps
from .llm import LLMClient, ask_json
from . import prompts

VERIF = [24, 48, 72]
SSHS = [(137, "cat 5"), (113, "cat 4"), (96, "cat 3"), (83, "cat 2"),
        (64, "cat 1"), (34, "tropical storm"), (0, "tropical depression")]


def _cat(v):
    for thr, name in SSHS:
        if v >= thr:
            return name
    return "depression"


def _fmt_pos(lat, lon):
    return f"{abs(lat):.1f}{'N' if lat >= 0 else 'S'} {abs(lon):.1f}{'E' if lon >= 0 else 'W'}"


def prior_weights(guidance: dict, calib, lead: int) -> dict:
    ms = _members_at(guidance, lead)
    if not ms:
        return {}
    w = {m: (calib.weight(m, lead) if calib else 0.0) for m in ms}
    tot = sum(w.values())
    if tot <= 1e-12:
        w = {m: 1.0 for m in ms}
        tot = float(len(ms))
    return {m: v / tot for m, v in w.items()}


def consensus_prior_vmax(guidance: dict, calib) -> dict:
    """Bias-corrected weighted-consensus intensity per lead (the prior)."""
    cw = consensus_weighted(guidance, calib, correct_bias=True)
    out = {}
    for l in VERIF:
        e = cw.get(str(l))
        if e and e.get("vmax") is not None:
            out[str(l)] = float(e["vmax"])
    return out


# ---------------------------------------------------------------------------
# briefing construction
# ---------------------------------------------------------------------------
def build_briefing(case: dict, diag: dict, sat: dict | None, guidance: dict,
                   calib, analogs: list[dict], analog_summary: dict,
                   include_diag=True, include_analogs=True,
                   prior_vmax: dict | None = None, anonymize: bool = False,
                   gate_advice: dict | None = None,
                   feature_block: bool = False) -> str:
    hist = case["history"]
    u, v = motion_uv_kmh(hist[-3]["lat"], hist[-3]["lon"], hist[-1]["lat"], hist[-1]["lon"], 12.0)
    speed = float(np.hypot(u, v))
    mdir = float((np.degrees(np.arctan2(u, v)) + 360) % 360)
    dv12 = case["vmax"] - hist[-3]["vmax"] if hist[-3]["vmax"] is not None else 0.0
    dv24 = case["vmax"] - hist[0]["vmax"] if hist[0]["vmax"] is not None else 0.0

    L = []
    if anonymize:
        # contamination control: no storm name, year shifted by -13 with the
        # calendar date (season phase) preserved
        import pandas as pd
        t = pd.Timestamp(case["init"])
        try:
            t2 = t.replace(year=t.year - 13)
        except ValueError:  # Feb 29
            t2 = t.replace(year=t.year - 13, day=28)
        L.append(f"STORM UNNAMED ({case['basin']} basin, {int(case['season']) - 13}); "
                 f"forecast initialized {t2}Z.")
    else:
        L.append(f"STORM {case['name']} ({case['basin']} basin, {case['season']}); "
                 f"forecast initialized {case['init']}Z.")
    L.append(f"Current: {_fmt_pos(case['lat'], case['lon'])}, Vmax {case['vmax']:.0f} kt "
             f"({_cat(case['vmax'])}), motion {mdir:.0f} deg at {speed:.0f} km/h, "
             f"trend {dv12:+.0f} kt/12h, {dv24:+.0f} kt/24h."
             + (f" Distance to land {case['dist2land']:.0f} km." if np.isfinite(case.get('dist2land', np.nan)) else ""))

    if include_diag and diag:
        d = diag
        env = [f"deep-layer shear {d.get('shear_kt')} kt from {d.get('shear_dir_deg')} deg",
               f"steering {d.get('steering_dir_deg')} deg at {d.get('steering_speed_kmh')} km/h"]
        if d.get("sst_c") is not None:
            env.append(f"SST {d['sst_c']} C")
        if d.get("rh_mid_pct") is not None:
            env.append(f"mid-level RH {d['rh_mid_pct']}%")
        if d.get("mpi_kt") is not None:
            env.append(f"MPI {d['mpi_kt']:.0f} kt (POT {d.get('pot_kt', 0):+.0f} kt)")
        if d.get("div200_1e7") is not None:
            env.append(f"200-hPa divergence {d['div200_1e7']}e-7 s-1")
        L.append("ENVIRONMENT: " + "; ".join(str(x) for x in env) + ".")
        if sat:
            L.append(f"SATELLITE IR: min BT {sat.get('bt_min_k')} K, core mean {sat.get('bt_core_mean_k')} K, "
                     f"cold(<208K) fraction {sat.get('cold_frac_208k')}, "
                     f"quadrant asymmetry {sat.get('quadrant_bt_std_k')} K.")

    L.append("GUIDANCE (per lead: position / raw vmax kt / bias-corrected vmax kt):")
    for m, fc in guidance.items():
        if not fc:
            continue
        row = [f"  {m:12s}"]
        for l in VERIF:
            e = fc.get(str(l)) or fc.get(l)
            if e is None:
                row.append(f"{l}h: --")
                continue
            vm = e.get("vmax", e.get("vmax_kt"))
            if vm is not None and calib is not None:
                vc = calib.correct_vmax(m, l, vm)
                row.append(f"{l}h: {_fmt_pos(e['lat'], e['lon'])} {vm:.0f}/{vc:.0f}kt")
            else:
                row.append(f"{l}h: {_fmt_pos(e['lat'], e['lon'])} {vm if vm is None else round(vm)}kt")
        L.append(" | ".join(row))

    for l in VERIF:
        ms = _members_at(guidance, l)
        if len(ms) >= 2:
            lat_c, lon_c = _mean_position(list(ms.values()))
            spread = float(np.mean([gc_distance_km(e["lat"], e["lon"], lat_c, lon_c) for e in ms.values()]))
            vs = [e.get("vmax", e.get("vmax_kt")) for e in ms.values()]
            vs = [x for x in vs if x is not None]
            L.append(f"SPREAD {l}h: mean track deviation {spread:.0f} km; vmax range "
                     f"{min(vs):.0f}-{max(vs):.0f} kt." if vs else
                     f"SPREAD {l}h: mean track deviation {spread:.0f} km.")

    if calib is not None:
        L.append("MEMBER SKILL (2018-2020 fit; mean track error km @24/48/72h; vmax bias kt @24h):")
        for m in guidance:
            t = calib.table.get(m, {})
            if not t:
                continue
            r = [t.get(str(l), {}).get("track_mae") for l in VERIF]
            b = t.get("24", {}).get("v_bias")
            rs = "/".join("--" if x is None else f"{x:.0f}" for x in r)
            L.append(f"  {m:12s} track {rs}; vmax bias {'--' if b is None else f'{b:+.0f}'} kt")
        pw = {l: prior_weights(guidance, calib, l) for l in VERIF}
        for l in VERIF:
            if pw[l]:
                L.append(f"PRIOR WEIGHTS {l}h: " + ", ".join(
                    f"{m} {w:.2f}" for m, w in sorted(pw[l].items(), key=lambda kv: -kv[1])))

    if feature_block:
        # representation-matched control: the exact engineered member features
        # the supervised gate consumes, computed by the same code path
        from ..combiner import member_rows
        L.append("MEMBER TRACK DIAGNOSTICS (computed for you; motion dev = deviation "
                 "of implied 0-24h motion from observed motion):")
        per_member = {}
        for l in VERIF:
            for row in member_rows(case, {}, guidance, calib, l, list(guidance)):
                per_member.setdefault(row["member"], {})[l] = row
        for m, rl in per_member.items():
            r0 = rl.get(24) or next(iter(rl.values()))
            dc = "/".join(f"{rl[l]['dist_cluster']:.0f}" if l in rl else "--"
                          for l in VERIF)
            dev = r0.get("motion_dev_deg")
            rat = r0.get("speed_ratio")
            dsp = r0.get("disp24")
            L.append(f"  {m:12s} motion dev "
                     f"{'--' if dev is None or not np.isfinite(dev) else f'{dev:.0f} deg'}, "
                     f"speed ratio {'--' if rat is None or not np.isfinite(rat) else f'{rat:.2f}'}, "
                     f"24h displacement {'--' if dsp is None or not np.isfinite(dsp) else f'{dsp:.0f} km'}; "
                     f"cluster distance 24/48/72h: {dc} km")

    if gate_advice:
        L.append("LEARNED COMBINER SUGGESTION (a supervised model's recommended member "
                 "weights for THIS case; use as extra evidence for your trust factors):")
        for l in VERIF:
            gw = gate_advice.get(str(l))
            if gw:
                L.append(f"  {l}h: " + ", ".join(
                    f"{m} {w:.2f}" for m, w in sorted(gw.items(), key=lambda kv: -kv[1])))

    if prior_vmax:
        L.append("CONSENSUS PRIOR Vmax (bias-corrected, your adjustment baseline): "
                 + ", ".join(f"{l}h: {prior_vmax[str(l)]:.0f} kt" for l in VERIF
                             if str(l) in prior_vmax))

    if include_analogs and analogs:
        L.append(f"HISTORICAL ANALOGS (top {len(analogs)} similar storms 1980-2015; "
                 f"their observed intensity changes):")
        for a in analogs[:8]:
            o = a["outcome"]
            dvs = "/".join(f"{o.get(f'dv_{l}', None):+.0f}" if o.get(f"dv_{l}") is not None else "--"
                           for l in VERIF)
            L.append(f"  {a['name']} ({a['time'][:10]}, {a['basin']}) sim={a['similarity']}: "
                     f"dV24/48/72 = {dvs} kt from {a['vmax']:.0f} kt")
        s = analog_summary
        if s:
            L.append(f"  ANALOG SUMMARY: median dV24 {s.get('dv24_median', 0):+.0f} kt "
                     f"(IQR {s.get('dv24_p25', 0):+.0f}..{s.get('dv24_p75', 0):+.0f}); "
                     f"median dV48 {s.get('dv48_median', 0):+.0f} kt; "
                     f"RI(>=+30kt/24h) rate {s.get('ri24_rate', 0):.0%}.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def assemble_track(guidance: dict, calib, trust: dict, nudge: dict | None) -> dict:
    out = {}
    for l in VERIF:
        ms = _members_at(guidance, l)
        if not ms:
            continue
        prior = prior_weights(guidance, calib, l)
        w = {}
        for m in ms:
            t = trust.get(str(l), {}).get(m, 1.0)
            try:
                t = float(t)
            except (TypeError, ValueError):
                t = 1.0
            w[m] = prior.get(m, 0.0) * float(np.clip(t, 0.25, 4.0))
        tot = sum(w.values())
        if tot <= 1e-12:
            w = prior
            tot = sum(w.values()) or 1.0
        ref = list(ms.values())[0]["lon"]
        lat = sum(ms[m]["lat"] * w[m] for m in ms) / tot
        lon_rel = sum(((ms[m]["lon"] - ref + 540) % 360 - 180) * w[m] for m in ms) / tot
        lon = float(wrap_lon(ref + lon_rel).item())
        if nudge:
            nd = nudge.get(str(l))
            if nd and nd.get("km"):
                km = float(np.clip(nd["km"], 0, 60))
                la2, lo2 = destination(lat, lon, float(nd.get("bearing_deg", 0)), km)
                lat, lon = float(la2), float(lo2)
        out[str(l)] = dict(lat=round(lat, 2), lon=round(lon, 2))
    return out


_OFFICE_CALIB_CACHE: dict = {"loaded": False, "coeffs": {}}


def office_calibration() -> dict:
    """Per-lead affine shrinkage of the specialist delta, fit on the
    calibration seasons (v = prior + a*delta + b). Empty -> identity."""
    if not _OFFICE_CALIB_CACHE["loaded"]:
        import os
        from ..config import work_dir
        path = os.environ.get("STORMDESK_OFFICE_CALIB") or \
            os.path.join(work_dir("models"), "office_calibration.json")
        if os.path.exists(path):
            with open(path) as f:
                _OFFICE_CALIB_CACHE["coeffs"] = json.load(f)
        _OFFICE_CALIB_CACHE["loaded"] = True
    return _OFFICE_CALIB_CACHE["coeffs"]


def _apply_deltas(prior_vmax: dict, deltas: dict, bound: float = 25.0) -> dict:
    coeffs = office_calibration()
    out = {}
    for l in VERIF:
        p = prior_vmax.get(str(l))
        if p is None:
            continue
        d = deltas.get(str(l), 0.0)
        try:
            d = float(d)
        except (TypeError, ValueError):
            d = 0.0
        d = float(np.clip(d, -bound, bound))
        c = coeffs.get(str(l))
        if c:
            d = float(c.get("a", 1.0)) * d + float(c.get("b", 0.0))
        out[str(l)] = float(p + d)
    return out


# ---------------------------------------------------------------------------
# the meeting
# ---------------------------------------------------------------------------
def run_office(case: dict, diag: dict, sat: dict | None, guidance: dict, calib,
               analogs: list[dict], analog_summary: dict, llm: LLMClient,
               mode: str = "full", sst_crop=None, anonymize: bool = False,
               gate_advice: dict | None = None,
               prior_override: dict | None = None,
               fewshot: list | None = None) -> dict:
    """mode: full | no_analogs | no_auditor | no_diagnostics | single |
    single_refine | featmatch | featmatch_fs (featmatch* = the full office with
    the gate's engineered member features in the briefing and the decisive
    track prompt; _fs additionally prepends worked examples to the track call).
    prior_override: {lead: vmax} replacing the bias-corrected consensus prior.
    """
    include_diag = mode != "no_diagnostics"
    include_analogs = mode != "no_analogs"
    feature_block = mode.startswith("featmatch")
    prior_vmax = ({k: float(v) for k, v in prior_override.items()}
                  if prior_override else consensus_prior_vmax(guidance, calib))
    briefing = build_briefing(case, diag if include_diag else {},
                              sat if include_diag else None, guidance, calib,
                              analogs if include_analogs else [],
                              analog_summary if include_analogs else {},
                              include_diag, include_analogs, prior_vmax,
                              anonymize=anonymize, gate_advice=gate_advice,
                              feature_block=feature_block)
    init = dict(lat=case["lat"], lon=case["lon"], vmax=case["vmax"])
    transcript = {"briefing": briefing, "mode": mode, "prior_vmax": prior_vmax}

    if mode in ("free", "free_schema", "free_delta"):
        # contract ablation: same briefing, direct free generation of the
        # forecast numbers; no anchors, no bounds, no calibration, no caps
        # (only coordinate/intensity sanity clamps). Three interface variants:
        #   free         absolute lat/lon, no sign convention stated
        #   free_schema  absolute lat/lon, explicit signed-decimal convention
        #   free_delta   bearing + distance from the current position
        sysp = {"free": prompts.FREE_AGENT_SYSTEM,
                "free_schema": prompts.FREE_SCHEMA_SYSTEM,
                "free_delta": prompts.FREE_DELTA_SYSTEM}[mode]
        js = ask_json(llm, sysp, briefing)
        final = {}
        for l in VERIF:
            e = (js.get("forecast") or {}).get(str(l))
            if not isinstance(e, dict):
                continue
            try:
                if mode == "free_delta":
                    brg = float(e["bearing_deg"])
                    dist = float(np.clip(float(e["dist_km"]), 0, 6000))
                    la2, lo2 = destination(case["lat"], case["lon"], brg, dist)
                    lat, lon = float(np.clip(float(la2), -60, 60)), float(wrap_lon(lo2).item())
                else:
                    lat = float(np.clip(float(e["lat"]), -60, 60))
                    lon = float(wrap_lon(float(e["lon"])).item())
                vm = e.get("vmax")
                vm = float(np.clip(float(vm), 10, 200)) if vm is not None else None
            except (TypeError, ValueError, KeyError):
                continue
            final[str(l)] = dict(lat=round(lat, 2), lon=round(lon, 2),
                                 vmax=None if vm is None else round(vm, 1))
        transcript.update(free=js)
        return dict(final=final, ri24_prob=js.get("ri24_prob"), transcript=transcript)

    if mode in ("single", "single_refine"):
        js = ask_json(llm, prompts.SINGLE_AGENT_SYSTEM, briefing)
        transcript["rounds"] = [js]
        if mode == "single_refine":
            # same total budget as the office (5 calls), same intermediate
            # feedback (automated checks on the assembled draft), one agent
            for _ in range(4):
                track = assemble_track(guidance, calib, js.get("trust", {}), js.get("nudge"))
                vmax = _apply_deltas(prior_vmax, js.get("delta_kt", {}))
                draft = {l: dict(track[l], vmax=vmax.get(l)) for l in track}
                draft = apply_hard_caps(draft, init, diag)
                issues = audit_forecast(draft, init, diag, sst_crop)
                user = (briefing
                        + "\n\nYOUR PREVIOUS ANSWER: " + json.dumps(js, default=str)
                        + "\nASSEMBLED DRAFT (after bounds and hard caps): " + json.dumps(draft)
                        + "\nAUTOMATED PHYSICS CHECKS: " + json.dumps(issues))
                js2 = ask_json(llm, prompts.SINGLE_REFINE_SYSTEM, user)
                for k in ("trust", "nudge", "delta_kt", "ri24_prob", "reasoning"):
                    if js2.get(k) is not None:
                        js[k] = js2[k]
                transcript["rounds"].append(js2)
        track = assemble_track(guidance, calib, js.get("trust", {}), js.get("nudge"))
        vmax = _apply_deltas(prior_vmax, js.get("delta_kt", {}))
        final = {l: dict(track[l], vmax=vmax.get(l)) for l in track}
        final = apply_hard_caps(final, init, diag)
        transcript.update(single=js)
        return dict(final=final, ri24_prob=js.get("ri24_prob"), transcript=transcript)

    if mode == "mini":
        # minimal sufficient office (3 calls): skill-prior consensus track
        # (no agenda, no track specialist), intensity + audit + synthesis
        agenda_txt = ""
        tr = {}
        track = assemble_track(guidance, calib, {}, None)
    else:
        # 1. chief agenda
        agenda = ask_json(llm, prompts.CHIEF_AGENDA_SYSTEM, briefing)
        transcript["agenda"] = agenda
        agenda_txt = (f"CHIEF'S AGENDA: {agenda.get('situation','')} "
                      f"Key questions: {'; '.join(str(q) for q in agenda.get('key_questions', []))} "
                      f"Guidance concerns: {agenda.get('guidance_concerns','')}")

        # 2. track specialist -> trust-adjusted consensus
        track_sys = (prompts.TRACK_FEATMATCH_SYSTEM if feature_block
                     else prompts.TRACK_SYSTEM)
        track_user = briefing + "\n\n" + agenda_txt
        if fewshot:
            ex = []
            for i, e in enumerate(fewshot, 1):
                ex.append(f"WORKED EXAMPLE {i} BRIEFING (a past-season storm):\n"
                          f"{e['briefing']}\n"
                          f"WORKED EXAMPLE {i} GOOD ANSWER:\n"
                          f"{json.dumps({'trust': e['trust']})}")
            track_user = ("The following worked examples show what effective "
                          "case-specific trust factors look like on past-season "
                          "storms.\n\n" + "\n\n".join(ex)
                          + "\n\nNOW THE ACTUAL CASE:\n\n" + track_user)
        tr = ask_json(llm, track_sys, track_user)
        transcript["track"] = tr
        track = assemble_track(guidance, calib, tr.get("trust", {}), tr.get("nudge"))

    # 3. intensity specialist -> bounded delta on the consensus prior
    it = ask_json(llm, prompts.INTENSITY_SYSTEM,
                  briefing + ("\n\n" + agenda_txt if agenda_txt else ""))
    transcript["intensity"] = it
    vmax = _apply_deltas(prior_vmax, it.get("delta_kt", {}))

    draft = {l: dict(track[l], vmax=vmax.get(l)) for l in track}
    issues = audit_forecast(draft, init, diag, sst_crop)
    transcript["code_audit"] = issues

    if mode == "no_auditor":
        final = apply_hard_caps(draft, init, diag)
        return dict(final=final, ri24_prob=it.get("ri24_prob"),
                    confidence=it.get("confidence"), transcript=transcript)

    # 4. physics auditor -> bounded intensity adjustments
    audit_input = (briefing + "\n\nDRAFT FORECAST: " + json.dumps(draft)
                   + "\nAUTOMATED CHECKS: " + json.dumps(issues)
                   + "\nSPECIALIST REASONING: track: " + str(tr.get("reasoning", ""))
                   + " | intensity: " + str(it.get("reasoning", "")))
    au = ask_json(llm, prompts.AUDITOR_SYSTEM, audit_input)
    transcript["auditor"] = au

    # 5. chief synthesis: accept/reject auditor adjustments, RI prob, discussion
    chief_input = (briefing + "\n\nDRAFT FORECAST: " + json.dumps(draft)
                   + "\nTRACK SPECIALIST: " + json.dumps({"reasoning": tr.get("reasoning")})
                   + "\nINTENSITY SPECIALIST: " + json.dumps(
                       {k: it.get(k) for k in ("reasoning", "ri24_prob", "confidence")})
                   + "\nAUDITOR: " + json.dumps(au)
                   + "\nAUTOMATED CHECKS: " + json.dumps(issues))
    ch = ask_json(llm, prompts.CHIEF_FINAL_SYSTEM, chief_input)
    transcript["chief"] = ch

    final = {l: dict(e) for l, e in draft.items()}
    accepted = ch.get("accept_adjustments") or {}
    for iss in (au.get("issues") or []):
        l = str(iss.get("lead"))
        adj = iss.get("adjust_kt")
        if l in final and adj is not None and accepted.get(l, True):
            try:
                final[l]["vmax"] = float(final[l]["vmax"]) + float(np.clip(float(adj), -20, 20))
            except (TypeError, ValueError):
                pass
    final = apply_hard_caps(final, init, diag)
    transcript["final_audit"] = audit_forecast(final, init, diag, sst_crop)

    return dict(final=final, ri24_prob=ch.get("ri24_prob", it.get("ri24_prob")),
                confidence=ch.get("confidence", it.get("confidence")),
                discussion=ch.get("discussion"), transcript=transcript)
