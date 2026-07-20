"""End-to-end office smoke test with a canned LLM (no server needed).

Validates: briefing assembly, agenda/track/intensity/auditor/chief plumbing,
JSON extraction, aggregation, audit, hard caps — on one real test case.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import work_dir
from stormdesk.agents.office import run_office
from stormdesk.analogs import AnalogLibrary, entry_features, summarize_analogs
from stormdesk.baselines import Calibration
from stormdesk.diagnostics import load_crop
from stormdesk.evaluate import case_id
from stormdesk.geo import motion_uv_kmh
from stormdesk.guidance.merge import load_guidance, load_features


class FakeLLM:
    """Returns plausible canned JSON per role, keyed by system prompt."""
    n_calls = 0
    n_prompt_tokens = 0
    n_completion_tokens = 0

    def chat(self, system, user, **kw):
        FakeLLM.n_calls += 1
        if "opening the forecast discussion" in system:
            return json.dumps({"situation": "Mature TC in a moderately favorable environment.",
                               "key_questions": ["RI risk next 24h?", "cross-track spread at 72h"],
                               "guidance_concerns": "AIWP members likely 25-35 kt too weak."})
        if "Track Specialist" in system:
            return json.dumps({"reasoning": "AIWP members consistent with steering; discount cliper.",
                               "trust": {"24": {"pangu": 1.0, "fengwu": 1.0, "gru": 1.0,
                                                "transformer": 1.0, "cliper": 0.5},
                                         "48": {"pangu": 1.5, "fengwu": 1.0},
                                         "72": {"pangu": 1.0, "fengwu": 1.0}},
                               "nudge": {"48": {"bearing_deg": 315, "km": 30}}})
        if "Intensity Specialist" in system:
            return json.dumps({"reasoning": "Favorable shear/SST; analogs support intensification.",
                               "delta_kt": {"24": 12, "48": 18, "72": 5},
                               "ri24_prob": 0.55, "confidence": "medium"})
        if "Physics Auditor" in system:
            return json.dumps({"verdict": "revise",
                               "issues": [{"lead": 48, "field": "vmax",
                                           "problem": "near MPI", "adjust_kt": -8}],
                               "notes": "otherwise coherent"})
        if "closing the forecast discussion" in system:
            return json.dumps({"accept_adjustments": {"48": True},
                               "ri24_prob": 0.5, "confidence": "medium",
                               "discussion": "Canned discussion text."})
        if "expert tropical cyclone forecaster" in system:
            return json.dumps({"reasoning": "canned",
                               "trust": {"24": {"pangu": 1.0, "fengwu": 1.0}},
                               "nudge": {},
                               "delta_kt": {"24": 5, "48": 8, "72": 0},
                               "ri24_prob": 0.3})
        raise ValueError("unknown role: " + system[:60])


def main():
    with open(os.path.join(work_dir("cases"), "test.pkl"), "rb") as f:
        cases = pickle.load(f)
    guidance_all = load_guidance("test")
    feats = load_features("test")
    lib = AnalogLibrary.load()
    try:
        calib = Calibration.load()
    except FileNotFoundError:
        print("NOTE: no calibration yet; using empty table")
        calib = Calibration({})

    # pick a case with guidance already computed
    row = None
    for _, r in cases.iterrows():
        cid = case_id(r)
        if cid in guidance_all and len(guidance_all[cid]) >= 2 and cid in feats:
            row = r
            break
    assert row is not None, "no case with guidance yet"
    cid = case_id(row)
    print("case:", cid, row["name"], row["basin"], "vmax", row["vmax"])
    g = guidance_all[cid]
    # attach DL/cliper forecasts if present
    for m in ("gru", "transformer", "cliper"):
        p = os.path.join(work_dir("forecasts"), f"test_{m}.jsonl")
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    d = json.loads(line)
                    if d["case_id"] == cid:
                        g[m] = d["forecast"]
                        break
    print("members:", list(g))

    ft = feats.get(cid, {})
    diag = ft.get("diag") or {}
    sat = ft.get("sat")
    h = row["history"]
    mu, mv = motion_uv_kmh(h[-3]["lat"], h[-3]["lon"], h[-1]["lat"], h[-1]["lon"], 12.0)
    dv12 = (row["vmax"] - h[-3]["vmax"]) if h[-3]["vmax"] is not None else 0.0
    f = entry_features(row["lat"], row["lon"], row["vmax"], dv12, float(mu), float(mv),
                       diag, row["init"])
    analogs = lib.query(f, row["lat"], row["sid"], k=12) if f else []
    summary = summarize_analogs(analogs, row["vmax"])
    sup = load_crop(row["sid"], row["season"], row["init"], "SUPPLEMENT")
    sst_crop = sup[0] if sup is not None else None

    for mode in ("full", "single", "no_auditor", "no_analogs", "no_diagnostics"):
        res = run_office(dict(row), diag, sat, g, calib, analogs, summary,
                         FakeLLM(), mode=mode, sst_crop=sst_crop)
        print(f"[{mode}] final:", json.dumps(res["final"]))
    print("\nBRIEFING PREVIEW ----------------------------------")
    from stormdesk.agents.office import build_briefing
    print(build_briefing(dict(row), diag, sat, g, calib, analogs, summary)[:3000])


if __name__ == "__main__":
    main()
