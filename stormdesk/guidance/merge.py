"""Merge per-member guidance shards into per-case guidance dicts."""
from __future__ import annotations

import glob
import json
import os

from ..config import work_dir

MEMBERS_AIWP = ["pangu", "fengwu"]
# cons_aiwp (the equal mean of the AIWP members) is itself a guidance member,
# as consensus aids are in operational practice
MEMBERS_ALL = MEMBERS_AIWP + ["gru", "transformer", "cliper", "cons_aiwp"]


def load_guidance(split: str, members=None) -> dict:
    """Returns {case_id: {member: forecast_dict}}.

    AIWP members live in <work>/guidance/ (rows keyed `member`); the DL and
    statistical members live in <work>/forecasts/ (rows keyed `method`).
    """
    members = members or MEMBERS_ALL
    out: dict = {}
    gdir = work_dir("guidance")
    fdir = work_dir("forecasts")
    for m in members:
        paths = sorted(glob.glob(os.path.join(gdir, f"{split}_{m}_shard*.jsonl")))
        for p in (os.path.join(gdir, f"{split}_{m}.jsonl"),
                  os.path.join(fdir, f"{split}_{m}.jsonl")):
            if os.path.exists(p):
                paths.append(p)
        for path in paths:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if d.get("forecast"):
                        name = d.get("member") or d.get("method") or m
                        out.setdefault(d["case_id"], {})[name] = _norm_fc(d["forecast"])
    return out


def _norm_fc(fc: dict) -> dict:
    """Normalize per-lead entries: tracker output uses vmax_kt, others vmax."""
    for e in fc.values():
        if isinstance(e, dict) and e.get("vmax") is None and e.get("vmax_kt") is not None:
            e["vmax"] = e["vmax_kt"]
    return fc


def load_features(split: str) -> dict:
    path = os.path.join(work_dir("features"), f"{split}.jsonl")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    out[d["case_id"]] = d
    return out
