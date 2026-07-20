"""Deterministic physics-rule corrector (no LLM).

Mechanically applies the same rules the Physics Auditor is prompted with, as
bounded (+-20 kt) fixes, followed by the usual hard caps. Used to test whether
the LLM auditor adds anything beyond a rule-executing post-processor, and to
strengthen the static hybrid baseline.

Rules (mirroring auditor.audit_forecast and the RI commit rule in the
intensity prompt):
- intensification-rate envelope: pull rates > 50 kt/24h back to the envelope
- MPI ceiling: pull vmax above MPI+15 back to MPI+15
- cold SST (< 24 C): no intensification
- high shear (> 30 kt): cap the 0-24 h rate at +15 kt/24h
- RI commit: if analog RI rate >= 0.3, shear < 15 kt, SST >= 28.5 C and
  POT >= 40 kt, raise the 24-h forecast toward init+30 kt (bounded +20)
"""
from __future__ import annotations

import numpy as np

from .agents.auditor import apply_hard_caps


def ri_rule_fires(diag: dict, analog_summary: dict) -> bool:
    shear = diag.get("shear_kt")
    sst = diag.get("sst_c")
    pot = diag.get("pot_kt")
    return (analog_summary.get("ri24_rate", 0.0) is not None
            and (analog_summary.get("ri24_rate") or 0.0) >= 0.3
            and shear is not None and shear < 15.0
            and sst is not None and sst >= 28.5
            and pot is not None and pot >= 40.0)


def deterministic_corrector(forecast: dict, init: dict, diag: dict,
                            analog_summary: dict, bound: float = 20.0) -> dict:
    fc = {str(l): dict(forecast.get(str(l), forecast.get(l)))
          for l in sorted(int(k) for k in forecast.keys())}
    mpi = diag.get("mpi_kt")
    sst = diag.get("sst_c")
    shear = diag.get("shear_kt")
    ri = ri_rule_fires(diag, analog_summary)

    prev_v, prev_l = float(init["vmax"]), 0
    for l in sorted(int(k) for k in fc):
        e = fc[str(l)]
        if e.get("vmax") is None:
            prev_l = l
            continue
        v = float(e["vmax"])
        dt = max(l - prev_l, 1e-9)
        rate = (v - prev_v) * 24.0 / dt
        targets = []
        # overshoot rules
        if rate > 50.0:
            targets.append(prev_v + 50.0 * dt / 24.0)
        if mpi is not None and v > mpi + 15.0:
            targets.append(mpi + 15.0)
        if sst is not None and sst < 24.0 and rate > 0.0:
            targets.append(prev_v)
        if shear is not None and shear > 30.0 and rate > 15.0 and l <= 24:
            targets.append(prev_v + 15.0 * dt / 24.0)
        if targets:
            target = min(targets)
        elif ri and l == 24 and (v - float(init["vmax"])) < 30.0:
            target = float(init["vmax"]) + 30.0
        else:
            target = v
        adj = float(np.clip(target - v, -bound, bound))
        e["vmax"] = round(v + adj, 1)
        prev_v, prev_l = float(e["vmax"]), l
    return apply_hard_caps(fc, init, diag)
