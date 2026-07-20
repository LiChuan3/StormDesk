"""Central configuration: data roots per node, channel contracts, era splits."""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Channel contract of the 69-channel stacks in era5_npy_cache / fengwu_cache /
# TC_ERA5 crops, established EMPIRICALLY (level-major):
#   Surface: 0 u10, 1 v10, 2 t2m, 3 msl(Pa).
#   Then per level (50 -> 1000 hPa), 5 channels: z, q, u, v, t.
# NOTE: the FengWu ONNX model and its data_mean/data_std use the official
# VARIABLE-major layout [surface, z x13, q x13, u x13, v x13, t x13];
# LEVMAJ_TO_VARMAJ permutes level-major states into that layout.
# ---------------------------------------------------------------------------
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
CH_U10, CH_V10, CH_T2M, CH_MSL = 0, 1, 2, 3
_VARS = ["z", "q", "u", "v", "t"]


def lev_idx(hpa: int) -> int:
    return LEVELS.index(hpa)


def ch_z(hpa: int) -> int:
    return 4 + 5 * lev_idx(hpa) + 0


def ch_q(hpa: int) -> int:
    return 4 + 5 * lev_idx(hpa) + 1


def ch_u(hpa: int) -> int:
    return 4 + 5 * lev_idx(hpa) + 2


def ch_v(hpa: int) -> int:
    return 4 + 5 * lev_idx(hpa) + 3


def ch_t(hpa: int) -> int:
    return 4 + 5 * lev_idx(hpa) + 4


# ---------------------------------------------------------------------------
# The TC_ERA5 80x80 CROPS use the official VARIABLE-major layout instead
# (verified empirically): surface 0-3, then z 4-16, q 17-29, u 30-42,
# v 43-55, t 56-68 (levels 50->1000 within each block).
# ---------------------------------------------------------------------------
def crop_z(hpa: int) -> int:
    return 4 + lev_idx(hpa)


def crop_q(hpa: int) -> int:
    return 17 + lev_idx(hpa)


def crop_u(hpa: int) -> int:
    return 30 + lev_idx(hpa)


def crop_v(hpa: int) -> int:
    return 43 + lev_idx(hpa)


def crop_t(hpa: int) -> int:
    return 56 + lev_idx(hpa)


def _build_perm():
    perm = [0, 1, 2, 3]
    for vv in range(5):          # variable-major var blocks z,q,u,v,t
        for i in range(13):      # levels 50 -> 1000
            perm.append(4 + 5 * i + vv)
    import numpy as _np
    return _np.array(perm)


LEVMAJ_TO_VARMAJ = _build_perm()          # varmaj_state = state[LEVMAJ_TO_VARMAJ]


def varmaj_to_levmaj_perm():
    import numpy as _np
    inv = _np.empty_like(LEVMAJ_TO_VARMAJ)
    inv[LEVMAJ_TO_VARMAJ] = _np.arange(len(LEVMAJ_TO_VARMAJ))
    return inv


VARMAJ_TO_LEVMAJ = varmaj_to_levmaj_perm()


# SUPPLEMENT_data.npy channel contract (verified against supplement_v2.py):
# 0 sst, 1-13 relative humidity r at LEVELS, 14-26 vertical velocity w at LEVELS.
SUP_SST = 0


def sup_r(hpa: int) -> int:
    return 1 + lev_idx(hpa)


def sup_w(hpa: int) -> int:
    return 14 + lev_idx(hpa)


# GRIDSAT_data.npy: 0 irwin (IR window BT, K), 1 irwvp (water vapor BT, K),
# 2 vschn (visible albedo). 286x286 at ~0.07 deg centered on the TC.

KT_PER_MS = 1.943844
DEG2KM = 111.194927  # km per degree of latitude

# ---------------------------------------------------------------------------
# Era splits
# ---------------------------------------------------------------------------
TRAIN_YEARS = (1980, 2015)   # analog library + DL training
VAL_YEARS = (2016, 2017)     # DL early stopping
CALIB_YEARS = (2018, 2020)   # consensus weights, bias profiles
TEST_YEARS = (2021, 2022)    # held-out evaluation

LEADS_H = list(range(6, 73, 6))          # forecast leads
VERIF_LEADS_H = [24, 48, 72]             # homogeneous verification leads
INPUT_STEPS = 4                          # 24 h of 6-h history fed to DL models


# ---------------------------------------------------------------------------
# Per-node data roots
# ---------------------------------------------------------------------------
@dataclass
class Paths:
    tc_era5: str = "/data/yuxiaoning/data/TC_ERA5"
    ibtracs_csv: str = "/data/yuxiaoning/data/TC_ERA5/ibtracs.ALL.list.v04r00.csv"
    era5_cache: str = "/data/yuxiaoning/data/era5_npy_cache"
    fengwu_cache: str = "/data/yuxiaoning/data/fengwu_cache"
    fengwu_onnx: str = "/data/yuxiaoning/projects/fengwu/inference/fengwu_v2.onnx"
    fengwu_mean: str = "/data/yuxiaoning/projects/fengwu/inference/data_mean.npy"
    fengwu_std: str = "/data/yuxiaoning/projects/fengwu/inference/data_std.npy"
    pangu_dir: str = "/data/yuxiaoning/projects/Pangu-Weather"
    fuxi_dir: str = "/data/webset/models/fuxi"
    work: str = "/data/yuxiaoning/projects/stormdesk_runtime"  # outputs (not in git)
    llm_base_url: str = os.environ.get("STORMDESK_LLM_URL", "http://192.168.100.5:8500/v1")
    llm_model: str = os.environ.get("STORMDESK_LLM_MODEL", "qwen2.5-14b")


@dataclass
class NodePaths(Paths):
    pass


def get_paths() -> Paths:
    host = socket.gethostname()
    p = Paths()
    if host in ("node3", "node4"):
        base = "/data_ssd/yuxiaoning/datasets/TC_ERA5"
        p.tc_era5 = base
        p.ibtracs_csv = os.path.join(base, "ibtracs.ALL.list.v04r00.csv")
        p.work = "/data_hdd/yuxiaoning/projects/stormdesk_runtime"
    override = os.environ.get("STORMDESK_WORK")
    if override:
        p.work = override
    return p


def work_dir(*sub: str) -> str:
    p = get_paths()
    d = os.path.join(p.work, *sub)
    os.makedirs(d, exist_ok=True)
    return d
