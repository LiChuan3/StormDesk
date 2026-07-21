# StormDesk — A Virtual Forecast Office of LLM Agents for Tropical Cyclone Prediction

> This is the original design note, written before the experiments were run.
> It is kept unchanged for provenance. Where it differs from the paper (for
> example, role counts or preliminary literature numbers), the paper and the
> top-level README are authoritative.

## 1. Motivation chain (from literature)

1. Multi-model consensus reduces TC track error by up to 11% at long leads —
   equivalent to >5 years of official forecast improvement (DeMaria et al. 2025,
   AIES / arXiv:2409.06735). But all operational consensus schemes use **static**
   (equal or historical-performance) weights (TCRR 2026 operational AI ensemble).
2. AI weather prediction (AIWP) models match or beat official track guidance but
   have **no intensity skill**: MAE plateaus near 40 kt at 72 h and bias ≈ MAE
   (systematically weak), rooted in ERA5 label bias + MSE smoothing (DeMaria 2025;
   Bonavita 2024 GRL: effective resolution 500–700 km).
3. Vision/foundation models get the physics wrong even when perception is right
   (TC-Bench, ICML 2026) → forecasts need **quantified physical auditing**.
4. Analog forecasting is a century-old skill of human forecasters, never wired
   into an agentic system as a retrieval tool.
5. ECMWF leadership names agentic workflows the next transformation of the
   forecast value chain (Dueben et al. 2026, arXiv:2606.25076).

**Gap**: no existing system combines LLM multi-agent orchestration of multiple
AIWP models + analog retrieval + physics auditing for TC track & intensity
forecasting, evaluated under the NHC homogeneous verification protocol.

## 2. System

Virtual forecast office with 4 agent roles (Virtual Lab, Nature 2025 pattern):

- **Chief Forecaster** (orchestrator): reads the briefing, sets the agenda,
  synthesizes the final forecast + NHC-style discussion.
- **Track Specialist**: per-lead selective weighting of guidance members with
  rationale (JSON weights).
- **Intensity Specialist**: Vmax per lead; uses guidance envelope + analog prior
  + environmental diagnostics; may exceed the envelope with justification (RI).
- **Physics Auditor** (critic): runs the quantified audit tool; flags violations;
  triggers one revision round.

Tools (ChemCrow-style toolbox):
1. `briefing` — storm status/history, motion, IBTrACS truth-so-far.
2. `diagnostics` — from ERA5 80×80 crops: deep-layer shear (200–850), SST under
   core, mid-level RH, steering flow (DLM wind), 200 hPa divergence, POT
   (potential-intensity proxy minus current Vmax), distance to land.
3. `satellite` — GridSat-B1 IR descriptors: min BT, inner-core mean BT, cold
   pixel fraction, axisymmetry.
4. `guidance` — forecast table from members: Pangu, FengWu, FuXi (tracker on
   rolled fields), Transformer/GRU DL specialists, CLIPER5-class statistical.
   Includes per-member per-lead bias profiles fit on 2018–2020 and spread stats.
5. `analogs` — top-K historical analogs from 1980–2015 library (feature-space
   KNN: position, date, intensity, motion, shear, SST, RH, steering) with their
   observed future evolutions and RI frequency.
6. `audit` — quantified physics checks: wind–pressure (Knaff–Zehr), 24-h
   intensification cap vs environment, Kaplan–DeMaria land decay, translation
   speed cap, steering-consistency, smoothness.

## 3. Experimental protocol

- Forecast task: at init t (storm alive, USA_WIND ≥ 34 kt), predict lat/lon/Vmax
  at leads 6..72 h (6-h steps). Verify at 24/48/72 h (homogeneous samples).
- Era split: analog library + DL training 1980–2015; DL val 2016–2017;
  calibration (weights, bias correction) 2018–2020; **test 2021–2022 global**,
  init 00/12Z.
- Truth: IBTrACS v04r00 USA agency (1-min wind, kt).
- Metrics: track great-circle error (km), Vmax MAE (kt) + bias, skill vs
  CLIPER5/SHIFOR-class baseline, RI (ΔV≥+30kt/24h) POD/FAR/CSI, physics
  violation rate. Paired t-test with serial-correlation-adjusted dof (NHC).
- Baselines: persistence, CLIPER5-class, each member, equal-weight consensus,
  performance-weighted consensus, bias-corrected consensus, simple-average of
  everything; ablations: −analogs, −auditor, −diagnostics, single-agent,
  backbone scale (Qwen2.5 7B/14B/72B-AWQ).

## 4. Data & compute map

| Asset | Location | Use |
|---|---|---|
| TC_ERA5 (MSETCD) 1980–2022, 421 GB | node1/2 `/data/yuxiaoning/data/TC_ERA5` | crops: diagnostics, analog features |
| `fengwu_cache` 26 TB, (4,69,721,1440)/init = +6/12/18/24 h | node1/2 | free FengWu guidance ≤24 h; states at +18/+24 seed ONNX continuation to 72 h |
| `era5_npy_cache` 4.8 TB (69,721,1440)/TC-time | node1/2 | AIWP initial conditions |
| Pangu onnx (1/3/6/24 h), FengWu v2 onnx, FuXi short onnx | node1/2 projects | AIWP members |
| IBTrACS v04r00 csv | all nodes | truth + CLIPER + analogs |
| GridSat-B1 crops in TC_ERA5 | node1/2 | satellite tool |
| env `llm-tc` (onnxruntime-gpu 1.23.2, torch cu128) | node1/2 `/data_small/user_envs/yuxiaoning/llm-tc` | inference + pipeline |
| vLLM + Qwen (to deploy) | node2 GPU1 (94 GB free) / node3 GPU5 | agent backbone |
| node3/4 | `/data_hdd/yuxiaoning/projects` | DL retrain, LLM ablations |

## 5. Paper skeleton (Virtual Lab × ChemCrow)

Title: *StormDesk: A Virtual Forecast Office of LLM Agents for Tropical
Cyclone Prediction*. No standalone Related Work (folded into Intro ¶2–3);
results-first. Figures: F1 motivation (test storms map + consensus-vs-static +
intensity-bias evidence), F2 architecture, F3 main error-vs-lead curves,
F4 RI skill, F5 ablations + scale, F6 case study with reasoning excerpt.
