# StormDesk

**Do Zero-Shot LLMs Add Case-Specific Skill to Ensemble Forecasting? A Controlled Study of Tropical Cyclones.**

StormDesk is a virtual forecast office of LLM agents for tropical cyclone (TC)
prediction, built deliberately as an *experimental subject*. Every policy, LLM
or not, reads the same tool-computed briefing and acts inside the same bounded
contract; a ladder of static, regime-conditioned, and supervised reference
policies measures what kind of skill each gain represents. The headline result
is negative and controlled: safety comes from the contract, case adaptation
comes from supervision, and the zero-shot LLM's headroom utilization is
statistically indistinguishable from zero.

This repository is the full release promised in the paper: pipeline code, exact
prompts, calibration artifacts, the frozen analysis manifest, every policy's
forecasts and transcripts, and every figure and table generator.

## Repository map

```
stormdesk/           the package: briefing tools, office/agents, contract,
                     combiner, corrector, evaluation (TOST, Holm, bootstrap)
  agents/prompts.py  the exact office prompts quoted in the supplementary
scripts/             numbered pipeline stages (00-32) + figure/export scripts
server/              vLLM launch/sync helpers for a multi-node cluster
docs/                internal design notes
paper/               figure/table generators for the paper
  figures/make_figs_v2.py       matplotlib results figures (Figs. 3-8)
  figures/make_fig{1,2}_ppt.py  overview / office diagrams (PowerPoint COM)
  make_supp_tables.py           supplementary tables from results CSVs
runtime/             released artifacts
  cases/             forecast-cycle tables per split (IBTrACS-derived)
  features/          environmental + satellite diagnostics
  guidance/          merged guidance per cycle
  models/            every statistical anchor: skill/bias profiles, per-pipeline
                     shrinkages, Platt scalings, gates, post-processors,
                     office_calibration*.json, ri_calibration.json, few-shot examples
  forecasts/         every policy's forecasts (test_*.jsonl, calib_*.jsonl)
  transcripts/       full office deliberations for every LLM run
  results/           frozen analysis manifest (test_manifest.json), metrics,
                     significance, equivalence, headroom, RI verification CSVs
```

## The pipeline

```
00_build_cases.py        case tables per era split (train/val/calib/test)
01_extract_features.py   environmental + satellite diagnostics from ERA5/GridSat crops
02_run_aiwp.py           Pangu-Weather / FengWu guidance (GPU, shardable)
03_train_specialists.py  CLIPER + GRU/Transformer specialists (year-split)
04_build_analogs.py      1980-2015 analog library
05_run_baselines.py      persistence/CLIPER/DL forecasts (+ --consensus family)
05b_fit_calibration.py   member skill profiles + intensity bias maps (2018-2020)
06_run_agent.py          the StormDesk office and every LLM policy variant
07_evaluate.py           NHC-style homogeneous verification + RI scores
08-32                    behavior analysis, RI classifiers, learned gate/stack,
                         replay identification, TOST, regime statics, headroom U,
                         few-shot construction, Llama recalibration
```

## Reproducing the paper numbers

All point-forecast means, tests, and confidence intervals are computed on the
frozen analysis manifest (`runtime/results/test_manifest.json`, per-lead case
lists with MD5 hashes). No number is transcribed by hand.

```
export STORMDESK_WORK=$PWD/runtime
python scripts/07_evaluate.py --split test --case-list runtime/results/test_manifest.json
python paper/figures/make_figs_v2.py      # Figures 3-8 (reads results CSVs)
python paper/make_supp_tables.py          # supplementary tables
```

Running the office itself needs a vLLM server
(`server/launch_vllm.sh`; Qwen2.5-7B/14B/72B-AWQ or Llama-3.1-8B) and
`STORMDESK_LLM_URL`; temperature 0 is the recommended deterministic setting.
Regenerating AIWP guidance from scratch needs ERA5 access and roughly 60
GPU-hours; the released `runtime/` artifacts let you skip every GPU stage and
replay the analysis directly.

## Data sources

IBTrACS v04r00 (NOAA), ERA5 (ECMWF/Copernicus), GridSat-B1 (NOAA), and the
released Pangu-Weather, FengWu, Qwen2.5, and Llama-3.1 checkpoints. Raw
reanalysis and satellite archives are not redistributed here; the derived
briefing inputs in `runtime/` are.

## License

MIT (see `LICENSE`).
