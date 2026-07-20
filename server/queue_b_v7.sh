#!/usr/bin/env bash
# Queue B (GPU4:8602): featmatch full test -> stronger-prior calib run ->
# fit its own shrinkage -> stronger-prior test run
set -e
cd "$(dirname "$0")/.."
source server/env.sh
URL=http://localhost:8602/v1
W=$STORMDESK_WORK

echo "=== B1: featmatch (features + decisive prompt, no examples) ==="
"$PY" scripts/06_run_agent.py --split test --mode featmatch --tag qwen14b \
  --workers 16 --llm-url $URL --llm-model qwen2.5-14b
"$PY" scripts/06_run_agent.py --split test --mode featmatch --tag qwen14b \
  --workers 8 --llm-url $URL --llm-model qwen2.5-14b

echo "=== B2: stronger-prior calibration run (identity shrinkage) ==="
STORMDESK_OFFICE_CALIB=/tmp/nonexistent_calib.json \
"$PY" scripts/06_run_agent.py --split calib --mode full --tag qwen14b_strongprior \
  --prior-file $W/forecasts/calib_aiwp_postproc.jsonl \
  --workers 16 --llm-url $URL --llm-model qwen2.5-14b
STORMDESK_OFFICE_CALIB=/tmp/nonexistent_calib.json \
"$PY" scripts/06_run_agent.py --split calib --mode full --tag qwen14b_strongprior \
  --prior-file $W/forecasts/calib_aiwp_postproc.jsonl \
  --workers 8 --llm-url $URL --llm-model qwen2.5-14b

echo "=== B3: fit stronger-prior shrinkage ==="
"$PY" scripts/09_fit_office_calibration.py --tag agent_full_qwen14b_strongprior \
  --split calib --out $W/models/office_calibration_strongprior.json

echo "=== B4: stronger-prior test run (own shrinkage) ==="
STORMDESK_OFFICE_CALIB=$W/models/office_calibration_strongprior.json \
"$PY" scripts/06_run_agent.py --split test --mode full --tag qwen14b_strongprior \
  --prior-file $W/forecasts/test_aiwp_postproc.jsonl \
  --workers 16 --llm-url $URL --llm-model qwen2.5-14b
STORMDESK_OFFICE_CALIB=$W/models/office_calibration_strongprior.json \
"$PY" scripts/06_run_agent.py --split test --mode full --tag qwen14b_strongprior \
  --prior-file $W/forecasts/test_aiwp_postproc.jsonl \
  --workers 8 --llm-url $URL --llm-model qwen2.5-14b

echo "QUEUE B DONE"
