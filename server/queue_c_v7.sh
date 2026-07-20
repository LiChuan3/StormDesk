#!/usr/bin/env bash
# Queue C (GPU1:8601): Llama-3.1-8B independent second-family calibration:
# calib-season run (identity shrinkage) -> fit own anchors -> 250-cycle test
# rerun with own anchors -> analysis
set -e
cd "$(dirname "$0")/.."
source server/env.sh
URL=http://localhost:8601/v1
W=$STORMDESK_WORK

echo "=== C1: Llama calibration-season run ==="
STORMDESK_OFFICE_CALIB=/tmp/nonexistent_calib.json \
"$PY" scripts/06_run_agent.py --split calib --mode full --tag llama8b \
  --workers 16 --llm-url $URL --llm-model llama-3.1-8b
STORMDESK_OFFICE_CALIB=/tmp/nonexistent_calib.json \
"$PY" scripts/06_run_agent.py --split calib --mode full --tag llama8b \
  --workers 8 --llm-url $URL --llm-model llama-3.1-8b

echo "=== C2: fit Llama's own shrinkage ==="
"$PY" scripts/09_fit_office_calibration.py --tag agent_full_llama8b \
  --split calib --out $W/models/office_calibration_llama8b.json

echo "=== C3: own-calibrated test rerun (250 cycles) ==="
STORMDESK_OFFICE_CALIB=$W/models/office_calibration_llama8b.json \
"$PY" scripts/06_run_agent.py --split test --mode full --tag llama8b_owncal \
  --limit 250 --workers 16 --llm-url $URL --llm-model llama-3.1-8b
STORMDESK_OFFICE_CALIB=$W/models/office_calibration_llama8b.json \
"$PY" scripts/06_run_agent.py --split test --mode full --tag llama8b_owncal \
  --limit 250 --workers 8 --llm-url $URL --llm-model llama-3.1-8b

echo "=== C4: recalibration analysis ==="
"$PY" scripts/32_llama_recal.py > $W/results/llama_recal.txt 2>&1
tail -20 $W/results/llama_recal.txt

echo "QUEUE C DONE"
