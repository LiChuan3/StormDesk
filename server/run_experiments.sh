#!/usr/bin/env bash
# Full experiment sequence once guidance + calibration inputs are complete and
# the vLLM server is up. Run on node1 or node2 (shared /data).
# Usage: LLM_URL=http://192.168.100.5:8500/v1 bash server/run_experiments.sh [stage]
set -e
cd "$(dirname "$0")/.."
source server/env.sh
LLM_URL=${LLM_URL:-http://192.168.100.5:8500/v1}
LOG=$STORMDESK_WORK/logs
STAGE=${1:-all}

run() { echo "=== $* ==="; "$@" 2>&1 | tail -5; }

if [ "$STAGE" = all ] || [ "$STAGE" = calib ]; then
  run $PY scripts/05b_fit_calibration.py
  run $PY scripts/05_run_baselines.py --split test --methods "" --consensus
fi

if [ "$STAGE" = all ] || [ "$STAGE" = agent ]; then
  for MODE in full no_analogs no_auditor no_diagnostics single; do
    echo "=== agent $MODE (qwen14b) ==="
    STORMDESK_LLM_URL=$LLM_URL $PY scripts/06_run_agent.py \
      --split test --mode $MODE --tag qwen14b --workers 16 \
      --llm-model qwen2.5-14b 2>&1 | tail -3
  done
fi

if [ "$STAGE" = all ] || [ "$STAGE" = eval ]; then
  run $PY scripts/07_evaluate.py --split test --baseline cliper --sig-against cons_bc
  run $PY scripts/08_office_analysis.py --split test --tag agent_full_qwen14b
  for F in fig1bc fig3 fig4 fig5 fig6; do
    run $PY scripts/07_figures.py $F
  done
fi
echo "experiments complete"
