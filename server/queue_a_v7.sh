#!/usr/bin/env bash
# Queue A (GPU0:8600): featmatch_fs full test -> temperature-0 full test
set -e
cd "$(dirname "$0")/.."
source server/env.sh
URL=http://localhost:8600/v1

echo "=== A1: featmatch_fs (features + decisive prompt + worked examples) ==="
"$PY" scripts/06_run_agent.py --split test --mode featmatch_fs --tag qwen14b \
  --workers 16 --llm-url $URL --llm-model qwen2.5-14b
# retry pass for null rows
"$PY" scripts/06_run_agent.py --split test --mode featmatch_fs --tag qwen14b \
  --workers 8 --llm-url $URL --llm-model qwen2.5-14b

echo "=== A2: temperature-0 office (deterministic replication) ==="
"$PY" scripts/06_run_agent.py --split test --mode full --tag qwen14b_t0 \
  --temperature 0.0 --workers 16 --llm-url $URL --llm-model qwen2.5-14b
"$PY" scripts/06_run_agent.py --split test --mode full --tag qwen14b_t0 \
  --temperature 0.0 --workers 8 --llm-url $URL --llm-model qwen2.5-14b

echo "QUEUE A DONE"
