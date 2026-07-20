#!/usr/bin/env bash
# Launch the vLLM OpenAI-compatible server for StormDesk agents.
# Usage: GPU=1 PORT=8500 MODEL=/data/yuxiaoning/models/Qwen2.5-14B-Instruct \
#        NAME=qwen2.5-14b bash server/launch_vllm.sh
set -e
ENV_DIR=${ENV_DIR:-/data_small/user_envs/yuxiaoning/stormdesk-vllm}
GPU=${GPU:-1}
PORT=${PORT:-8500}
MODEL=${MODEL:-/data/yuxiaoning/models/Qwen2.5-14B-Instruct}
NAME=${NAME:-qwen2.5-14b}
UTIL=${UTIL:-0.85}
MAXLEN=${MAXLEN:-8192}

source "$ENV_DIR/bin/activate"
export CUDA_VISIBLE_DEVICES=$GPU
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" \
  --no-enable-log-requests
