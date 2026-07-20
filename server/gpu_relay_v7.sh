#!/usr/bin/env bash
# Launch the vLLM servers for the v7 review experiments on node2.
# 14B x2 (GPU0:8600, GPU4:8602) + Llama-8B (GPU1:8601), max_model_len 12288
# (the featmatch_fs track prompt with worked examples exceeds 8192).
cd "$(dirname "$0")/.."
mkdir -p /data/yuxiaoning/projects/stormdesk_runtime/logs
GPU=0 PORT=8600 MODEL=/data/yuxiaoning/models/Qwen2.5-14B-Instruct \
  NAME=qwen2.5-14b UTIL=0.50 MAXLEN=12288 nohup bash server/launch_vllm.sh \
  > /data/yuxiaoning/projects/stormdesk_runtime/logs/vllm14b_v7.log 2>&1 &
echo "14B on GPU0:8600 pid $!"
GPU=4 PORT=8602 MODEL=/data/yuxiaoning/models/Qwen2.5-14B-Instruct \
  NAME=qwen2.5-14b UTIL=0.50 MAXLEN=12288 nohup bash server/launch_vllm.sh \
  > /data/yuxiaoning/projects/stormdesk_runtime/logs/vllm14b2_v7.log 2>&1 &
echo "14B on GPU4:8602 pid $!"
GPU=1 PORT=8601 MODEL=/data/yuxiaoning/models/Llama-3.1-8B-Instruct \
  NAME=llama-3.1-8b UTIL=0.30 MAXLEN=12288 nohup bash server/launch_vllm.sh \
  > /data/yuxiaoning/projects/stormdesk_runtime/logs/vllm8b_v7.log 2>&1 &
echo "Llama-8B on GPU1:8601 pid $!"
