#!/usr/bin/env bash
# One-time setup of the vLLM env + Qwen weights on a GPU node (China mirrors).
# Usage: bash server/setup_vllm.sh [env_dir] [model_dir]
set -e
ENV_DIR=${ENV_DIR:-${1:-/data_small/user_envs/USER/stormdesk-vllm}}
MODEL_DIR=${MODEL_DIR:-${2:-/data/USER/models}}
PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
# stale proxy settings on the nodes break pip/hf downloads (both env vars and
# a dead proxy hardcoded in ~/.pip/pip.conf)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PIP_CONFIG_FILE=/dev/null

PYBIN=${PYBIN:-/data_small/user_envs/USER/llm-tc/bin/python}
if [ ! -f "$ENV_DIR/bin/activate" ]; then
  "$PYBIN" -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
python -m pip install -q -i $PIP_INDEX --upgrade pip
python -m pip install -i $PIP_INDEX vllm 2>&1 | tail -3
python -m pip install -i $PIP_INDEX "huggingface_hub[cli]" 2>&1 | tail -2

mkdir -p "$MODEL_DIR"
for M in "$@"; do :; done
download() {
  local repo=$1
  local name=$(basename "$repo")
  if [ ! -d "$MODEL_DIR/$name" ] || [ -z "$(ls -A "$MODEL_DIR/$name" 2>/dev/null)" ]; then
    echo "downloading $repo ..."
    python -m huggingface_hub.commands.huggingface_cli download "$repo" \
      --local-dir "$MODEL_DIR/$name" --exclude "*.pth" 2>&1 | tail -3
  else
    echo "$name already present"
  fi
}
download Qwen/Qwen2.5-14B-Instruct
echo "setup done. models in $MODEL_DIR"
