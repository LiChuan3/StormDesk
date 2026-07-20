#!/usr/bin/env bash
# Source me: sets PY and LD_LIBRARY_PATH so onnxruntime-gpu finds cuDNN/cuBLAS
# bundled inside the llm-tc torch environment.
export ENV_ROOT=${ENV_ROOT:-/data_small/user_envs/yuxiaoning/llm-tc}
export PY=$ENV_ROOT/bin/python
NV=$ENV_ROOT/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:$NV/cuda_runtime/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cuda_nvrtc/lib:$NV/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export STORMDESK_WORK=${STORMDESK_WORK:-/data/yuxiaoning/projects/stormdesk_runtime}
