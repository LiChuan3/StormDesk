#!/usr/bin/env bash
# Run one guidance member for one shard on one GPU across splits.
# Usage: bash server/run_aiwp_shard.sh <gpu> <member> <shard> <nshards> [splits...]
set -e
cd "$(dirname "$0")/.."
source server/env.sh
GPU=$1; MEMBER=$2; SHARD=$3; NSHARDS=$4; shift 4
SPLITS=("$@")
if [ ${#SPLITS[@]} -eq 0 ]; then SPLITS=(test calib); fi
LOG=$STORMDESK_WORK/logs
mkdir -p "$LOG"
for SPLIT in "${SPLITS[@]}"; do
  echo "=== $SPLIT $MEMBER shard $SHARD/$NSHARDS on GPU $GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU nice -n 5 $PY scripts/02_run_aiwp.py \
    --split "$SPLIT" --member "$MEMBER" --gpu 0 \
    --shard "$SHARD" --nshards "$NSHARDS" \
    >> "$LOG/aiwp_${SPLIT}_${MEMBER}_s${SHARD}.log" 2>&1
done
echo "$MEMBER shard $SHARD complete"
