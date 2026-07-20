#!/usr/bin/env bash
# Stage-2 fill pipeline: evaluate the new GPU runs on the frozen manifest.
# Run on node1 (or node2) after the queues finish. Usage:
#   bash server/run_v7_fill.sh [featmatch|t0|strongprior|all]
set -e
cd "$(dirname "$0")/.."
source server/env.sh
R=$STORMDESK_WORK/results
M=$R/test_manifest.json
WHAT=${1:-all}

POOL=hybrid_static,static_convex,static_basin,learned_gbt2,learned_gbt_contract,aiwp_postproc,agent_full_qwen14b,agent_full_qwen14b_gateadv,agent_single_refine_qwen14b

if [ "$WHAT" = "featmatch" ] || [ "$WHAT" = "all" ]; then
  echo "=== evaluate featmatch runs on the manifest ==="
  $PY scripts/07_evaluate.py --split test \
    --methods $POOL,agent_featmatch_qwen14b,agent_featmatch_fs_qwen14b \
    --baseline hybrid_static --case-list $M \
    --sig-against hybrid_static,agent_full_qwen14b \
    --sig-targets agent_featmatch_qwen14b,agent_featmatch_fs_qwen14b \
    --out $R/v7_featmatch_metrics.csv > $R/final_eval_v7_featmatch.txt 2>&1
  grep -A14 "=== lead" $R/final_eval_v7_featmatch.txt | head -50
  echo "=== headroom U incl. featmatch ==="
  $PY scripts/30_headroom.py --case-list $M --learned-ref learned_gbt2 \
    --policies agent_full_qwen14b,agent_full_qwen14b_gateadv,agent_featmatch_qwen14b,agent_featmatch_fs_qwen14b,agent_single_refine_qwen14b \
    > $R/headroom_v7_featmatch.txt 2>&1
  cat $R/headroom_v7_featmatch.txt
  echo "=== TOST featmatch vs hybrid ==="
  $PY scripts/21_equivalence.py --case-list $M --pairs \
    "agent_featmatch_qwen14b:hybrid_static,agent_featmatch_fs_qwen14b:hybrid_static,agent_featmatch_fs_qwen14b:agent_full_qwen14b" \
    > $R/tost_featmatch.txt 2>&1
  tail -12 $R/tost_featmatch.txt
fi

if [ "$WHAT" = "t0" ] || [ "$WHAT" = "all" ]; then
  echo "=== evaluate temperature-0 run ==="
  $PY scripts/07_evaluate.py --split test \
    --methods hybrid_static,agent_full_qwen14b,agent_full_qwen14b_t0 \
    --baseline hybrid_static --case-list $M \
    --out $R/v7_t0_metrics.csv > $R/final_eval_v7_t0.txt 2>&1
  grep -A6 "=== lead" $R/final_eval_v7_t0.txt | head -30
  grep -A6 "RI via explicit" $R/final_eval_v7_t0.txt | head -8
fi

if [ "$WHAT" = "strongprior" ] || [ "$WHAT" = "all" ]; then
  echo "=== evaluate stronger-prior office ==="
  $PY scripts/07_evaluate.py --split test \
    --methods hybrid_static,aiwp_postproc,agent_full_qwen14b,agent_full_qwen14b_strongprior \
    --baseline hybrid_static --case-list $M \
    --sig-against aiwp_postproc --sig-targets agent_full_qwen14b_strongprior \
    --out $R/v7_sp_metrics.csv > $R/final_eval_v7_sp.txt 2>&1
  grep -A8 "=== lead" $R/final_eval_v7_sp.txt | head -36
  grep -A10 "paired tests" $R/final_eval_v7_sp.txt | head -16
fi

echo "FILL DONE ($WHAT)"
