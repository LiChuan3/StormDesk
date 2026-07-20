#!/usr/bin/env bash
# Unified-manifest analysis pipeline (v6 review round): regime statics, the
# frozen homogeneous sample, and every manifest-locked analysis.
set -e
cd "$(dirname "$0")/.."
source server/env.sh
R=$STORMDESK_WORK/results
M=$R/test_manifest.json

echo "=== [1/7] regime-conditioned statics + ridge gate ==="
$PY scripts/29_regime_static.py --split test

echo "=== [2/7] learned auditor + aiwp postproc (with calib CV prior) ==="
$PY scripts/26_learned_auditor.py --emit-calib-cv

# the frozen Table-1 pool: every point method reported in main or supp tables
POOL=persistence,cliper,gru,transformer,pangu,fengwu,cons_equal,cons_weighted,cons_bc,cons_aiwp,hybrid_static,hybrid_rules,static_convex,static_basin,static_icat,gate_ridge,gbt_static,gbt_case,gbt_shuffled,learned_gbt2,learned_gbt_contract,analog_median,analog_linear,aiwp_postproc,agent_full_qwen14b,agent_full_qwen72b,agent_full_qwen7b,agent_mini_qwen14b,agent_single_qwen14b,agent_single_refine_qwen14b,agent_no_analogs_qwen14b,agent_no_auditor_qwen14b,agent_no_diagnostics_qwen14b,agent_full_qwen14b_anon,agent_full_qwen14b_gateadv

echo "=== [3/7] unified evaluation -> manifest ==="
$PY scripts/07_evaluate.py --split test --methods $POOL \
  --baseline cliper \
  --sig-against hybrid_static,static_convex,agent_full_qwen14b \
  --sig-targets agent_full_qwen14b,agent_full_qwen72b,agent_mini_qwen14b,agent_single_refine_qwen14b,learned_gbt2,learned_gbt_contract,static_convex,static_basin,static_icat,gate_ridge,gbt_case,aiwp_postproc,agent_full_qwen14b_gateadv \
  --bootstrap 2000 --bootstrap-ref hybrid_static \
  --manifest-out $M \
  --out $R/v7_metrics.csv > $R/final_eval_v7.txt 2>&1
tail -5 $R/final_eval_v7.txt

echo "=== [4/7] TOST equivalence on the manifest sample ==="
$PY scripts/21_equivalence.py --case-list $M --pairs \
"agent_full_qwen14b:hybrid_static,agent_full_qwen14b_anon:agent_full_qwen14b,learned_gbt2:hybrid_static,learned_gbt_contract:hybrid_static,agent_mini_qwen14b:hybrid_static,agent_single_refine_qwen14b:hybrid_static,static_convex:hybrid_static,static_basin:static_convex,static_icat:static_convex,gate_ridge:static_convex" \
  > $R/test_equivalence_v7.txt 2>&1
cp $STORMDESK_WORK/results/test_equivalence.json $R/test_equivalence_v7.json || true
tail -3 $R/test_equivalence_v7.txt

echo "=== [5/7] auditor identification on the manifest sample ==="
$PY scripts/20_auditor_id.py --case-list $M > $R/auditor_id_v7.txt 2>&1
tail -8 $R/auditor_id_v7.txt

echo "=== [6/7] learned auditor / postproc on the manifest sample ==="
$PY scripts/26_learned_auditor.py --case-list $M > $R/learned_auditor_v7.txt 2>&1
tail -10 $R/learned_auditor_v7.txt

echo "=== [7/7] headroom utilization U ==="
$PY scripts/30_headroom.py --case-list $M > $R/headroom_v7.txt 2>&1
cat $R/headroom_v7.txt

echo "=== few-shot exemplars (for featmatch_fs) ==="
$PY scripts/31_build_fewshot.py > $R/fewshot_build.txt 2>&1
tail -5 $R/fewshot_build.txt

echo "PIPELINE DONE"
