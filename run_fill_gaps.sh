#!/usr/bin/env bash
# 补齐组A(LLM=30000 Qwen + embedding=30001 text-embedding-3-small) 与
# 组B(LLM=30001 MiniMax + embedding=30001 text-embedding-3-small) 的空缺实验。
# EA/EA-v1 框架仅支持 hotpotqa（n=50）。
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GRAG_PY="/home/user/.conda/envs/grag/bin/python"
GRAPH_PY="/home/user/.conda/envs/graphrag/bin/python"
SLS_PY="/home/user/.conda/envs/sls/bin/python"
N=50
log() { echo "[$(date +%H:%M:%S)] $*"; }

################ 组A: Qwen30000 ################
run_a_ea() {
  log "==> [A] EA hotpotqa n=$N (vllm_qwen14b)"
  setsid $GRAPH_PY run_ea_only.py hotpotqa $N configs/vllm_qwen14b.yaml ea_graphrag \
    > logs/r_ea_A_hotpotqa.log 2>&1 < /dev/null & disown
  log "    pid=$!"
}
run_a_eav1() {
  log "==> [A] EA-v1 hotpotqa n=$N (groupA qwen30000)"
  setsid $GRAPH_PY run_ea_only.py hotpotqa $N configs/ea_v1_groupA_qwen30000.yaml ea_graphrag_v1 \
    > logs/r_ea_v1_A_hotpotqa.log 2>&1 < /dev/null & disown
  log "    pid=$!"
}
run_a_graphrag() {
  local DS="$1"
  log "==> [A] graphrag $DS n=$N (vllm 30000)"
  cd "$ROOT/baseline/graphrag"
  setsid $GRAPH_PY run_dataset_vllm.py --dataset "$DS" --n "$N" --corpus_limit 450 \
    --output "../../results/baseline_graphrag_${DS}_vllm.csv" \
    > run_graphrag_${DS}_vllm.log 2>&1 < /dev/null & disown
  log "    pid=$!"
  cd "$ROOT"
}
run_a_hipporag() {
  local DS="$1"
  log "==> [A] hipporag $DS n=$N (vllm 30000)"
  cd "$ROOT/baseline/hipporag"
  setsid bash run_vllm.sh "$DS" "$N" \
    > run_hipporag_${DS}_vllm.log 2>&1 < /dev/null & disown
  log "    pid=$!"
  cd "$ROOT"
}

################ 组B: MiniMax30001 ################
run_b_ea() {
  log "==> [B] EA hotpotqa n=$N (minimax30001)"
  setsid $GRAPH_PY run_ea_only.py hotpotqa $N configs/minimax.yaml ea_graphrag \
    > logs/r_ea_B_hotpotqa.log 2>&1 < /dev/null & disown
  log "    pid=$!"
}
run_b_eav1() {
  log "==> [B] EA-v1 hotpotqa n=$N (groupB minimax30001)"
  setsid $GRAPH_PY run_ea_only.py hotpotqa $N configs/ea_v1_groupB_minimax30001.yaml ea_graphrag_v1 \
    > logs/r_ea_v1_B_hotpotqa.log 2>&1 < /dev/null & disown
  log "    pid=$!"
}
run_b_graphrag() {
  local DS="$1"
  log "==> [B] graphrag $DS n=$N (minimax30001)"
  cd "$ROOT/baseline/graphrag"
  setsid $GRAPH_PY run_dataset_minimax_api.py --dataset "$DS" --n "$N" --corpus_limit 0 \
    --output "../../results/baseline_graphrag_${DS}_minimax_api.csv" \
    > run_graphrag_${DS}_minimax_api.log 2>&1 < /dev/null & disown
  log "    pid=$!"
  cd "$ROOT"
}
run_b_hipporag() {
  local DS="$1"
  log "==> [B] hipporag $DS n=$N (minimax30001)"
  cd "$ROOT/baseline/hipporag"
  setsid bash run_minimax_api.sh "$DS" "$N" \
    > run_hipporag_${DS}_minimax_api.log 2>&1 < /dev/null & disown
  log "    pid=$!"
  cd "$ROOT"
}
run_b_logicrag() {
  local DS="$1"
  log "==> [B] logicrag $DS n=$N (minimax30001)"
  cd "$ROOT/baseline/logicrag"
  setsid bash run_minimax_api.sh "$DS" "$N" \
    > run_logicrag_${DS}_minimax_api.log 2>&1 < /dev/null & disown
  log "    pid=$!"
  cd "$ROOT"
}

log "=============== 组A 补齐 ==============="
run_a_ea
run_a_eav1
run_a_graphrag hotpotqa
run_a_graphrag 2wikimultihopqa
for DS in hotpotqa musique 2wikimultihopqa; do run_a_hipporag "$DS"; done

log "=============== 组B 补齐 ==============="
run_b_ea
run_b_eav1
run_b_graphrag hotpotqa
run_b_graphrag 2wikimultihopqa
for DS in hotpotqa musique 2wikimultihopqa; do run_b_hipporag "$DS"; done
run_b_logicrag hotpotqa
run_b_logicrag 2wikimultihopqa

log "All fill-gap jobs launched. (组B LightRAG 3数据集已在跑, 组A GraphRAG musique 已在跑)"
