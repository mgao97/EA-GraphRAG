#!/usr/bin/env bash
# 组B补齐(MiniMax 30001 LLM + text-embedding-3-small 30001 embedding), 三个数据集各 n=50。
# EA/EA-v1 仅 hotpotqa。
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
GRAG_PY="/home/user/.conda/envs/grag/bin/python"
GRAPH_PY="/home/user/.conda/envs/graphrag/bin/python"
SLS_PY="/home/user/.conda/envs/sls/bin/python"
N=50
log() { echo "[$(date +%H:%M:%S)] $*"; }

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

log "=============== 组B 补齐启动 ==============="
run_b_ea
run_b_eav1
run_b_graphrag hotpotqa
run_b_graphrag 2wikimultihopqa
for DS in hotpotqa musique 2wikimultihopqa; do run_b_hipporag "$DS"; done
run_b_logicrag hotpotqa
run_b_logicrag 2wikimultihopqa
log "组B 所有 job launched."
