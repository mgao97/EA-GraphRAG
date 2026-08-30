#!/usr/bin/env bash
# 仅组A补齐(Qwen30000 LLM + bianxie text-embedding-3-small 30001), 不含组B。
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
GRAG_PY="/home/user/.conda/envs/grag/bin/python"
GRAPH_PY="/home/user/.conda/envs/graphrag/bin/python"
SLS_PY="/home/user/.conda/envs/sls/bin/python"
N=50
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_a_graphrag() {
  local DS="$1"
  log "==> [A] graphrag $DS n=$N (vllm 30000)"
  cd "$ROOT/baseline/graphrag"
  setsid $GRAPH_PY run_dataset_vllm.py --dataset "$DS" --n "$N" --corpus_limit 450 \
    --output "../../results/baseline_graphrag_${DS}_vllm.csv" \
    > run_graphrag_${DS}_vllm.log 2>&1 < /dev/null & disown
  cd "$ROOT"
}
run_a_hipporag() {
  local DS="$1"
  log "==> [A] hipporag $DS n=$N (vllm 30000)"
  cd "$ROOT/baseline/hipporag"
  setsid bash run_vllm.sh "$DS" "$N" \
    > run_hipporag_${DS}_vllm.log 2>&1 < /dev/null & disown
  cd "$ROOT"
}

log "=============== 组A 补齐(缺失项) ==============="
run_a_graphrag hotpotqa
for DS in hotpotqa musique 2wikimultihopqa; do run_a_hipporag "$DS"; done
log "组A 补齐完成(不含已跑的EA/EA-v1/GraphRAG musique/2wiki)."
