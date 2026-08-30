#!/usr/bin/env bash
# 统一实验设置：所有 baseline 都使用同一份数据集 (data/raw/hotpotqa_sample)，
# 同一批 30 道题，LLM 统一走本地 MiniMax 代理 (127.0.0.1:30001, MiniMax-M3)，
# embedding 统一走本地 Ollama (nomic-embed-text)，评测口径统一 EM/F1。
#
# 用法：
#   bash run_all_baselines_minimax.sh            # 全部后台启动
#   bash run_all_baselines_minimax.sh lightrag   # 只跑指定 baseline
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS="hotpotqa_sample"
N=30
ONLY="${1:-all}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_naive() {
  log "==> naive-minimax (env: grag)"
  cd "$ROOT/baseline/lightrag"
  setsid /home/user/.conda/envs/grag/bin/python run_naive_minimax.py \
    --data "../../data/raw/$DS/questions.json" \
    --corpus "../../data/raw/$DS/corpus.json" \
    --n "$N" --top_k 5 \
    --output "../../results/baseline_naive_${DS}_minimax.csv" \
    > run_naive_${DS}.log 2>&1 < /dev/null & disown
  log "    naive pid=$!"
}

run_lightrag() {
  log "==> lightrag (env: grag)"
  cd "$ROOT/baseline/lightrag"
  setsid /home/user/.conda/envs/grag/bin/python run_dataset_minimax_api.py \
    --dataset "$DS" --n "$N" --mode hybrid \
    --output "../../results/baseline_lightrag_${DS}_minimax_api.csv" \
    > run_lightrag_${DS}.log 2>&1 < /dev/null & disown
  log "    lightrag pid=$!"
}

run_graphrag() {
  log "==> graphrag (env: graphrag)"
  cd "$ROOT/baseline/graphrag"
  setsid /home/user/.conda/envs/graphrag/bin/python run_dataset_minimax_api.py \
    --dataset "$DS" --n "$N" --corpus_limit 0 \
    --output "../../results/baseline_graphrag_${DS}_minimax_api.csv" \
    > run_graphrag_${DS}.log 2>&1 < /dev/null & disown
  log "    graphrag pid=$!"
}

run_hipporag() {
  log "==> hipporag (env: sls)"
  cd "$ROOT/baseline/hipporag"
  setsid bash run_minimax_api.sh "$DS" "$N" \
    > run_hipporag_${DS}.log 2>&1 < /dev/null & disown
  log "    hipporag pid=$!"
}

run_logicrag() {
  log "==> logicrag (env: sls)"
  cd "$ROOT/baseline/logicrag"
  setsid bash run_minimax_api.sh "$DS" "$N" \
    > run_logicrag_${DS}.log 2>&1 < /dev/null & disown
  log "    logicrag pid=$!"
}

case "$ONLY" in
  all)       run_naive; run_lightrag; run_graphrag; run_hipporag; run_logicrag ;;
  naive)     run_naive ;;
  lightrag)  run_lightrag ;;
  graphrag)  run_graphrag ;;
  hipporag)  run_hipporag ;;
  logicrag)  run_logicrag ;;
  *) echo "unknown baseline: $ONLY"; exit 1 ;;
esac

log "All requested baselines launched. Logs under baseline/<name>/run_<name>_${DS}.log"
