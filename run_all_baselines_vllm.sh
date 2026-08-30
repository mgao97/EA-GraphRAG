#!/usr/bin/env bash
# 统一实验设置（vLLM 本地 LLM 版）：所有 baseline 的 LLM 统一走本地 vLLM
# (127.0.0.1:30000, Qwen2.5-14B-Instruct-1M, GPU3+4 tp2)，embedding 统一走本地
# Ollama (nomic-embed-text, 11434/11436)。评测口径统一 EM/F1。
#
# 与 minimax_api 版的区别：LLM 后端从 MiniMax 云端代理(30001) 换成本地 vLLM(30000)。
# 索引目录彼此隔离（graphrag_vllm/、lightrag_workspace_*_vllm、hipporag * _vllm、
# logicrag 同 dataset 但不同 LLM），不与 minimax 实验冲突。
#
# 用法：
#   bash run_all_baselines_vllm.sh            # 三数据集全部后台启动（DS 串行, baseline 并行）
#   bash run_all_baselines_vllm.sh musique    # 只跑指定数据集
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N=50
ONLY="${1:-all}"

GRAG_PY="/home/user/.conda/envs/grag/bin/python"
GRAPH_PY="/home/user/.conda/envs/graphrag/bin/python"
SLS_PY="/home/user/.conda/envs/sls/bin/python"

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_naive() {
  log "==> naive (vLLM, env: grag)"
  cd "$ROOT/baseline/lightrag"
  $GRAG_PY run_naive_vllm.py \
    --data "../../data/raw/$DS/questions.json" \
    --corpus "../../data/raw/$DS/corpus.json" \
    --n "$N" --top_k 5 \
    --output "../../results/baseline_naive_${DS}_vllm.csv" \
    > run_naive_${DS}_vllm.log 2>&1 &
}

run_lightrag() {
  log "==> lightrag (vLLM, env: grag)"
  cd "$ROOT/baseline/lightrag"
  $GRAG_PY run_dataset_vllm.py \
    --dataset "$DS" --n "$N" --mode hybrid \
    --output "../../results/baseline_lightrag_${DS}_vllm.csv" \
    > run_lightrag_${DS}_vllm.log 2>&1 &
}

run_graphrag() {
  log "==> graphrag (vLLM, env: graphrag)"
  cd "$ROOT/baseline/graphrag"
  $GRAPH_PY run_dataset_vllm.py \
    --dataset "$DS" --n "$N" --corpus_limit 450 \
    --output "../../results/baseline_graphrag_${DS}_vllm.csv" \
    > run_graphrag_${DS}_vllm.log 2>&1 &
}

run_hipporag() {
  log "==> hipporag (vLLM, env: sls)"
  cd "$ROOT/baseline/hipporag"
  bash run_vllm.sh "$DS" "$N" \
    > run_hipporag_${DS}_vllm.log 2>&1 &
}

run_logicrag() {
  log "==> logicrag (vLLM, env: sls)"
  cd "$ROOT/baseline/logicrag"
  bash run.sh "$DS" "$N" \
    > run_logicrag_${DS}_vllm.log 2>&1 &
}

run_ds() {
  DS="$1"
  log "===== 数据集 $DS (n=$N) ====="
  case "$ONLY" in
    all)       run_naive; run_lightrag; run_graphrag; run_hipporag; run_logicrag ;;
    naive)     run_naive ;;
    lightrag)  run_lightrag ;;
    graphrag)  run_graphrag ;;
    hipporag)  run_hipporag ;;
    logicrag)  run_logicrag ;;
    *) echo "unknown baseline: $ONLY"; exit 1 ;;
  esac
  wait
  log "===== 数据集 $DS 完成 ====="
}

for DS in musique 2wikimultihopqa hotpotqa; do
  run_ds "$DS"
done

log "All vLLM baselines finished. Results under results/*_vllm.csv"
