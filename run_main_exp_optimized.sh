#!/usr/bin/env bash
# EA-GraphRAG 优化版重跑（图度剪枝 + beam expansion）
# 使用 configs/default.yaml 中的 graph.max_degree / retrieval.max_nodes_per_hop
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=1,2,3
LOGDIR=logs
mkdir -p "$LOGDIR"

echo "===== EA-OPTIMIZED START $(date) =====" | tee -a "$LOGDIR/main_opt.log"

for D in musique 2wikimultihopqa hotpotqa; do
  echo "########################################################" | tee -a "$LOGDIR/main_opt.log"
  echo "# EA-OPT: $D (n=50)" | tee -a "$LOGDIR/main_opt.log"
  echo "########################################################" | tee -a "$LOGDIR/main_opt.log"
  bash run.sh "$D" --with-real-data --skip-tests --n 50 \
    >> "$LOGDIR/main_exp_${D}_50.log" 2>&1
  echo "<<< [EA/$D] exit=$? $(date)" | tee -a "$LOGDIR/main_opt.log"
done

echo "===== EA-OPTIMIZED ALL DONE $(date) =====" | tee -a "$LOGDIR/main_opt.log"
