#!/usr/bin/env bash
# 主实验 EA-GraphRAG 三数据集 × 50 题（新设置，与 baseline 对齐）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=1,2,3   # 避开卡0（留给 baseline）；主实验 GPU 实际来自 Ollama
# 2wikimultihopqa 已由独立进程(2660322)在跑，这里跳过避免竞争同一 workspace
for D in musique hotpotqa; do
  echo "########################################################"
  echo "# MAIN-EXP DATASET: $D (n=50)"
  echo "########################################################"
  bash run.sh "$D" --with-real-data --skip-tests --n 50 \
    >> "logs/main_exp_${D}_50.log" 2>&1
  echo "<<< [MAIN/$D] exit=$?  $(date)"
done
echo "MAIN-EXP ALL DONE $(date)" >> logs/main_exp_all.log
