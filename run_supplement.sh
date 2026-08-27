#!/usr/bin/env bash
# 补跑脚本（分阶段）：Ollama 已稳定（卡1/2/3, NUM_PARALLEL=4）
# 用法:
#   bash run_supplement.sh main      # 补主实验 EA-GraphRAG: musique, 2wikimultihopqa
#   bash run_supplement.sh logicrag  # 补 LogicRAG/musique（run_all 不重试该失败组）
# 每项均带"已完成则跳过"守卫，可重复安全执行。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=1,2,3
LOGDIR=logs
mkdir -p "$LOGDIR"
MODE="${1:-main}"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/supplement.log"; }
main_done(){ [ -f "results/e1_overall.csv" ] && grep -q "$1" "results/e1_overall.csv" 2>/dev/null; }
logicrag_done(){ ls baseline/logicrag/result/${1}_evaluation_results.json >/dev/null 2>&1; }

log "===== SUPPLEMENT($MODE) START $(date) ====="

if [ "$MODE" = "main" ]; then
  for D in musique 2wikimultihopqa; do
    if main_done "$D"; then log "MAIN/$D 已完成，跳过"; continue; fi
    log ">>> SUPPLEMENT MAIN-EXP: $D (n=50)"
    bash run.sh "$D" --with-real-data --skip-tests --n 50 >> "$LOGDIR/main_exp_${D}_50.log" 2>&1
    log "<<< MAIN/$D exit=$?"
  done
elif [ "$MODE" = "logicrag" ]; then
  for D in musique 2wikimultihopqa hotpotqa; do
    if logicrag_done "$D"; then log "LogicRAG/$D 已完成，跳过"; continue; fi
    log ">>> SUPPLEMENT LogicRAG: $D (n=50)"
    bash baseline/logicrag/run.sh "$D" 50 >> "$LOGDIR/logicrag_${D}_50.log" 2>&1
    log "<<< LogicRAG/$D exit=$?"
  done
else
  log "未知模式: $MODE (用 main|logicrag)"
fi

log "===== SUPPLEMENT($MODE) DONE $(date) ====="
