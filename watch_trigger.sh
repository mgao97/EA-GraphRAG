#!/usr/bin/env bash
# 双 watcher：
#  A) 监听主实验 launcher(run_main_exp.sh) 退出 且 results/e1_overall.csv 含 hotpotqa
#     -> 触发: bash run_supplement.sh main
#  B) 监听 run_all.sh 50 进程退出 -> 触发: bash run_supplement.sh logicrag
# 各自用 flag 文件防重复触发。每 2 分钟轮询一次。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
LOG=logs/watcher.log
POLL=120
FLAG_MAIN=logs/.trig_main
FLAG_LR=logs/.trig_logicrag

echo "[$(date)] dual-watcher started (pid $$)" >> "$LOG"

while true; do
  sleep "$POLL"

  LAUNCHER=$(pgrep -f "run_main_exp.sh" | head -1)
  RUNALL=$(pgrep -f "run_all.sh" | head -1)
  HPQA_CSV=results/e1_overall.csv

  # --- A: 主实验补跑 ---
  if [ ! -f "$FLAG_MAIN" ] && [ -z "$LAUNCHER" ] && [ -f "$HPQA_CSV" ] && grep -q "hotpotqa" "$HPQA_CSV" 2>/dev/null; then
    echo "[$(date)] A-TRIGGER: launcher gone + hotpotqa done -> run_supplement.sh main" >> "$LOG"
    touch "$FLAG_MAIN"
    nohup bash run_supplement.sh main > logs/supplement_launcher.log 2>&1 &
    echo "[$(date)] main supplement launched pid $!" >> "$LOG"
  fi

  # --- B: LogicRAG 补跑（run_all 完全结束）---
  if [ ! -f "$FLAG_LR" ] && [ -z "$RUNALL" ]; then
    echo "[$(date)] B-TRIGGER: run_all gone -> run_supplement.sh logicrag" >> "$LOG"
    touch "$FLAG_LR"
    nohup bash run_supplement.sh logicrag > logs/supplement_logicrag.log 2>&1 &
    echo "[$(date)] logicrag supplement launched pid $!" >> "$LOG"
  fi

  # 两者都触发过则退出 watcher
  if [ -f "$FLAG_MAIN" ] && [ -f "$FLAG_LR" ]; then
    echo "[$(date)] both triggered, watcher exit" >> "$LOG"
    exit 0
  fi

  echo "[$(date)] poll: launcher=${LAUNCHER:-none} runall=${RUNALL:-none} hpqacsv=$([ -f "$HPQA_CSV" ] && echo yes || echo no)" >> "$LOG"
done
