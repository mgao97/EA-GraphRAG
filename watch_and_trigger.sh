#!/usr/bin/env bash
# 监听 hotpotqa 主实验结束 -> 自动触发补跑脚本 run_supplement.sh
# 触发条件（严格，避免误触发）：
#   1) 主实验 launcher 进程（run_main_exp.sh）已退出
#   2) results/e1_overall.csv 已生成 且 包含 hotpotqa 行（说明 hotpotqa 主实验真正完成）
#   3) run_supplement.sh 尚未启动过（用 flag 文件防重复）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
LOG=logs/watcher.log
FLAG=logs/.supplement_triggered
POLL=120   # 每 2 分钟检测一次

echo "[$(date)] watcher started (pid $$)" >> "$LOG"

while true; do
  sleep "$POLL"

  # launcher 是否还在
  LAUNCHER=$(pgrep -f "run_main_exp.sh" | head -1)
  HPQA_CSV=results/e1_overall.csv

  if [ -f "$FLAG" ]; then
    # 已触发过，静默退出 watcher
    echo "[$(date)] supplement already triggered, watcher exit" >> "$LOG"
    exit 0
  fi

  if [ -z "$LAUNCHER" ] && [ -f "$HPQA_CSV" ] && grep -q "hotpotqa" "$HPQA_CSV" 2>/dev/null; then
    echo "[$(date)] CONDITION MET: launcher gone + hotpotqa result present. Triggering supplement." >> "$LOG"
    touch "$FLAG"
    nohup bash run_supplement.sh > logs/supplement_launcher.log 2>&1 &
    echo "[$(date)] supplement launched pid $!" >> "$LOG"
    exit 0
  fi

  echo "[$(date)] check: launcher=${LAUNCHER:-none} hpqacsv=$( [ -f "$HPQA_CSV" ] && echo yes || echo no )" >> "$LOG"
done
