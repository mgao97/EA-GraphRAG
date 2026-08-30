#!/usr/bin/env bash
# 最终收尾：等待 graphrag 重跑完成，重建 summary，清理多余中间文件。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/user/.conda/envs/grag/bin/python
RES="$ROOT/results"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

# 1) 等 graphrag 重跑进程结束（最多约 40 分钟）
for i in $(seq 1 80); do
  if ! pgrep -f "run_dataset_minimax_api.py --dataset hotpotqa_sample" >/dev/null; then
    log "graphrag rerun finished."; break
  fi
  sleep 30
done
sleep 10

# 2) 重建 summary（此时 5 个 CSV 均为正确结果）
log "rebuilding summary"
$PY "$ROOT/scripts/build_summary.py"

# 3) 清理多余中间文件
log "cleaning redundant files"
rm -f "$RES/baseline_naive_minimax_test.csv" \
      "$RES/_test_hotpotqa_skip.csv" \
      "$RES/_cfg_2wikimultihopqa.yaml" \
      "$RES/_cfg_hotpotqa.yaml" \
      "$RES/_cfg_musique.yaml" \
      "$RES/baseline_naive_minimax.csv"

log "all done. summary: $RES/baseline_unified_hotpotqa_sample_summary.md"
