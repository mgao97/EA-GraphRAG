#!/usr/bin/env bash
# 后台收尾脚本：等待 5 个统一 baseline 全部产出 CSV，然后清理多余中间文件并生成汇总。
# 不会删除运行必需的 workspace / reproduce / output 目录，也不会删除历史正式结果。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$ROOT/results"
DS="hotpotqa_sample"

TARGETS=(
  "$RES/baseline_naive_${DS}_minimax.csv"
  "$RES/baseline_lightrag_${DS}_minimax_api.csv"
  "$RES/baseline_graphrag_${DS}_minimax_api.csv"
  "$RES/baseline_hipporag_${DS}_minimax_api.csv"
  "$RES/baseline_logicrag_${DS}_minimax_api.csv"
)

log(){ echo "[$(date +%H:%M:%S)] $*"; }

# 等待所有目标 CSV 出现（最多约 2 小时）
for i in $(seq 1 120); do
  missing=0
  for t in "${TARGETS[@]}"; do
    [ -f "$t" ] || { missing=1; break; }
  done
  if [ "$missing" -eq 0 ]; then
    log "All 5 baseline CSVs present."; break
  fi
  sleep 60
done

# 再多等 30s 让最后一个进程把文件刷盘
sleep 30

log "=== Cleaning redundant intermediate files ==="
# 明确多余/调试产物
rm -f "$RES/baseline_naive_minimax_test.csv" \
      "$RES/_test_hotpotqa_skip.csv" \
      "$RES/_cfg_2wikimultihopqa.yaml" \
      "$RES/_cfg_hotpotqa.yaml" \
      "$RES/_cfg_musique.yaml"
# 旧 naive 跑（非统一命名）若残留也删
rm -f "$RES/baseline_naive_minimax.csv"
log "Cleanup done."

# 生成汇总 markdown + csv
SUMMARY="$RES/baseline_unified_${DS}_summary.md"
{
  echo "# Unified Baseline Summary (dataset=$DS, n=30, LLM=MiniMax-M3 via :30001)"
  echo
  echo "| baseline | EM | F1 | rows |"
  echo "| --- | --- | --- | --- |"
} > "$SUMMARY"

SUMMARY_CSV="$RES/baseline_unified_${DS}_summary.csv"
echo "baseline,em,f1,rows" > "$SUMMARY_CSV"

for t in "${TARGETS[@]}"; do
  [ -f "$t" ] || continue
  name=$(basename "$t" | sed "s/baseline_//;s/_${DS}_minimax.*//;s/_minimax_api//")
  # 用 python 计算 EM/F1（取 em,f1 列的均值）
  /home/user/.conda/envs/grag/bin/python - "$t" "$name" <<'PY' >> "$SUMMARY"
import sys, csv, os
path, name = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(path)))
ems = [float(r["em"]) for r in rows if r.get("em") not in (None,"")]
f1s = [float(r["f1"]) for r in rows if r.get("f1") not in (None,"")]
em = sum(ems)/len(ems) if ems else 0
f1 = sum(f1s)/len(f1s) if f1s else 0
n = len(rows)
print(f"| {name} | {em:.4f} | {f1:.4f} | {n} |")
csvpath = os.path.join(os.path.dirname(path), "baseline_unified_hotpotqa_sample_summary.csv")
with open(csvpath, "a", newline="") as f:
    w = csv.writer(f); w.writerow([name, f"{em:.4f}", f"{f1:.4f}", n])
PY
done

log "=== Summary written ==="
cat "$SUMMARY"
log "Wake-up check: cat $SUMMARY"
