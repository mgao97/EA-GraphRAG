#!/usr/bin/env bash
# Unified scheduler: run EA-GraphRAG (main) + 4 baselines on all three
# multi-hop QA datasets using the SAME per-dataset retrieval corpus.
#
# Datasets: musique | 2wikimultihopqa | hotpotqa  (1000 questions each)
# All methods index/retrieve over data/raw/<dataset>/corpus.json so the
# comparison is fair (supporting + distractor passages pooled per dataset).
#
# Usage:
#   bash run_all.sh [n] [datasets...]
#   bash run_all.sh 50                                   # smoke: 50 q, all 3 datasets
#   bash run_all.sh 1000 musique 2wikimultihopqa hotpotqa  # full
#
# Logs: run_all_<timestamp>.log
set -uo pipefail

N="${1:-50}"
shift || true
if [[ $# -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(musique 2wikimultihopqa hotpotqa)
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p results logs

LOG="logs/run_all_$(date +%Y%m%d_%H%M%S).log"
echo "=== run_all: n=$N datasets=${DATASETS[*]} ===" | tee "$LOG"

run() {
  local label="$1"; shift
  echo "" | tee -a "$LOG"
  echo ">>> [$label] $*" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "<<< [$label] OK" | tee -a "$LOG"
  else
    echo "<<< [$label] FAILED (exit $?)" | tee -a "$LOG"
  fi
}

for D in "${DATASETS[@]}"; do
  echo "########################################################" | tee -a "$LOG"
  echo "# DATASET: $D (n=$N)" | tee -a "$LOG"
  echo "########################################################" | tee -a "$LOG"

  # Baselines
  run "LogicRAG/$D"        bash baseline/logicrag/run.sh        "$D" "$N"
  run "HippoRAG/$D"        bash baseline/hipporag/run.sh        "$D" "$N"
  run "LightRAG/$D"        bash baseline/lightrag/run.sh        "$D" "$N"
  run "GraphRAG/$D"        bash baseline/graphrag/run.sh        "$D" "$N" 0

  # Main experiment (EA-GraphRAG) is intentionally NOT run here: baseline/
  # contains only the external comparison methods (LogicRAG/HippoRAG/LightRAG/
  # GraphRAG). The EA-GraphRAG method itself is validated separately via
  # run_ea_only.py against the optimised config. Uncomment to re-enable:
  # run "EA-GraphRAG/$D"     bash run.sh "$D" --n "$N" --skip-tests --skip-figures
done

echo "" | tee -a "$LOG"
echo "=== DONE. Per-dataset CSVs under results/: baseline_*_<dataset>.csv + ea_graphrag results ===" | tee -a "$LOG"
