#!/usr/bin/env bash
# Phase 1 end-to-end runner for EA-GraphRAG.
#
# This script ALWAYS uses the project's conda environment `grag`.
# It does not touch the system Python, the system pip, or any other env.
#
# Usage:
#     ./run.sh                       # full Phase 1 with synthetic data
#     ./run.sh --with-real-data      # requires data/raw/hotpot_*_v1.1.json
#     ./run.sh --skip-tests          # skip unit tests
#     ./run.sh --skip-figures        # skip matplotlib PDF generation
#     ./run.sh --config <yaml>       # use a specific config (default: configs/default.yaml)
#     ./run.sh --n 200               # run 200 questions instead of the default
#
# Environment overrides:
#     CONDA_ENV=grag                 # change conda env if you renamed it
#     N_QUESTIONS=50
#     SKIP_TESTS=1 / SKIP_FIGURES=1

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CONDA_ENV="${CONDA_ENV:-grag}"
N_QUESTIONS="${N_QUESTIONS:-30}"
SKIP_TESTS="${SKIP_TESTS:-0}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"
WITH_REAL=0
CONFIG="configs/default.yaml"

usage() {
  sed -n '2,20p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tests)      SKIP_TESTS=1; shift ;;
    --skip-figures)    SKIP_FIGURES=1; shift ;;
    --with-real-data)  WITH_REAL=1; shift ;;
    --config)          CONFIG="$2"; shift 2 ;;
    --n|--n-questions) N_QUESTIONS="$2"; shift 2 ;;
    -h|--help)         usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

# ----------------------------------------------------------------------
# 0. Locate conda.
# ----------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH." >&2
  exit 1
fi
echo "==> Conda:    $(command -v conda) ($(conda --version 2>&1 | head -1))"

# ----------------------------------------------------------------------
# 1. Verify the `grag` environment exists, and capture its Python path.
# ----------------------------------------------------------------------
CONDA_PY="$(conda run -n "$CONDA_ENV" which python 2>/dev/null || true)"
if [[ -z "$CONDA_PY" ]]; then
  echo "ERROR: conda env '$CONDA_ENV' not found.  Create it with:" >&2
  echo "       conda create -n $CONDA_ENV python=3.10 -y" >&2
  echo "       conda run -n $CONDA_ENV pip install numpy pyyaml networkx matplotlib pandas openai datasets" >&2
  exit 1
fi
echo "==> Conda env: $CONDA_ENV"
echo "==> Python:    $CONDA_PY ($("$CONDA_PY" --version 2>&1))"

# Sanity check: confirm we're inside the right env by using `conda run`.
CHECK_ENV="$(conda run -n "$CONDA_ENV" python -c 'import os; print(os.environ.get("CONDA_DEFAULT_ENV",""))' 2>/dev/null || true)"
if [[ "$CHECK_ENV" != "$CONDA_ENV" ]]; then
  echo "ERROR: 'conda run -n $CONDA_ENV' did not enter the right env (got '$CHECK_ENV')." >&2
  exit 1
fi

# Verify essential deps are present.
echo "==> Verifying dependencies ..."
conda run -n "$CONDA_ENV" python - <<'PY'
import importlib, sys
missing = []
for m in ("numpy", "yaml", "networkx", "matplotlib", "pandas", "openai", "datasets"):
    try:
        importlib.import_module(m)
    except ImportError:
        missing.append(m)
if missing:
    print("MISSING:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
print("All required packages are installed.")
PY

# Single helper: every command from here on is funnelled through the env.
# We deliberately use $CONDA_PY (absolute path) instead of `conda run -n`
# so the venv-isolation guarantees don't depend on PATH or conda's TTY
# quirks (e.g. with `set -e` and subprocess streaming).
VENV_RUN() { "$CONDA_PY" "$@"; }

export PYTHONPATH="$HERE:${PYTHONPATH:-}"
export MPLCONFIGDIR="$HERE/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# ----------------------------------------------------------------------
# 2. Unit tests + sanity check.
# ----------------------------------------------------------------------
if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "==> Running unit tests under $CONDA_PY"
  VENV_RUN tests/test_signals.py
  VENV_RUN tests/test_controller.py
  VENV_RUN scripts/sanity_check.py
fi

# ----------------------------------------------------------------------
# 3. Data
# ----------------------------------------------------------------------
mkdir -p data/raw results
if [[ "$WITH_REAL" -eq 0 ]]; then
  if [[ ! -f data/sample_hotpotqa.json ]]; then
    echo "==> Generating synthetic HotpotQA-style data (n=$N_QUESTIONS)"
    VENV_RUN scripts/build_sample_data.py --output data/sample_hotpotqa.json --n "$N_QUESTIONS"
  fi
  VENV_RUN - <<'PY'
import yaml, pathlib
p = pathlib.Path("configs/default.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["dataset"]["name"] = "synthetic"
cfg["dataset"]["synthetic_path"] = "data/sample_hotpotqa.json"
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  DATASET_PATH="data/sample_hotpotqa.json"
  SYNTH_FLAG="--synthetic"
else
  if [[ ! -f data/raw/hotpot_train_v1.1.json ]]; then
    echo "ERROR: --with-real-data requested but data/raw/hotpot_train_v1.1.json not found." >&2
    echo "       Run first:  conda run -n $CONDA_ENV python scripts/download_hotpotqa.py --split train" >&2
    exit 2
  fi
  VENV_RUN - <<'PY'
import yaml, pathlib
p = pathlib.Path("configs/default.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["dataset"]["name"] = "hotpotqa"
cfg["dataset"]["path"] = "data/raw/hotpot_train_v1.1.json"
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  DATASET_PATH="data/raw/hotpot_train_v1.1.json"
  SYNTH_FLAG=""
fi

# ----------------------------------------------------------------------
# 4. Build the unified knowledge graph.
# ----------------------------------------------------------------------
echo "==> Building knowledge graph"
VENV_RUN scripts/build_graph.py --dataset "$DATASET_PATH" \
    --output data/graph.json $SYNTH_FLAG --limit "${N_QUESTIONS}"

# ----------------------------------------------------------------------
# 5. Run the four Phase 1 experiments.
# ----------------------------------------------------------------------
echo "==> Running Phase 1 experiments (E1, E3, E5, E7)"
VENV_RUN scripts/run_phase1.py --config "$CONFIG" --n_questions "$N_QUESTIONS"

# ----------------------------------------------------------------------
# 6. Summarise + figures.
# ----------------------------------------------------------------------
echo "==> Summarising results"
VENV_RUN scripts/analyze_results.py

if [[ "$SKIP_FIGURES" -eq 0 ]]; then
  echo "==> Generating PDF figures (matplotlib if available)"
  VENV_RUN scripts/make_figures.py || echo "    figure generation failed"
fi

echo
echo "==> Done.  All commands ran under conda env '$CONDA_ENV'."
echo "    Python: $CONDA_PY"
echo "    Results: results/, trajectories: results/trajectories/."
ls -1 results/ 2>/dev/null || true
