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
N_QUESTIONS="${N_QUESTIONS:-200}"
SKIP_TESTS="${SKIP_TESTS:-0}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"
WITH_REAL=0
CONFIG="configs/default.yaml"

usage() {
  sed -n '2,20p' "$0"
  exit 0
}

DATASET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tests)      SKIP_TESTS=1; shift ;;
    --skip-figures)    SKIP_FIGURES=1; shift ;;
    --with-real-data)  WITH_REAL=1; shift ;;
    --config)          CONFIG="$2"; shift 2 ;;
    --n|--n-questions) N_QUESTIONS="$2"; shift 2 ;;
    -h|--help)         usage ;;
    *) DATASET="$1"; shift ;;   # first positional = dataset name
  esac
done

if [[ -n "$DATASET" && "$DATASET" != "musique" && "$DATASET" != "2wikimultihopqa" && "$DATASET" != "hotpotqa" ]]; then
  echo "ERROR: dataset must be one of: musique | 2wikimultihopqa | hotpotqa (got '$DATASET')" >&2
  exit 2
fi

if [[ "${CONFIG}" = /* ]]; then
  CONFIG_PATH="${CONFIG}"
else
  CONFIG_PATH="${HERE}/${CONFIG}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

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

export PYTHONNOUSERSITE=1
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
export MPLCONFIGDIR="$HERE/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
ACTIVE_CONFIG="$(mktemp "${MPLCONFIGDIR}/run_config.XXXXXX.yaml")"
trap 'rm -f "${ACTIVE_CONFIG}"' EXIT

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
  SRC_CONFIG="${CONFIG_PATH}" DST_CONFIG="${ACTIVE_CONFIG}" VENV_RUN - <<'PY'
import os
import yaml
from pathlib import Path

p = Path(os.environ["SRC_CONFIG"])
out = Path(os.environ["DST_CONFIG"])
cfg = yaml.safe_load(p.read_text())
cfg["dataset"]["name"] = "synthetic"
cfg["dataset"]["synthetic_path"] = "data/sample_hotpotqa.json"
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  DATASET_PATH="data/sample_hotpotqa.json"
  SYNTH_FLAG="--synthetic"
else
  # Fair comparison: use the SAME HotpotQA dev split as the baselines
  # (baseline/* evaluate on data/raw/hotpot_dev_v1.1.json, n=200).
  REAL_PATH="data/raw/hotpot_dev_v1.1.json"
  if [[ ! -f "$REAL_PATH" ]]; then
    echo "ERROR: --with-real-data requested but $REAL_PATH not found." >&2
    echo "       Make sure data/raw/hotpot_dev_v1.1.json exists (used by baselines too)." >&2
    exit 2
  fi
  SRC_CONFIG="${CONFIG_PATH}" DST_CONFIG="${ACTIVE_CONFIG}" VENV_RUN - <<'PY'
import os
import yaml
from pathlib import Path

p = Path(os.environ["SRC_CONFIG"])
out = Path(os.environ["DST_CONFIG"])
cfg = yaml.safe_load(p.read_text())
cfg["dataset"]["name"] = "hotpotqa"
cfg["dataset"]["path"] = "data/raw/hotpot_dev_v1.1.json"
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  DATASET_PATH="$REAL_PATH"
  SYNTH_FLAG=""
fi

# ----------------------------------------------------------------------
# 4. Build the unified knowledge graph (skipped when evaluating on a
#    per-dataset corpus: run_phase1 builds the KG from the shared corpus).
# ----------------------------------------------------------------------
if [[ -n "$DATASET" ]]; then
  echo "==> Dataset '$DATASET': graph will be built from the shared per-dataset corpus inside run_phase1."
else
  echo "==> Building knowledge graph"
  VENV_RUN scripts/build_graph.py --dataset "$DATASET_PATH" \
      --output data/graph.json $SYNTH_FLAG --limit "${N_QUESTIONS}"
fi

# ----------------------------------------------------------------------
# 5. Run the four Phase 1 experiments.
# ----------------------------------------------------------------------
echo "==> Running Phase 1 experiments (E1, E3, E5, E7)"
if [[ -n "$DATASET" ]]; then
  echo "==> Dataset mode: $DATASET (n=$N_QUESTIONS)"
  VENV_RUN scripts/run_phase1.py --config "$ACTIVE_CONFIG" --n_questions "$N_QUESTIONS" --dataset "$DATASET"
else
  echo "==> Effective config: ${CONFIG_PATH}"
  VENV_RUN scripts/run_phase1.py --config "$ACTIVE_CONFIG" --n_questions "$N_QUESTIONS"
fi

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
