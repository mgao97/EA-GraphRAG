#!/usr/bin/env bash
# Install Ollama and pull recommended models for EA-GraphRAG.
#
# Usage:
#     ./scripts/setup_local_llm.sh                       # defaults
#     ./scripts/setup_local_llm.sh --models qwen2.5:7b llama3.1:8b
#     ./scripts/setup_local_llm.sh --help
#
# Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1
# which the EA-GraphRAG backend uses out of the box.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

MODELS=("qwen2.5:7b")
EMBED_MODEL="nomic-embed-text"

print_help() {
  cat <<USAGE
Usage: $0 [--models MODEL ...] [--embed-model MODEL]

Defaults to pulling qwen2.5:7b (chat) and nomic-embed-text (embedding).
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      MODELS=("$@")
      break ;;
    --embed-model)
      EMBED_MODEL="$2"
      shift 2 ;;
    -h|--help)
      print_help
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      print_help
      exit 1 ;;
  esac
done

# 1. Install Ollama if missing.
if ! command -v ollama >/dev/null 2>&1; then
  echo "==> Ollama not found, installing..."
  if [[ "$(uname)" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      brew install ollama
    else
      echo "ERROR: install Homebrew first (https://brew.sh/) or download Ollama manually." >&2
      exit 1
    fi
  else
    curl -fsSL https://ollama.com/install.sh | sh
  fi
else
  echo "==> Ollama already installed: $(ollama --version)"
fi

# 2. Start the Ollama server (idempotent).
if ! curl -fsS -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "==> Starting Ollama server..."
  if [[ "$(uname)" == "Darwin" ]]; then
    open -a Ollama 2>/dev/null || (nohup ollama serve >/tmp/ollama.log 2>&1 &)
  else
    nohup ollama serve >/tmp/ollama.log 2>&1 &
  fi
  # Wait for it to come up.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi
echo "==> Ollama API: http://localhost:11434/v1"

# 3. Pull models.
for m in "${MODELS[@]}"; do
  echo "==> Pulling model: $m"
  ollama pull "$m"
done

# 4. Pull embedding model if not already requested.
NEEDS_EMBED=1
for m in "${MODELS[@]}"; do
  if [[ "$m" == "$EMBED_MODEL" ]]; then NEEDS_EMBED=0; fi
done
if [[ $NEEDS_EMBED -eq 1 ]]; then
  echo "==> Pulling embedding model: $EMBED_MODEL"
  ollama pull "$EMBED_MODEL"
fi

# 5. Print a copy-pasteable config snippet.
cat <<YAML

==> Done. Suggested EA-GraphRAG config (configs/local.yaml):

embedding:
  backend: openai
  model: ${EMBED_MODEL}
  base_url: http://localhost:11434/v1
  api_key_env: OPENAI_API_KEY      # any non-empty value works for Ollama

llm:
  backend: openai
  provider: ollama
  model: ${MODELS[0]}
  temperature: 0.0

controller:
  sufficiency:
    tau_sem: 0.55                  # raise for real embeddings
    tau_reason: 0.66
    tau_cons: 0.85

dataset:
  name: hotpotqa
  path: data/raw/hotpot_train_v1.1.json
  limit: 200

Then run:
    conda run -n grag python scripts/run_phase1.py --config configs/local.yaml
YAML
