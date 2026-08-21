# EA-GraphRAG — Evidence-Aware GraphRAG

Implementation of the experimental protocol defined in [`experiment.md`](experiment.md).

> **Core idea**: re-formulate GraphRAG as a **sufficiency-aware evidence
> acquisition** problem.  An *Evidence Acquisition Controller* (EAC) inspects
> four signals — Semantic Relevance, Structural Information Gain, Reasoning
> Completeness, Evidence Consistency — and dynamically chooses among
> `RETRIEVE`, `EXPAND`, `BRIDGE`, `VERIFY`, `STOP` actions to minimise
> evidence cost subject to a sufficiency threshold.

The first-phase MVP focuses on **HotpotQA** and four experiments
(`E1`, `E3`, `E5`, `E7`) covering overall QA, efficiency, ablation, and
reasoning complexity.

---

> **All commands in this README run inside the conda environment `grag`**
> (Python 3.10) — never the system Python or any other conda env.  The
> `run.sh` wrapper handles `conda run -n grag` automatically.  If you run a
> command yourself, prefix it with `conda run -n grag …`.

## 1. Quick start (offline, no LLM needed)

The project ships with everything needed to run a 30-example smoke test.
**All commands run inside the `grag` conda env**, never the system Python.

```bash
cd /Users/santa/Desktop/EA-GraphRAG

# 1. one-shot smoke test (offline, no LLM)
./run.sh

# 2. explicit pre-flight check
conda run -n grag python scripts/sanity_check.py
```

`./run.sh` will:

1. Verify the `grag` env exists and has the right deps.
2. Generate a 30-example synthetic HotpotQA-style dataset.
3. Build the unified knowledge graph.
4. Run the four Phase 1 experiments (E1, E3, E5, E7).
5. Aggregate CSVs into summary tables.
6. Generate PDF figures.
7. Run unit tests.

---

## 2. Run on real HotpotQA + a local LLM

This is the workflow you'll use most often.

### 2.1 Install Ollama + pull models

```bash
# macOS (Apple Silicon / Intel)
./scripts/setup_local_llm.sh                        # default: qwen2.5:7b + nomic-embed-text
./scripts/setup_local_llm.sh --models llama3.1:8b qwen2.5:7b-instruct:14b

# Linux
curl -fsSL https://ollama.com/install.sh | sh
./scripts/setup_local_llm.sh
```

The script installs Ollama, starts the server, and pulls both a chat model
and an embedding model.  Ollama exposes an OpenAI-compatible API at
`http://localhost:11434/v1`, which `EA-GraphRAG` talks to directly.

### 2.2 Download HotpotQA

```bash
conda run -n grag pip install datasets        # needed by the downloader
conda run -n grag python scripts/download_hotpotqa.py --split train
# or for a smaller file:
conda run -n grag python scripts/download_hotpotqa.py --split dev --limit 1000
```

The file is saved to `data/raw/hotpot_<split>_v1.1.json` and validated
against the official schema.

### 2.3 Run

The repo ships with two pre-made configs:

* `configs/local.yaml` — Ollama defaults (`qwen2.5:7b`, `nomic-embed-text`)
* `configs/openai.yaml` — Hosted OpenAI defaults (`gpt-4o-mini`,
  `text-embedding-3-small`)

```bash
conda run -n grag python scripts/run_phase1.py --config configs/local.yaml --n_questions 200
```

To switch to OpenAI later, just:

```bash
export OPENAI_API_KEY=sk-...
conda run -n grag python scripts/run_phase1.py --config configs/openai.yaml --n_questions 200
```

No code change is required — only the config and the `OPENAI_API_KEY`
environment variable.

### 2.4 Use vLLM / LM Studio / your own endpoint

`llm.base_url` and `embedding.base_url` in any YAML config override the
default OpenAI endpoint, so vLLM (`http://localhost:8000/v1`) and LM Studio
(`http://localhost:1234/v1`) work out of the box:

```yaml
llm:
  backend: openai
  base_url: http://localhost:8000/v1    # vLLM
  model: meta-llama/Llama-3.1-8B-Instruct

embedding:
  backend: openai
  base_url: http://localhost:8000/v1
  model: BAAI/bge-m3
```

Local servers do not require a real API key; the runner auto-substitutes a
placeholder.

---

## 3. Project layout

```
EA-GraphRAG/
├── experiment.md            # Full experimental design doc
├── README.md                # You are here
├── requirements.txt         # Core scientific stack
├── run.sh                   # One-shot offline Phase 1 pipeline
├── configs/
│   ├── default.yaml         # Offline defaults (dummy LLM + dummy embedder)
│   ├── local.yaml           # Ollama template
│   └── openai.yaml          # Hosted OpenAI template
├── src/
│   ├── data/                # HotpotQA loader + graph builder
│   ├── evidence/            # Evidence state + 4 signals + sufficiency
│   ├── controller/          # Actions, EAC, Oracle
│   ├── methods/             # Baselines + EA-GraphRAG
│   ├── llm/                 # LLM interfaces (dummy + OpenAI-compatible)
│   ├── eval/                # EM / F1 + trajectory writer
│   └── utils/               # Embedding, IO
├── scripts/
│   ├── build_sample_data.py # Synthetic HotpotQA-style data
│   ├── build_graph.py       # Build the unified KG
│   ├── download_hotpotqa.py # HotpotQA downloader (HF + mirrors)
│   ├── setup_local_llm.sh   # Install Ollama + pull models
│   ├── runner.py            # Method runner used by every script
│   ├── run_phase1.py        # Run E1+E3+E5+E7
│   ├── run_e1..e7.py        # Individual experiments
│   ├── analyze_results.py   # Aggregate CSVs into summary tables
│   ├── make_figures.py      # Optional PDF figures
│   └── sanity_check.py      # Pre-flight smoke test
├── tests/
│   ├── test_signals.py
│   └── test_controller.py
└── results/
    ├── e1_overall.csv      … per-experiment CSVs
    ├── *_summary.csv / _summary.json
    ├── trajectories/       # per-query trajectory.json files
    └── figures/            # PDF figures (if matplotlib available)
```

---

## 4. Method implementations

| Method              | Description                                                                                  |
|---------------------|----------------------------------------------------------------------------------------------|
| `fixed_hop_{k}`     | Retrieve once, expand k-hops, stop.  No controller.                                          |
| `graphrag`          | One retrieve + one expand pass.  Relevance-driven.                                          |
| `react_graphrag`    | ReAct agent that picks actions from `RETRIEVE/EXPAND/BRIDGE/VERIFY/STOP` via the LLM.        |
| `ea_graphrag`       | EA-GraphRAG with the full EAC.                                                               |
| `ea_graphrag_ablated_*` | EA-GraphRAG with one of the four signals disabled.                                       |
| `oracle`            | Gold-only controller used to isolate controller vs retrieval quality.                        |

All methods share the **same** underlying KG and retrieval primitives so the
ablations isolate the *controller* design.

## 5. Evidence state

The controller maintains an :class:`EvidenceState` with the following
signals, all in `[0, 1]`:

| Signal                     | Formula (Phase 1)                                              |
|----------------------------|----------------------------------------------------------------|
| Semantic relevance         | Mean of top-5 cosine similarities between query and evidence.  |
| Structural information gain | `0.6·ΔH + 0.4·(1 − exp(−new_nodes − new_edges))`.              |
| Reasoning completeness     | Fraction of gold supporting-fact titles covered (oracle).      |
| Evidence consistency       | `1 − conflict_ratio` over duplicated edges.                    |

Sufficiency follows `experiment.md` §6:

```
sufficient ⇔ semantic ≥ τ_sem  ∧  reasoning ≥ τ_reason  ∧  consistency ≥ τ_cons
stop      ⇔ sufficient          ∧  structural_gain < ΔH_min
```

The thresholds are in `configs/default.yaml → controller.sufficiency`.
The defaults (`τ_sem = 0.20`) are tuned for the deterministic dummy
embedder.  When you switch to a real embedder (BGE-M3 / OpenAI), **raise
`τ_sem` to ~0.55** so the semantic signal becomes meaningful.

## 6. Tests

```bash
conda run -n grag python tests/test_signals.py
conda run -n grag python tests/test_controller.py
conda run -n grag python scripts/sanity_check.py
```

## 7. Cost-aware LLM usage tips

A single Phase-1 run on 200 questions can make thousands of LLM calls if
you include `react_graphrag` and `ea_graphrag`.  To keep things sane:

```bash
# Start with a tiny smoke run.
conda run -n grag python scripts/run_phase1.py --config configs/local.yaml --n_questions 10
# Then scale up.
conda run -n grag python scripts/run_phase1.py --config configs/local.yaml --n_questions 200
```

You can also run a single experiment (much faster):

```bash
conda run -n grag python scripts/run_e5.py --config configs/local.yaml --n_questions 50
```

## 8. License & data

This implementation is for research purposes.  HotpotQA is distributed under
its original licence; see <https://hotpotqa.github.io/>.
