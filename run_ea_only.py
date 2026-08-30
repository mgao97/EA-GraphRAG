"""Quick runner: only the EA-GraphRAG method (ea_graphrag) on a given dataset.

This bypasses the full E1 comparison table (which includes bm25, fixed_hop_*,
graphrag, react_graphrag) and runs ONLY the EA-GraphRAG method, so the effect
of the graph-pruning / beam-expansion optimisation can be validated fast.

Usage:
    python scripts/run_ea_only.py <dataset> <n> [config] [method]
    e.g.
      python scripts/run_ea_only.py musique 50 configs/optimized_11435.yaml ea_graphrag
      python scripts/run_ea_only.py musique 50 configs/ea_v1_11438.yaml ea_graphrag_v1
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runner import run_experiment  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python scripts/run_ea_only.py <dataset> <n> [config] [method]")
        sys.exit(1)
    dataset = sys.argv[1]
    n = int(sys.argv[2])
    cfg_path = sys.argv[3] if len(sys.argv) > 3 else "configs/optimized_11435.yaml"
    method = sys.argv[4] if len(sys.argv) > 4 else "ea_graphrag"

    # Point at the EA method only.
    methods = [method]
    experiment = "e1"
    out = f"results/{method}_{dataset}_minimax.csv"
    traj = f"logs/traj_{method}_{dataset}"

    # Override the dataset name from the CLI so the runner loads the correct
    # questions file (the YAML may hard-code a different dataset.name).
    cfg_overrides = {"dataset": {"name": dataset}}

    print(f"[ea_only] dataset={dataset} n={n} cfg={cfg_path} method={methods}")
    summary = run_experiment(
        methods=methods,
        cfg_path=cfg_path,
        experiment=experiment,
        output_csv=out,
        trajectory_dir=traj,
        n_questions=n,
        cfg_overrides=cfg_overrides,
    )
    print(f"[ea_only] done. summary written to {out}")
    print(summary)


if __name__ == "__main__":
    main()
