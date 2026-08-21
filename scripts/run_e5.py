"""Convenience wrapper for E5 (ablation)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runner import run_experiment  # noqa: E402

METHODS_E5 = [
    "ea_graphrag",
    "ea_graphrag_ablated_semantic",
    "ea_graphrag_ablated_structural",
    "ea_graphrag_ablated_reasoning",
    "ea_graphrag_ablated_consistency",
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n_questions", type=int, default=None)
    args = parser.parse_args()
    run_experiment(METHODS_E5, args.config, "e5",
                    "results/e5_ablation.csv",
                    "results/trajectories",
                    n_questions=args.n_questions)


if __name__ == "__main__":
    main()
