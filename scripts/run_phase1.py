"""Phase 1 runner: E1, E3, E5, E7.

Reuses the shared ``runner.run_experiment`` helper to produce per-query CSVs
and method-level summaries for the Phase 1 minimum viable experiments.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runner import run_experiment  # noqa: E402

DEFAULT_METHODS_E1 = [
    "bm25",
    "fixed_hop_1", "fixed_hop_2", "fixed_hop_3", "fixed_hop_4",
    "graphrag", "react_graphrag", "ea_graphrag",
]
DEFAULT_METHODS_E3 = [
    "fixed_hop_1", "fixed_hop_2", "fixed_hop_3", "fixed_hop_4",
    "graphrag", "ea_graphrag",
]
DEFAULT_METHODS_E5 = [
    "ea_graphrag",
    "ea_graphrag_ablated_semantic",
    "ea_graphrag_ablated_structural",
    "ea_graphrag_ablated_reasoning",
    "ea_graphrag_ablated_consistency",
]
DEFAULT_METHODS_E7 = [
    "fixed_hop_1", "fixed_hop_2", "fixed_hop_3", "fixed_hop_4",
    "graphrag", "ea_graphrag",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiments", nargs="+", default=["e1", "e3", "e5", "e7"])
    parser.add_argument("--results", default="results")
    parser.add_argument("--trajectories", default="results/trajectories")
    parser.add_argument("--n_questions", type=int, default=None)
    args = parser.parse_args()

    plan = {
        "e1": (DEFAULT_METHODS_E1, "e1_overall.csv"),
        "e3": (DEFAULT_METHODS_E3, "e3_efficiency.csv"),
        "e5": (DEFAULT_METHODS_E5, "e5_ablation.csv"),
        "e7": (DEFAULT_METHODS_E7, "e7_complexity.csv"),
    }
    for exp in args.experiments:
        methods, csv_name = plan[exp]
        output_csv = os.path.join(args.results, csv_name)
        print(f"\n=== {exp.upper()} ({len(methods)} methods) ===")
        run_experiment(methods, args.config, exp, output_csv, args.trajectories,
                        n_questions=args.n_questions)


if __name__ == "__main__":
    main()
