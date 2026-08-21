"""Convenience wrapper for E3 only."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runner import run_experiment  # noqa: E402

METHODS_E3 = [
    "fixed_hop_1", "fixed_hop_2", "fixed_hop_3", "fixed_hop_4",
    "graphrag", "ea_graphrag",
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n_questions", type=int, default=None)
    args = parser.parse_args()
    run_experiment(METHODS_E3, args.config, "e3",
                    "results/e3_efficiency.csv",
                    "results/trajectories",
                    n_questions=args.n_questions)


if __name__ == "__main__":
    main()
