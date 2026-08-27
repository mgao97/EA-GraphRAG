"""Quick runner: only EA-GraphRAG-v1 on a given dataset.

Usage:
    python run_ea_v1.py <dataset> <n> [config]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runner import run_experiment  # noqa: E402


DATASET_PATHS = {
    "hotpotqa": "data/raw/hotpot_dev_v1.1.json",
    "musique": "data/raw/musique/questions.json",
    "2wikimultihopqa": "data/raw/2wikimultihopqa/questions.json",
}


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python run_ea_v1.py <dataset> <n> [config]")
        sys.exit(1)
    dataset = sys.argv[1]
    n = int(sys.argv[2])
    cfg_path = sys.argv[3] if len(sys.argv) > 3 else "configs/ea_v1_11438.yaml"

    if dataset not in DATASET_PATHS:
        print(f"unknown dataset {dataset}; choose from {list(DATASET_PATHS)}")
        sys.exit(1)

    methods = ["ea_graphrag_v1"]
    experiment = "e1"
    out = f"results/ea_v1_only_{dataset}.csv"
    traj = f"logs/traj_ea_v1_only_{dataset}"
    cfg_overrides = {
        "dataset": {"name": dataset, "path": DATASET_PATHS[dataset]}
    }

    print(f"[ea_v1] dataset={dataset} n={n} cfg={cfg_path} method={methods}")
    summary = run_experiment(
        methods=methods,
        cfg_path=cfg_path,
        experiment=experiment,
        output_csv=out,
        trajectory_dir=traj,
        n_questions=n,
        cfg_overrides=cfg_overrides,
    )
    print(f"[ea_v1] done. summary written to {out}")
    print(summary)


if __name__ == "__main__":
    main()
