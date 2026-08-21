"""Trajectory writer.

Each query writes a ``trajectory.json`` file inside ``results/trajectories/``.
The writer can be reused by all baselines.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from ..evidence.state import Trajectory
from ..utils.io import ensure_dir, write_json


class TrajectoryWriter:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    def write(self, trajectory: Trajectory, experiment: str = "e1") -> Path:
        out_dir = ensure_dir(self.root / experiment)
        path = out_dir / f"{trajectory.method}__{trajectory.qid}.json"
        write_json(path, trajectory.to_dict())
        return path
