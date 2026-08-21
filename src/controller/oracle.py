"""Oracle controller.

Uses the gold supporting facts to make the optimal next move.  This isolates
controller quality from retrieval quality in controlled experiments (see
``experiment.md`` §25).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from ..data.graph_builder import GraphBuilder
from ..evidence.state import EvidenceState, EvidenceStep, Trajectory
from ..evidence.sufficiency import SufficiencyConfig, should_stop
from ..utils.embedding import BaseEmbedder
from .actions import Action, ActionExecutor, ActionResult
from .eac import EvidenceAcquisitionController, reasoning_completeness_covered


class OracleController:
    """Gold-only oracle for ablation experiments.

    The oracle picks the action that minimises the gap between the current
    evidence and the gold supporting facts:

        * If 0 gold facts are covered -> ``RETRIEVE``.
        * If 1 gold fact is covered -> ``EXPAND``.
        * If 2+ gold facts are covered but a bridge is missing -> ``BRIDGE``.
        * Otherwise -> ``STOP``.
    """

    def __init__(self, executor: ActionExecutor,
                  gold_titles: Set[str],
                  title_to_node: Dict[str, str],
                  sufficiency: Optional[SufficiencyConfig] = None,
                  max_iterations: int = 6):
        self.executor = executor
        self.gold_titles = set(gold_titles)
        self.title_to_node = dict(title_to_node)
        self.sufficiency = sufficiency or SufficiencyConfig()
        self.max_iterations = max_iterations

    def _decision(self, state: EvidenceState) -> Action:
        covered = [t for t in self.gold_titles
                   if self.title_to_node.get(t) in state.nodes]
        if not covered:
            return Action.RETRIEVE
        if len(covered) == len(self.gold_titles):
            return Action.STOP
        if len(covered) == 1:
            return Action.EXPAND
        # Some gold titles are missing -> try bridging.
        missing = [t for t in self.gold_titles
                   if self.title_to_node.get(t) not in state.nodes]
        if missing:
            return Action.BRIDGE
        return Action.STOP

    def run(self, query: str, qid: str, gold_answer: str = "",
            trajectory: Optional[Trajectory] = None) -> tuple[EvidenceState, Trajectory]:
        state = EvidenceState(query=query)
        trajectory = trajectory or Trajectory(qid=qid, question=query,
                                              gold_answer=gold_answer,
                                              method="oracle")
        for it in range(self.max_iterations):
            state.iteration = it
            action = self._decision(state)
            if action == Action.STOP:
                # Track coverage at stopping time.
                state.reasoning_coverage = reasoning_completeness_covered(
                    state, self.title_to_node, self.gold_titles)
                state.sufficiency_score = 1.0
                step = state.as_step(action=Action.STOP.value, action_target=[])
                trajectory.add_step(step)
                trajectory.stop_reason = "oracle_stop"
                break
            result = self.executor.run(action, state)
            state.reasoning_coverage = reasoning_completeness_covered(
                state, self.title_to_node, self.gold_titles)
            state.sufficiency_score = 1.0 if state.reasoning_coverage >= 0.99 else 0.0
            step = state.as_step(action=action.value,
                                  action_target=result.target,
                                  new_nodes=result.new_nodes,
                                  new_edges=result.new_edges,
                                  retrieval_score=result.score)
            trajectory.add_step(step)
        else:
            trajectory.stop_reason = "max_iterations"
        trajectory.final_evidence = list(state.nodes)
        trajectory.total_cost = {
            "nodes": len(state.nodes),
            "edges": len(state.edges),
            "tokens": state.token_count,
            "retrieval_calls": max(0, len(trajectory.steps) - 1),
            "iterations": len(trajectory.steps),
        }
        return state, trajectory
