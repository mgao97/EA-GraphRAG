"""Fixed-hop baseline.

Retrieves the top-k nodes by semantic similarity and then expands a fixed
number of hops around them.  Does **not** compute sufficiency, structural
gain, or consistency.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..controller.actions import Action, ActionExecutor
from ..data.hotpotqa import HotpotQAExample
from ..evidence.state import EvidenceState, Trajectory
from ..utils.embedding import BaseEmbedder
from .base import BaseMethod, MethodOutput


class FixedHopMethod(BaseMethod):
    name = "fixed_hop"

    def __init__(self, graph, embedder, llm, hops: int = 2,
                  top_k_retrieve: int = 5, **kwargs):
        super().__init__(graph, embedder, llm)
        self.hops = hops
        self.executor = ActionExecutor(graph, embedder,
                                         top_k_retrieve=top_k_retrieve,
                                         expand_hops=hops)

    def answer(self, example: HotpotQAExample) -> MethodOutput:
        trajectory = Trajectory(qid=example.qid, question=example.question,
                                 gold_answer=example.answer,
                                 method=f"fixed_hop_{self.hops}")
        state = EvidenceState(query=example.question)
        # Step 1: retrieve
        r = self.executor.run(Action.RETRIEVE, state)
        trajectory.add_step(state.as_step(action=Action.RETRIEVE.value,
                                            action_target=r.target,
                                            new_nodes=r.new_nodes,
                                            new_edges=r.new_edges,
                                            retrieval_score=r.score))
        # Step 2: expand
        r = self.executor.run(Action.EXPAND, state, hops=self.hops)
        trajectory.add_step(state.as_step(action=Action.EXPAND.value,
                                            action_target=r.target,
                                            new_nodes=r.new_nodes,
                                            new_edges=r.new_edges,
                                            retrieval_score=r.score))
        # Always STOP after fixed hops.
        trajectory.add_step(state.as_step(action=Action.STOP.value,
                                            action_target=[]))
        trajectory.final_evidence = list(state.nodes)
        trajectory.total_cost = {
            "nodes": len(state.nodes),
            "edges": len(state.edges),
            "tokens": state.token_count,
            "retrieval_calls": 2,
            "iterations": 3,
        }
        trajectory.stop_reason = "fixed_hops_done"
        pred = self._llm_answer(state, example)
        trajectory.final_answer = pred
        from ..eval.metrics import answer_metrics
        metrics = answer_metrics(pred, example.answer)
        return MethodOutput(qid=example.qid, question=example.question,
                             gold_answer=example.answer,
                             predicted_answer=pred,
                             trajectory=trajectory,
                             metrics=metrics)
