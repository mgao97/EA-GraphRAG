"""Vanilla GraphRAG baseline.

Performs one semantic retrieval call and a single graph expansion step.
Acts as a *relevance-driven* baseline that does not adapt to sufficiency.
"""
from __future__ import annotations

from ..controller.actions import Action, ActionExecutor
from ..data.hotpotqa import HotpotQAExample
from ..evidence.state import EvidenceState, Trajectory
from .base import BaseMethod, MethodOutput


class GraphRAGMethod(BaseMethod):
    name = "graphrag"

    def __init__(self, graph, embedder, llm, expand_hops: int = 2,
                  top_k_retrieve: int = 8, **kwargs):
        super().__init__(graph, embedder, llm)
        self.executor = ActionExecutor(graph, embedder,
                                         top_k_retrieve=top_k_retrieve,
                                         expand_hops=expand_hops)

    def answer(self, example: HotpotQAExample) -> MethodOutput:
        trajectory = Trajectory(qid=example.qid, question=example.question,
                                 gold_answer=example.answer,
                                 method=self.name)
        state = EvidenceState(query=example.question)
        # Single retrieve + expand pass
        r1 = self.executor.run(Action.RETRIEVE, state)
        trajectory.add_step(state.as_step(action=Action.RETRIEVE.value,
                                            action_target=r1.target,
                                            new_nodes=r1.new_nodes,
                                            new_edges=r1.new_edges,
                                            retrieval_score=r1.score))
        r2 = self.executor.run(Action.EXPAND, state)
        trajectory.add_step(state.as_step(action=Action.EXPAND.value,
                                            action_target=r2.target,
                                            new_nodes=r2.new_nodes,
                                            new_edges=r2.new_edges,
                                            retrieval_score=r2.score))
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
        trajectory.stop_reason = "single_pass_done"
        pred = self._llm_answer(state, example)
        trajectory.final_answer = pred
        from ..eval.metrics import answer_metrics
        return MethodOutput(qid=example.qid, question=example.question,
                             gold_answer=example.answer,
                             predicted_answer=pred,
                             trajectory=trajectory,
                             metrics=answer_metrics(pred, example.answer))
