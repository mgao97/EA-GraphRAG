"""EA-GraphRAG: the full Evidence-Aware GraphRAG method."""
from __future__ import annotations

from typing import Optional, Sequence, Set

from ..controller.actions import Action, ActionExecutor
from ..controller.eac import EvidenceAcquisitionController, AblationConfig
from ..data.hotpotqa import HotpotQAExample
from ..evidence.state import EvidenceState, Trajectory
from ..evidence.sufficiency import SufficiencyConfig
from .base import BaseMethod, MethodOutput


class EAGraphRAGMethod(BaseMethod):
    name = "ea_graphrag"

    def __init__(self, graph, embedder, llm, top_k_retrieve: int = 5,
                  expand_hops: int = 2, bridge_max_hops: int = 3,
                  max_iterations: int = 6,
                  sufficiency: Optional[SufficiencyConfig] = None,
                  ablation: Optional[AblationConfig] = None,
                  gold_titles: Optional[Set[str]] = None,
                  title_to_node: Optional[dict] = None,
                  **kwargs):
        super().__init__(graph, embedder, llm)
        self.executor = ActionExecutor(graph, embedder,
                                         top_k_retrieve=top_k_retrieve,
                                         expand_hops=expand_hops,
                                         bridge_max_hops=bridge_max_hops)
        self.sufficiency = sufficiency or SufficiencyConfig()
        self.ablation = ablation or AblationConfig()
        self.max_iterations = max_iterations
        self.gold_titles = gold_titles or set()
        self.title_to_node = title_to_node or {}
        self.controller = EvidenceAcquisitionController(
            embedder=embedder,
            executor=self.executor,
            sufficiency=self.sufficiency,
            ablation=self.ablation,
            max_iterations=self.max_iterations,
            gold_titles=self.gold_titles,
            title_to_node=self.title_to_node,
        )

    def answer(self, example: HotpotQAExample) -> MethodOutput:
        # Refresh gold-titles for this example so the controller has access.
        if self.gold_titles != set(example.supporting_titles):
            self.controller.gold_titles = set(example.supporting_titles)
        trajectory = Trajectory(qid=example.qid, question=example.question,
                                 gold_answer=example.answer,
                                 method=self.name)
        state, trajectory = self.controller.run(
            example.question, example.qid, example.answer, trajectory=trajectory)
        pred = self._llm_answer(state, example)
        trajectory.final_answer = pred
        from ..eval.metrics import answer_metrics
        return MethodOutput(qid=example.qid, question=example.question,
                             gold_answer=example.answer,
                             predicted_answer=pred,
                             trajectory=trajectory,
                             metrics=answer_metrics(pred, example.answer))
