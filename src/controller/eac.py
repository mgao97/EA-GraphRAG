"""Evidence Acquisition Controller (EAC).

The controller implements the core sufficiency-driven acquisition loop:

    loop:
        compute signals (semantic, structural, reasoning, consistency)
        compute sufficiency
        if sufficient and marginal_gain is small -> STOP
        else: choose action via decision rule

The Phase 1 controller uses a *rule-based decision policy* so the experiments
are reproducible and oracle-friendly.  An LLM can be plugged in later by
implementing ``BaseLLM`` and switching the ``llm`` argument.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..evidence.signals import (
    evidence_consistency,
    reasoning_completeness,
    semantic_relevance,
    structural_entropy,
    structural_information_gain,
)
from ..evidence.state import EvidenceState, EvidenceStep, Trajectory
from ..evidence.sufficiency import (
    SufficiencyConfig,
    assess_sufficiency,
    should_stop,
)
from ..utils.embedding import BaseEmbedder
from .actions import Action, ActionExecutor, ActionResult


@dataclass
class AblationConfig:
    """Toggle the four signals on/off."""
    use_semantic: bool = True
    use_structural: bool = True
    use_reasoning: bool = True
    use_consistency: bool = True


@dataclass
class ControllerDecision:
    action: Action
    target: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "target": list(self.target),
            "rationale": self.rationale,
        }


class EvidenceAcquisitionController:
    """Main EAC.

    Parameters
    ----------
    embedder:
        Embedding model used for semantic retrieval and the semantic signal.
    executor:
        :class:`ActionExecutor` that actually performs graph retrieval.
    sufficiency:
        Threshold configuration.
    ablation:
        Toggle the four signals individually (used by the ablation experiment).
    gold_titles:
        Optional set of gold supporting-fact titles used to compute the
        *oracle* reasoning coverage.  When ``None`` we fall back to a
        heuristic that uses the predicted evidence-vs-query similarity.
    """

    def __init__(self, embedder: BaseEmbedder, executor: ActionExecutor,
                  sufficiency: Optional[SufficiencyConfig] = None,
                  ablation: Optional[AblationConfig] = None,
                  max_iterations: int = 6,
                  gold_titles: Optional[Set[str]] = None,
                  title_to_node: Optional[Dict[str, str]] = None):
        self.embedder = embedder
        self.executor = executor
        self.sufficiency = sufficiency or SufficiencyConfig()
        self.ablation = ablation or AblationConfig()
        self.max_iterations = int(max_iterations)
        self.gold_titles = gold_titles or set()
        self.title_to_node = title_to_node or {}

    # ------------------------------------------------------------------
    # signals
    # ------------------------------------------------------------------
    def _query_embedding(self, query: str):
        return self.embedder.embed([query])[0]

    def _node_embeddings(self, state: EvidenceState):
        if not state.nodes:
            return None
        texts = [state.texts.get(n, self.executor._node_text(n)) for n in state.nodes]
        return self.embedder.embed(texts)

    def update_signals(self, state: EvidenceState,
                        prev_entropy: float) -> None:
        q_emb = self._query_embedding(state.query)
        n_emb = self._node_embeddings(state)
        sem = semantic_relevance(state, q_emb, n_emb) if n_emb is not None else 0.0
        degrees = [self.executor.graph.graph.out_degree(n) +
                   self.executor.graph.graph.in_degree(n) for n in state.nodes]
        ent = structural_entropy(state, degrees) if degrees else 0.0
        gain = structural_information_gain(state, prev_entropy, ent,
                                              new_nodes=len(state.nodes),
                                              new_edges=len(state.edges))
        if self.gold_titles:
            reason = reasoning_completeness_covered(state, self.title_to_node,
                                                     self.gold_titles)
        else:
            reason = semantic_relevance(state, q_emb, n_emb) if n_emb is not None else 0.0
            # Use semantic as a proxy for "reasoning progress" when gold is
            # not available.
            reason = 0.5 * reason + 0.5 * min(1.0, len(state.nodes) / max(1, len(self.gold_titles or {2})))
        cons = evidence_consistency(state, pair_conflicts=[])
        # Apply ablation.
        if not self.ablation.use_semantic:
            sem = 0.0
        if not self.ablation.use_structural:
            gain = 0.0
            ent = 0.0
        if not self.ablation.use_reasoning:
            reason = 0.0
        if not self.ablation.use_consistency:
            cons = 0.0
        state.semantic_score = float(sem)
        state.structural_entropy = float(ent)
        state.structural_gain = float(gain)
        state.reasoning_coverage = float(reason)
        state.consistency_score = float(cons)
        # Compute the boolean sufficiency score (1.0 if all constraints met).
        from ..evidence.sufficiency import assess_sufficiency
        info = assess_sufficiency(state, self.sufficiency)
        state.sufficiency_score = 1.0 if info["sufficient"] else 0.0

    # ------------------------------------------------------------------
    # decision rule
    # ------------------------------------------------------------------
    def decide(self, state: EvidenceState,
                last_action: Optional[ActionResult] = None) -> ControllerDecision:
        # STOP first
        if should_stop(state, self.sufficiency):
            return ControllerDecision(Action.STOP, rationale="sufficient+low_gain")
        # If the previous action produced no new evidence, we have exhausted
        # the retrieval graph and must stop.
        if last_action is not None and not last_action.new_nodes and last_action.action in {
            Action.RETRIEVE, Action.EXPAND, Action.BRIDGE
        }:
            return ControllerDecision(Action.STOP, rationale="no_new_evidence")
        # Identify *which* constraint failed.
        info = assess_sufficiency(state, self.sufficiency)
        if not info["constraints"].get("semantic", True):
            return ControllerDecision(Action.RETRIEVE,
                                       rationale="semantic_below_threshold")
        if not info["constraints"].get("reasoning", True):
            # Need more evidence -> try BRIDGE if we have at least two clusters
            if len(state.nodes) >= 2:
                return ControllerDecision(Action.BRIDGE,
                                           rationale="reasoning_gap")
            return ControllerDecision(Action.EXPAND,
                                       rationale="reasoning_gap_no_clusters")
        if not info["constraints"].get("consistency", True):
            return ControllerDecision(Action.VERIFY,
                                       rationale="consistency_below_threshold")
        # Everything satisfied but gain is still high -> keep expanding.
        if state.structural_gain >= self.sufficiency.marginal_gain_threshold:
            return ControllerDecision(Action.EXPAND,
                                       rationale="gain_still_high")
        return ControllerDecision(Action.STOP, rationale="fallback_stop")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self, query: str, qid: str, gold_answer: str = "",
            trajectory: Optional[Trajectory] = None) -> Tuple[EvidenceState, Trajectory]:
        state = EvidenceState(query=query)
        trajectory = trajectory or Trajectory(qid=qid, question=query,
                                              gold_answer=gold_answer,
                                              method="ea_graphrag")
        prev_entropy = 0.0
        last_action: ActionResult = ActionResult(Action.STOP)
        decision = ControllerDecision(Action.STOP)
        for it in range(self.max_iterations):
            state.iteration = it
            self.update_signals(state, prev_entropy)
            decision = self.decide(state, last_action)
            if decision.action == Action.STOP:
                step = state.as_step(action=decision.action.value,
                                      action_target=decision.target,
                                      retrieval_score=0.0)
                trajectory.add_step(step)
                trajectory.stop_reason = decision.rationale
                last_action = None
                break
            last_action = self.executor.run(decision.action, state)
            step = state.as_step(action=last_action.action.value,
                                  action_target=last_action.target,
                                  new_nodes=last_action.new_nodes,
                                  new_edges=last_action.new_edges,
                                  retrieval_score=last_action.score)
            trajectory.add_step(step)
            prev_entropy = state.structural_entropy
            # If the action returned no new evidence, we are done (the decide
            # rule will short-circuit next iteration, but break here to avoid
            # a wasted iteration).
            if (not last_action.new_nodes and
                    last_action.action in {Action.RETRIEVE, Action.EXPAND, Action.BRIDGE}):
                # Record a STOP step with the reason.
                stop_decision = ControllerDecision(Action.STOP,
                                                     rationale="no_new_evidence")
                stop_step = state.as_step(action=Action.STOP.value,
                                            action_target=[],
                                            retrieval_score=0.0)
                trajectory.add_step(stop_step)
                trajectory.stop_reason = stop_decision.rationale
                break
        else:
            if trajectory.stop_reason == "":
                trajectory.stop_reason = "max_iterations_reached"
        trajectory.final_evidence = list(state.nodes)
        trajectory.total_cost = {
            "nodes": len(state.nodes),
            "edges": len(state.edges),
            "tokens": state.token_count,
            "retrieval_calls": max(0, len(trajectory.steps) - 1),
            "iterations": len(trajectory.steps),
        }
        return state, trajectory


def reasoning_completeness_covered(state: EvidenceState,
                                    title_to_node: Dict[str, str],
                                    gold_titles: Set[str]) -> float:
    covered = 0
    for title in gold_titles:
        node = title_to_node.get(title)
        if node and node in state.nodes:
            covered += 1
    return covered / len(gold_titles) if gold_titles else 1.0
