"""Evidence state representation.

The evidence state at time step ``t`` is ::

    S_t = (V_E, E_E, semantic, structural, reasoning, consistency)

where ``V_E`` are the evidence nodes, ``E_E`` are the evidence edges, and the
four signals capture semantic relevance, structural information gain, reasoning
completeness, and evidence consistency respectively.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceStep:
    step: int
    evidence_nodes: List[str] = field(default_factory=list)
    evidence_edges: List[List[str]] = field(default_factory=list)

    semantic_score: float = 0.0
    structural_gain: float = 0.0
    structural_entropy: float = 0.0
    reasoning_coverage: float = 0.0
    consistency_score: float = 0.0

    sufficiency_score: float = 0.0
    sufficient: bool = False

    action: str = ""
    action_target: List[str] = field(default_factory=list)

    newly_retrieved_nodes: List[str] = field(default_factory=list)
    newly_retrieved_edges: List[List[str]] = field(default_factory=list)
    retrieval_score: float = 0.0

    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceState:
    """Mutable working state passed to the controller."""
    query: str
    nodes: List[str] = field(default_factory=list)
    edges: List[List[str]] = field(default_factory=list)
    texts: Dict[str, str] = field(default_factory=dict)

    semantic_score: float = 0.0
    structural_gain: float = 0.0
    structural_entropy: float = 0.0
    reasoning_coverage: float = 0.0
    consistency_score: float = 0.0
    sufficiency_score: float = 0.0

    iteration: int = 0
    token_count: int = 0

    def add_node(self, node_id: str, text: str = "") -> None:
        if node_id not in self.nodes:
            self.nodes.append(node_id)
        if text:
            self.texts[node_id] = text

    def add_edge(self, src: str, dst: str) -> None:
        pair = [src, dst]
        if pair not in self.edges:
            self.edges.append(pair)

    def copy(self) -> "EvidenceState":
        return EvidenceState(
            query=self.query,
            nodes=list(self.nodes),
            edges=[list(e) for e in self.edges],
            texts=dict(self.texts),
            semantic_score=self.semantic_score,
            structural_gain=self.structural_gain,
            structural_entropy=self.structural_entropy,
            reasoning_coverage=self.reasoning_coverage,
            consistency_score=self.consistency_score,
            sufficiency_score=self.sufficiency_score,
            iteration=self.iteration,
            token_count=self.token_count,
        )

    def as_step(self, action: str = "", action_target: Optional[List[str]] = None,
                 new_nodes: Optional[List[str]] = None,
                 new_edges: Optional[List[List[str]]] = None,
                 retrieval_score: float = 0.0) -> EvidenceStep:
        return EvidenceStep(
            step=self.iteration,
            evidence_nodes=list(self.nodes),
            evidence_edges=[list(e) for e in self.edges],
            semantic_score=self.semantic_score,
            structural_gain=self.structural_gain,
            structural_entropy=self.structural_entropy,
            reasoning_coverage=self.reasoning_coverage,
            consistency_score=self.consistency_score,
            sufficiency_score=self.sufficiency_score,
            sufficient=self.sufficiency_score >= 1.0,
            action=action,
            action_target=list(action_target or []),
            newly_retrieved_nodes=list(new_nodes or []),
            newly_retrieved_edges=[list(e) for e in (new_edges or [])],
            retrieval_score=float(retrieval_score),
            token_count=self.token_count,
        )


@dataclass
class Trajectory:
    qid: str
    question: str
    gold_answer: str
    method: str
    steps: List[EvidenceStep] = field(default_factory=list)
    final_answer: str = ""
    final_evidence: List[str] = field(default_factory=list)
    total_cost: Dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: EvidenceStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "method": self.method,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "final_evidence": list(self.final_evidence),
            "total_cost": dict(self.total_cost),
            "stop_reason": self.stop_reason,
            "metadata": dict(self.metadata),
        }
