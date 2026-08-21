"""Unit tests for the four evidence signals + sufficiency."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.evidence.signals import (
    evidence_consistency,
    reasoning_completeness,
    semantic_relevance,
    structural_information_gain,
)
from src.evidence.state import EvidenceState
from src.evidence.sufficiency import SufficiencyConfig, assess_sufficiency, should_stop


def test_semantic_relevance_empty():
    state = EvidenceState(query="Who directed Inception?")
    val = semantic_relevance(state, np.zeros(8), np.zeros((0, 8)))
    assert val == 0.0


def test_semantic_relevance_matches_self():
    state = EvidenceState(query="hello world")
    q = np.array([1.0, 0.0, 0.0])
    node = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    val = semantic_relevance(state, q, node, top_k=2)
    assert 0.0 < val <= 1.0


def test_structural_information_gain_first_iter():
    state = EvidenceState(query="x")
    gain = structural_information_gain(state, prev_entropy=0.0,
                                        new_entropy=0.0,
                                        new_nodes=2, new_edges=3)
    assert gain == 1.0  # first iteration


def test_structural_information_gain_diminishing():
    state = EvidenceState(query="x")
    state.iteration = 3
    gain = structural_information_gain(state, prev_entropy=0.9,
                                        new_entropy=0.9,
                                        new_nodes=0, new_edges=0)
    assert 0.0 <= gain < 1.0


def test_reasoning_completeness():
    assert reasoning_completeness({"a", "b"}, {"a", "b", "c"}) == 2 / 3
    assert reasoning_completeness(set(), {"a"}) == 0.0
    assert reasoning_completeness({"a"}, set()) == 1.0


def test_evidence_consistency_perfect():
    state = EvidenceState(query="x")
    state.edges = [["a", "b"], ["c", "d"]]
    assert evidence_consistency(state, pair_conflicts=[]) == 1.0


def test_evidence_consistency_triple_duplicate_lowers_score():
    """Triple duplicate of an undirected edge lowers consistency."""
    state = EvidenceState(query="x")
    state.edges = [["a", "b"], ["b", "a"], ["a", "b"], ["b", "a"], ["c", "d"]]
    cons = evidence_consistency(state, pair_conflicts=[])
    # 2 unique edges, 1 has 4 occurrences => 1 true dup
    assert 0.0 < cons < 1.0


def test_evidence_consistency_bidirectional_no_conflict():
    """Single bidirectional edge pair should not be flagged as a duplicate."""
    state = EvidenceState(query="x")
    state.edges = [["a", "b"], ["b", "a"], ["c", "d"]]
    cons = evidence_consistency(state, pair_conflicts=[])
    assert cons == 1.0


def test_assess_sufficiency_constraints():
    state = EvidenceState(query="x")
    state.semantic_score = 0.4
    state.reasoning_coverage = 0.9
    state.consistency_score = 1.0
    cfg = SufficiencyConfig(tau_sem=0.5, tau_reason=0.6, tau_cons=0.8)
    info = assess_sufficiency(state, cfg)
    assert not info["sufficient"]
    assert not info["constraints"]["semantic"]
    state.semantic_score = 0.6
    info = assess_sufficiency(state, cfg)
    assert info["sufficient"]


def test_should_stop_requires_low_gain():
    state = EvidenceState(query="x")
    state.semantic_score = 1.0
    state.reasoning_coverage = 1.0
    state.consistency_score = 1.0
    state.structural_gain = 0.5
    cfg = SufficiencyConfig()
    assert not should_stop(state, cfg)
    state.structural_gain = 0.0
    assert should_stop(state, cfg)


if __name__ == "__main__":
    test_semantic_relevance_empty()
    test_semantic_relevance_matches_self()
    test_structural_information_gain_first_iter()
    test_structural_information_gain_diminishing()
    test_reasoning_completeness()
    test_evidence_consistency_perfect()
    test_evidence_consistency_triple_duplicate_lowers_score()
    test_evidence_consistency_bidirectional_no_conflict()
    test_assess_sufficiency_constraints()
    test_should_stop_requires_low_gain()
    print("All signal tests passed.")
