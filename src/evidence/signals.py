"""Four core evidence signals.

All four signals return floats in ``[0, 1]`` so they can be combined by the
sufficiency assessor.  Each signal has an "oracle" mode (used during controlled
experiments) where the gold supporting facts are compared against the current
evidence, and a "model" mode used in real experiments.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List, Sequence, Set

import numpy as np

from ..utils.embedding import BaseEmbedder
from .state import EvidenceState


# ---------------------------------------------------------------------------
# 5.1  Semantic relevance
# ---------------------------------------------------------------------------
def semantic_relevance(state: EvidenceState, query_embedding: np.ndarray,
                        node_embeddings: np.ndarray,
                        top_k: int = 5) -> float:
    """Mean of top-k cosine similarities between the query and the evidence
    nodes.  Empty evidence returns ``0``.
    """
    if node_embeddings.size == 0 or query_embedding.size == 0:
        return 0.0
    if node_embeddings.ndim == 1:
        node_embeddings = node_embeddings.reshape(1, -1)
    sims = (node_embeddings @ query_embedding.reshape(-1, 1)).flatten()
    if sims.size == 0:
        return 0.0
    k = min(top_k, sims.size)
    top = np.partition(sims, -k)[-k:]
    val = float(np.clip(np.mean(top), 0.0, 1.0))
    return val


def semantic_relevance_from_texts(embedder: BaseEmbedder,
                                   query: str,
                                   texts: Sequence[str],
                                   top_k: int = 5) -> float:
    """Convenience wrapper that re-embeds the texts on the fly."""
    if not texts:
        return 0.0
    q = embedder.embed([query])
    nt = embedder.embed(list(texts))
    return semantic_relevance(EvidenceState(query=query), q[0], nt, top_k=top_k)


# ---------------------------------------------------------------------------
# 5.2  Structural information gain
# ---------------------------------------------------------------------------
def _degree_entropy(degrees: Iterable[int]) -> float:
    deg = [d for d in degrees if d > 0]
    if not deg:
        return 0.0
    counts = Counter(deg)
    total = sum(counts.values())
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log(p + 1e-12)
    # Normalise by the maximum entropy of the support size (uniform)
    max_H = math.log(len(counts) + 1e-12)
    return H / max_H if max_H > 0 else 0.0


def structural_entropy(state: EvidenceState, degree_sequence: Sequence[int]) -> float:
    """Shannon entropy of the degree distribution of the current evidence
    subgraph, normalised to ``[0, 1]``."""
    return _degree_entropy(degree_sequence)


def structural_information_gain(state: EvidenceState,
                                 prev_entropy: float,
                                 new_entropy: float,
                                 new_nodes: int,
                                 new_edges: int) -> float:
    """Information gain = how much the structure changed.

    Combines normalised entropy drop with the marginal contribution of newly
    added nodes/edges.  Returns a value in ``[0, 1]`` where ``1`` means a large
    structural change and ``0`` means no change.
    """
    if state.iteration == 0:
        return 1.0
    if prev_entropy == 0 and new_entropy == 0:
        change = 0.0
    else:
        denom = max(prev_entropy, new_entropy, 1e-9)
        change = abs(prev_entropy - new_entropy) / denom
    newness = 0.5 * (1.0 - math.exp(-new_nodes)) + 0.5 * (1.0 - math.exp(-new_edges))
    return float(np.clip(0.6 * change + 0.4 * newness, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 5.3  Reasoning completeness
# ---------------------------------------------------------------------------
def reasoning_completeness(covered: Set[str], gold: Set[str]) -> float:
    """Fraction of gold supporting-fact titles (or sentence ids) that are
    covered by the current evidence.

    In production we use a *predicted* reasoning completion estimator; here we
    provide both oracle and approximate variants.  Both fall in ``[0, 1]``.
    """
    if not gold:
        return 1.0
    if not covered:
        return 0.0
    return len(covered & gold) / len(gold)


def reasoning_completeness_approx(state: EvidenceState,
                                   gold_titles: Set[str],
                                   title_to_node: dict) -> float:
    """Approximate coverage using the title -> primary entity mapping built
    during graph construction.  When no gold data is available this returns
    the proportion of evidence nodes that contain at least one gold keyword.
    """
    if not gold_titles:
        return 1.0
    covered = 0
    for title in gold_titles:
        node = title_to_node.get(title)
        if node and node in state.nodes:
            covered += 1
    return covered / len(gold_titles)


# ---------------------------------------------------------------------------
# 5.4  Evidence consistency
# ---------------------------------------------------------------------------
def evidence_consistency(state: EvidenceState,
                          pair_conflicts: Sequence[Sequence[str]]) -> float:
    """``1 - conflict_ratio``.  Conflicts are pairs of nodes/relations that
    contradict each other.  For Phase 1 we approximate conflict ratio using a
    crude duplicate-relation heuristic.
    """
    if not state.edges and not pair_conflicts:
        return 1.0
    # Collapse (A->B) and (B->A) into the same undirected relation so
    # bidirectional co-occurrence edges don't artificially lower consistency.
    edge_pairs = Counter(tuple(sorted(e)) for e in state.edges)
    # Treat a pair of bidirectional edges as a single logical edge (count=1).
    # A true duplicate requires the same undirected edge to appear 3+ times.
    true_dups = sum(1 for c in edge_pairs.values() if c > 2)
    unique_edges = len(edge_pairs)
    total = max(unique_edges, 1)
    conflict_ratio = min(1.0, (true_dups + len(pair_conflicts)) / total)
    return float(np.clip(1.0 - conflict_ratio, 0.0, 1.0))
