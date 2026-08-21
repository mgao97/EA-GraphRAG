"""Action implementations for the Evidence Acquisition Controller.

The action space consists of:

    RETRIEVE   – semantic similarity search over node texts.
    EXPAND     – hop expansion around the current frontier.
    BRIDGE     – shortest-path search between two evidence clusters.
    VERIFY     – inspect potential conflicts / duplicates.
    STOP       – terminate acquisition.

Each :class:`ActionExecutor` operates on the graph, the embedder, and the
mutable :class:`EvidenceState`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..data.graph_builder import GraphBuilder
from ..utils.embedding import BaseEmbedder
from ..evidence.state import EvidenceState


class Action(str, Enum):
    RETRIEVE = "RETRIEVE"
    EXPAND = "EXPAND"
    BRIDGE = "BRIDGE"
    VERIFY = "VERIFY"
    STOP = "STOP"


@dataclass
class ActionResult:
    action: Action
    target: List[str] = field(default_factory=list)
    new_nodes: List[str] = field(default_factory=list)
    new_edges: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value if isinstance(self.action, Action) else str(self.action),
            "target": list(self.target),
            "new_nodes": list(self.new_nodes),
            "new_edges": [list(e) for e in self.new_edges],
            "score": float(self.score),
            "metadata": dict(self.metadata),
        }


class ActionExecutor:
    """Executes the controller actions on the graph."""

    def __init__(self, graph: GraphBuilder, embedder: BaseEmbedder,
                  top_k_retrieve: int = 5, expand_hops: int = 2,
                  bridge_max_hops: int = 3,
                  rng: Optional[random.Random] = None):
        self.graph = graph
        self.embedder = embedder
        self.top_k_retrieve = top_k_retrieve
        self.expand_hops = expand_hops
        self.bridge_max_hops = bridge_max_hops
        self.rng = rng or random.Random(0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _all_node_ids(self) -> List[str]:
        return list(self.graph.graph.nodes())

    def _node_text(self, nid: str) -> str:
        if nid in self.graph.node_text:
            return self.graph.node_text[nid]
        return self.graph.graph.nodes[nid].get("entity_name", nid)

    def _semantic_search(self, query: str,
                          candidates: Sequence[str], top_k: int) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        texts = [self._node_text(c) for c in candidates]
        q = self.embedder.embed([query])[0]
        m = self.embedder.embed(texts)
        sims = (m @ q.reshape(-1, 1)).flatten()
        order = np.argsort(-sims)[: max(0, min(top_k, len(candidates)))]
        return [(candidates[i], float(sims[i])) for i in order]

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    def retrieve(self, state: EvidenceState, top_k: Optional[int] = None) -> ActionResult:
        top_k = top_k or self.top_k_retrieve
        candidates = [n for n in self._all_node_ids() if n not in set(state.nodes)]
        scored = self._semantic_search(state.query, candidates, top_k)
        new_nodes: List[str] = []
        new_edges: List[List[str]] = []
        for nid, score in scored:
            if nid not in state.nodes:
                state.add_node(nid, self._node_text(nid))
                new_nodes.append(nid)
                state.token_count += len(self._node_text(nid).split())
        # Add edges to the existing frontier.
        frontier = list(state.nodes)
        for nid in new_nodes:
            for existing in frontier:
                if self.graph.graph.has_edge(existing, nid):
                    state.add_edge(existing, nid)
                    new_edges.append([existing, nid])
        score = float(np.mean([s for _, s in scored])) if scored else 0.0
        return ActionResult(Action.RETRIEVE,
                              target=[n for n, _ in scored],
                              new_nodes=new_nodes,
                              new_edges=new_edges,
                              score=score)

    def expand(self, state: EvidenceState, seeds: Optional[Sequence[str]] = None,
                hops: Optional[int] = None) -> ActionResult:
        hops = hops or self.expand_hops
        seed_nodes = list(seeds) if seeds else list(state.nodes)
        new_nodes: List[str] = []
        new_edges: List[List[str]] = []
        frontier = seed_nodes or state.nodes
        for seed in frontier:
            for nb in self.graph.neighbors(seed, max_hops=hops):
                if nb not in state.nodes:
                    state.add_node(nb, self._node_text(nb))
                    new_nodes.append(nb)
                    state.token_count += len(self._node_text(nb).split())
                if self.graph.graph.has_edge(seed, nb):
                    state.add_edge(seed, nb)
                    new_edges.append([seed, nb])
        return ActionResult(Action.EXPAND,
                              target=list(seeds) if seeds else [],
                              new_nodes=new_nodes,
                              new_edges=new_edges)

    def bridge(self, state: EvidenceState, endpoints: Optional[Sequence[str]] = None,
                cutoff: Optional[int] = None) -> ActionResult:
        """If ``endpoints`` are given, try to bridge between them; otherwise pick
        two random evidence clusters and search for a connecting path."""
        cutoff = cutoff or self.bridge_max_hops
        if len(state.nodes) < 2:
            return ActionResult(Action.BRIDGE)
        if endpoints and len(endpoints) >= 2:
            a, b = endpoints[0], endpoints[-1]
        else:
            nodes = list(state.nodes)
            a, b = self.rng.sample(nodes, 2)
        path = self.graph.shortest_path(a, b, cutoff=cutoff)
        new_nodes: List[str] = []
        new_edges: List[List[str]] = []
        if path:
            for node in path:
                if node not in state.nodes:
                    state.add_node(node, self._node_text(node))
                    new_nodes.append(node)
                    state.token_count += len(self._node_text(node).split())
            for i in range(len(path) - 1):
                state.add_edge(path[i], path[i + 1])
                new_edges.append([path[i], path[i + 1]])
        return ActionResult(Action.BRIDGE,
                              target=[a, b],
                              new_nodes=new_nodes,
                              new_edges=new_edges,
                              metadata={"path": path} if path else {})

    def verify(self, state: EvidenceState) -> ActionResult:
        """No-op action that just inspects consistency.  The actual conflict
        resolution is performed by :func:`evidence_consistency` in the signals
        module; this method returns the diagnostic metadata.
        """
        edge_counts: Dict[Tuple[str, str], int] = {}
        for e in state.edges:
            key = tuple(sorted(e))
            edge_counts[key] = edge_counts.get(key, 0) + 1
        dups = [list(k) for k, v in edge_counts.items() if v > 1]
        return ActionResult(Action.VERIFY,
                              target=[n for k in dups for n in k],
                              metadata={"duplicate_edges": dups})

    # ------------------------------------------------------------------
    # convenience dispatcher
    # ------------------------------------------------------------------
    def run(self, action: Action | str, state: EvidenceState, **kwargs) -> ActionResult:
        if isinstance(action, str):
            action = Action(action)
        if action == Action.RETRIEVE:
            return self.retrieve(state, **kwargs)
        if action == Action.EXPAND:
            return self.expand(state, **kwargs)
        if action == Action.BRIDGE:
            return self.bridge(state, **kwargs)
        if action == Action.VERIFY:
            return self.verify(state)
        if action == Action.STOP:
            return ActionResult(Action.STOP)
        raise ValueError(f"Unknown action: {action}")
