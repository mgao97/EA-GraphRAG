"""BM25 baseline.

Pure lexical retrieval baseline — used as a sanity check that the
*controller* in EA-GraphRAG is what helps, not the underlying retrieval
quality alone.

The BM25 index is built over the dataset passages (not the knowledge graph
nodes).  Top-k passages are concatenated into the LLM context for answer
generation, so the answer-extraction path is identical to the other
methods.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.graph_builder import GraphBuilder
from ..data.hotpotqa import HotpotQAExample
from ..evidence.state import EvidenceState, EvidenceStep, Trajectory
from ..llm.base import BaseLLM
from .base import BaseMethod, MethodOutput


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class BM25Index:
    """Wraps a ``rank_bm25.BM25Okapi`` index and the underlying corpus."""

    bm25: Any
    passages: List[Dict[str, str]]  # list of {title, text}
    tokenized_corpus: List[List[str]]

    @classmethod
    def from_examples(cls, examples: Sequence[HotpotQAExample]) -> "BM25Index":
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "rank_bm25 is required for the BM25 baseline. "
                "Install it with `pip install rank_bm25`."
            ) from exc
        passages: List[Dict[str, str]] = []
        seen: set = set()
        for ex in examples:
            for title, sents in ex.context:
                text = " ".join(sents)
                key = (title, text)
                if key in seen:
                    continue
                seen.add(key)
                passages.append({"title": title, "text": text})
        tokenized = [_tokenize(p["title"] + " " + p["text"]) for p in passages]
        bm25 = BM25Okapi(tokenized)
        return cls(bm25=bm25, passages=passages, tokenized_corpus=tokenized)

    def topk(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        q_tokens = _tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        if not scores.size:
            return []
        k = max(0, min(k, len(scores)))
        order = scores.argsort()[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]


class BM25Method(BaseMethod):
    """BM25 baseline as specified in ``experiment.md`` §9.

    The method does *not* use the graph — it retrieves passages directly
    from the dataset corpus.  This isolates the contribution of the
    graph-based controller.
    """

    name = "bm25"

    def __init__(self, graph: GraphBuilder, embedder, llm: BaseLLM,
                  index: Optional[BM25Index] = None,
                  top_k: int = 5,
                  examples: Optional[Sequence[HotpotQAExample]] = None,
                  **kwargs):
        super().__init__(graph, embedder, llm)
        self.top_k = top_k
        # Reuse a shared index across all examples when given one (more
        # efficient for batched runs).
        self.index = index
        if self.index is None:
            if examples is None:
                raise ValueError(
                    "BM25Method needs either `index` or `examples` at construction time"
                )
            self.index = BM25Index.from_examples(examples)

    # ------------------------------------------------------------------
    def _format_passages(self, ids: List[int]) -> str:
        return "\n".join(
            f"{self.index.passages[i]['title']}: {self.index.passages[i]['text']}"
            for i in ids
        )

    def answer(self, example: HotpotQAExample) -> MethodOutput:
        trajectory = Trajectory(qid=example.qid, question=example.question,
                                 gold_answer=example.answer,
                                 method=self.name)
        state = EvidenceState(query=example.question)

        top = self.index.topk(example.question, k=self.top_k)
        retrieved_titles: List[str] = []
        token_count = 0
        for i, score in top:
            p = self.index.passages[i]
            retrieved_titles.append(p["title"])
            token_count += len(_tokenize(p["text"]))
            state.add_node(f"passage::{i}", f"{p['title']}: {p['text']}")
        state.token_count = token_count

        # Record the single RETRIEVE step (no controller / no expand / no bridge).
        step = EvidenceStep(
            step=0,
            evidence_nodes=list(state.nodes),
            evidence_edges=[],
            semantic_score=0.0,
            structural_gain=0.0,
            structural_entropy=0.0,
            reasoning_coverage=0.0,
            consistency_score=1.0,
            sufficiency_score=0.0,
            sufficient=False,
            action="RETRIEVE",
            action_target=retrieved_titles,
            newly_retrieved_nodes=list(state.nodes),
            newly_retrieved_edges=[],
            retrieval_score=float(top[0][1]) if top else 0.0,
            token_count=state.token_count,
        )
        trajectory.add_step(step)
        trajectory.add_step(EvidenceStep(
            step=1, evidence_nodes=list(state.nodes), evidence_edges=[],
            action="STOP", action_target=[],
        ))
        trajectory.final_evidence = list(state.nodes)
        trajectory.total_cost = {
            "nodes": len(state.nodes),
            "edges": 0,
            "tokens": state.token_count,
            "retrieval_calls": 1,
            "iterations": 2,
        }
        trajectory.stop_reason = "bm25_topk_done"

        # Build the LLM context directly from the retrieved passages (we
        # don't have graph nodes here).
        context = self._format_passages([i for i, _ in top])
        from ..llm.base import LLMMessage
        prompt = (
            "Use the following context to answer the question. "
            "If the answer is not contained in the context, say 'unknown'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {example.question}\nAnswer:"
        )
        msgs = [
            LLMMessage("system", "You are a careful multi-hop QA assistant."),
            LLMMessage("user", prompt),
        ]
        pred = self.llm.chat(msgs).text.strip()
        trajectory.final_answer = pred

        from ..eval.metrics import answer_metrics
        return MethodOutput(qid=example.qid, question=example.question,
                             gold_answer=example.answer,
                             predicted_answer=pred,
                             trajectory=trajectory,
                             metrics=answer_metrics(pred, example.answer))
