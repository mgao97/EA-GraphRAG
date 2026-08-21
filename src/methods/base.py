"""Base retriever interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..data.graph_builder import GraphBuilder
from ..data.hotpotqa import HotpotQAExample
from ..evidence.state import EvidenceState, Trajectory
from ..llm.base import BaseLLM
from ..utils.embedding import BaseEmbedder


@dataclass
class MethodOutput:
    qid: str
    question: str
    gold_answer: str
    predicted_answer: str
    trajectory: Trajectory
    metrics: Dict[str, float] = field(default_factory=dict)


class BaseMethod(ABC):
    name: str = "base"

    def __init__(self, graph: GraphBuilder, embedder: BaseEmbedder,
                  llm: BaseLLM, use_gold_context: bool = False, **kwargs):
        """Parameters
        ----------
        use_gold_context:
            When ``True`` the method answers from the gold supporting
            passages (oracle upper-bound).  When ``False`` (default) the
            method answers from its *retrieved* evidence state, which is the
            realistic GraphRAG evaluation setting.
        """
        self.graph = graph
        self.embedder = embedder
        self.llm = llm
        self.use_gold_context = use_gold_context

    @abstractmethod
    def answer(self, example: HotpotQAExample) -> MethodOutput: ...

    # ------------------------------------------------------------------
    def _format_context(self, state: EvidenceState, example: HotpotQAExample,
                          max_sentences: int = 12) -> str:
        # 1) Use the evidence state texts (what the method actually retrieved).
        chunks: List[str] = []
        for n in state.nodes:
            txt = state.texts.get(n) or self.graph.node_text.get(n) or n
            if txt:
                chunks.append(txt)
            if len(chunks) >= max_sentences:
                break
        if chunks:
            return "\n".join(chunks)
        # 2) Fallback: gold supporting passages (oracle mode).
        if self.use_gold_context:
            titles = set(example.supporting_titles)
            for title, sents in example.context:
                if title in titles:
                    for sent in sents[: max_sentences - len(chunks)]:
                        chunks.append(f"{title}: {sent}")
                        if len(chunks) >= max_sentences:
                            break
                if len(chunks) >= max_sentences:
                    break
        if not chunks:
            for title, sents in example.context:
                for sent in sents:
                    chunks.append(f"{title}: {sent}")
                    if len(chunks) >= max_sentences:
                        break
                if len(chunks) >= max_sentences:
                    break
        return "\n".join(chunks) or "No context available."

    def _llm_answer(self, state: EvidenceState, example: HotpotQAExample) -> str:
        from ..llm.base import LLMMessage
        prompt_user = (
            "Use the following context to answer the question. "
            "If the answer is not contained in the context, say 'unknown'.\n\n"
            f"Context:\n{self._format_context(state, example)}\n\n"
            f"Question: {example.question}\nAnswer:"
        )
        msgs = [
            LLMMessage("system", "You are a careful multi-hop QA assistant."),
            LLMMessage("user", prompt_user),
        ]
        resp = self.llm.chat(msgs)
        return resp.text.strip()
