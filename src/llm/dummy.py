"""A deterministic, dependency-free LLM used for offline experiments.

The dummy LLM implements two modes:

* ``"heuristic"`` (default): extracts an answer span from the context using a
  small set of heuristics.  This lets the entire pipeline run without any
  external service.
* ``"gold"``: returns the gold answer.  Used to validate that the retrieval /
  controller works correctly independent of LLM quality.

Both modes are deterministic and seeded, which is important for
reproducibility.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from ..utils.embedding import BaseEmbedder, DummyEmbedder
from .base import BaseLLM, LLMMessage, LLMResponse, join_messages


_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_FIRST_QUOTED_RE = re.compile(r"\"([^\"]{2,40})\"")

# Relation patterns used by the heuristic extractor.
_RELATION_PATTERNS = [
    # (regex, "subj_first") – when True, the *first* capture group is the
    # subject (the thing being asked about) and the second is the answer.
    # When False, the answer is the first capture group.
    (re.compile(r"([A-Z][A-Za-z0-9' \-]+?)\s+was directed by\s+([A-Z][A-Za-z0-9' \-]+)"), "answer_second"),
    (re.compile(r"([A-Z][A-Za-z0-9' \-]+?)\s+directed\s+([A-Z][A-Za-z0-9' \-]+)"), "answer_first"),
    (re.compile(r"([A-Z][A-Za-z0-9' \-]+?)\s+stars\s+([A-Z][A-Za-z0-9' \-]+)"), "answer_second"),
    (re.compile(r"([A-Z][A-Za-z0-9' \-]+?)\s+was written by\s+([A-Z][A-Za-z0-9' \-]+)"), "answer_second"),
    (re.compile(r"([A-Z][A-Za-z0-9' \-]+?)\s+is from the\s+([A-Z][A-Za-z0-9' \-]+)"), "answer_second"),
    (re.compile(r"([A-Z][A-Za-z0-9' \-]+?)\s+is from\s+([A-Z][A-Za-z0-9' \-]+)"), "answer_second"),
    (re.compile(r"([A-Z][A-Za-z0-9' \-]+?)\s+was born in\s+([A-Z][A-Za-z0-9' \-]+)"), "answer_second"),
]

# Question patterns mapped to the relation regex we should prefer.
_QUESTION_HINTS = [
    # Ordered from most-specific to least-specific so multi-part questions
    # resolve to the *last* matching aspect (see _answer_from_context).
    (re.compile(r"who starred|starring|star in|starred", re.I), 2),
    (re.compile(r"who (wrote|authored)|author of|written by", re.I), 3),
    (re.compile(r"who directed|director of|which director|directed ", re.I), 0),
    # Nationality / country hints.  Pattern index 5 = "X is from Y" (matches
    # synthetic phrasing); index 4 = "X is from the Y".
    (re.compile(r"what country|country is|which country|nationality|from which country|where is .* from", re.I), 5),
]


@dataclass
class DummyLLM(BaseLLM):
    name: str = "dummy"
    mode: str = "heuristic"   # "heuristic" or "gold"
    gold_answer: str = ""
    prefer_quoted: bool = True
    embedder: BaseEmbedder = field(default=None)

    def __post_init__(self):
        if self.embedder is None:
            object.__setattr__(self, "embedder", DummyEmbedder(dim=128, seed=0))

    def chat(self, messages: Sequence[LLMMessage], temperature: float = 0.0,
              max_tokens: int = 512) -> LLMResponse:
        prompt = join_messages(messages)
        answer = self._extract(prompt)
        return LLMResponse(text=answer,
                            usage={"prompt_tokens": len(prompt.split()),
                                   "completion_tokens": len(answer.split())})

    # ------------------------------------------------------------------
    # public extraction
    # ------------------------------------------------------------------
    def _extract(self, prompt: str) -> str:
        if self.mode == "gold" and self.gold_answer:
            return self.gold_answer
        if not prompt:
            return "unknown"
        ctx = self._extract_context(prompt)
        question = self._extract_question(prompt)
        return self._answer_from_context(question or "", ctx)

    # ------------------------------------------------------------------
    # context / question parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_context(prompt: str) -> str:
        m = re.search(r"Context:\s*(.+?)(?:\nQuestion:|\n\nQuestion:|$)", prompt, re.DOTALL)
        return m.group(1) if m else prompt

    @staticmethod
    def _extract_question(prompt: str) -> str:
        m = re.search(r"Question:\s*(.+?)(?:\nAnswer:|$)", prompt, re.DOTALL)
        if not m:
            return ""
        return m.group(1).strip()

    # ------------------------------------------------------------------
    # answer heuristics
    # ------------------------------------------------------------------
    def _answer_from_context(self, question: str, ctx: str) -> str:
        # 1) Choose preferred relation based on the question wording.  We
        # prefer the *last* hint that matches so multi-part questions
        # ("X and what country") prefer the trailing aspect.
        preferred = None
        for pattern, idx in _QUESTION_HINTS:
            if pattern.search(question):  # flags already set in pattern
                preferred = idx
        rel_order = list(range(len(_RELATION_PATTERNS)))
        if preferred is not None:
            rel_order.remove(preferred)
            rel_order.insert(0, preferred)

        # Detect questions that ask for multiple answers (e.g. 'both X and Y').
        wants_all = bool(re.search(r"\bboth\b|\beach\b", question, re.I))
        collected: List[str] = []
        for idx in rel_order:
            regex, kind = _RELATION_PATTERNS[idx]
            matches = list(regex.finditer(ctx))
            # Prefer matches whose subject is mentioned in the question.
            matches.sort(key=lambda m: 0 if self._mentioned(question, m.group(1).strip())
                         else 1)
            for m in matches:
                subj, obj = m.group(1).strip(), m.group(2).strip()
                if kind == "answer_second":
                    candidate, other = obj, subj
                else:
                    candidate, other = subj, obj
                # Skip candidates mentioned only in the question (and not in
                # the context).
                if self._mentioned(question, candidate) and not self._mentioned(ctx, candidate):
                    continue
                if wants_all and not self._mentioned(question, subj):
                    continue
                if not candidate:
                    continue
                if wants_all and candidate in collected:
                    continue
                if wants_all:
                    collected.append(candidate)
                else:
                    return candidate
        if wants_all and collected:
            if len(collected) == 1:
                return collected[0]
            return " and ".join(collected)
        # 2) Year.
        ym = _YEAR_RE.search(ctx)
        if ym:
            return ym.group(1)
        # 3) Quoted.
        if self.prefer_quoted:
            qm = _FIRST_QUOTED_RE.search(ctx)
            if qm:
                return qm.group(1).strip()
        # 4) Embedding-similarity fallback: pick the most relevant sentence and
        #    return its trailing capitalised phrase.
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", ctx) if s.strip()]
        if sentences and question:
            q_emb = self.embedder.embed([question])[0]
            s_emb = self.embedder.embed(sentences)
            sims = (s_emb @ q_emb.reshape(-1, 1)).flatten()
            best = sentences[int(np.argmax(sims))]
            phrase = self._trailing_noun_phrase(best)
            if phrase:
                return phrase
        if sentences:
            return self._trailing_noun_phrase(sentences[0]) or sentences[0].split()[0]
        return "unknown"

    @staticmethod
    def _mentioned(text: str, entity: str) -> bool:
        return bool(entity) and re.search(rf"\b{re.escape(entity)}\b", text) is not None

    @staticmethod
    def _trailing_noun_phrase(sentence: str) -> str:
        cap = re.findall(r"[A-Z][A-Za-z0-9' \-]+", sentence)
        if not cap:
            return ""
        return cap[-1].strip().rstrip(".,;:")

    # ------------------------------------------------------------------
    # gold-answer mode helper
    # ------------------------------------------------------------------
    def set_gold(self, gold: str) -> None:
        self.gold_answer = gold
