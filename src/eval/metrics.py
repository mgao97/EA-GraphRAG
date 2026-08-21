"""Standard QA metrics (EM and F1) plus trajectory aggregations."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    return _f1(pred, gold)


def answer_metrics(pred: str, gold: str) -> Dict[str, float]:
    return {
        "em": exact_match_score(pred, gold),
        "f1": f1_score(pred, gold),
    }


def aggregate_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Mean EM/F1 over a list of {em, f1} dicts."""
    em_vals: List[float] = []
    f1_vals: List[float] = []
    for r in records:
        em_vals.append(float(r.get("em", 0.0)))
        f1_vals.append(float(r.get("f1", 0.0)))
    return {
        "em": sum(em_vals) / max(1, len(em_vals)),
        "f1": sum(f1_vals) / max(1, len(f1_vals)),
        "n": len(em_vals),
    }
