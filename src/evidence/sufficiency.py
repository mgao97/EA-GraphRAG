"""Sufficiency assessment.

The Phase 1 design follows ``experiment.md`` §6: sufficiency is a *constraint
satisfaction* problem rather than a weighted average, and structural gain is
used for *stopping decisions* rather than as a fourth additive term.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .state import EvidenceState


@dataclass
class SufficiencyConfig:
    tau_sem: float = 0.55
    tau_reason: float = 0.66
    tau_cons: float = 0.85
    marginal_gain_threshold: float = 0.02


def assess_sufficiency(state: EvidenceState, cfg: SufficiencyConfig) -> Dict[str, Any]:
    """Return the boolean sufficiency decision plus the active constraints."""
    constraints = {
        "semantic": state.semantic_score >= cfg.tau_sem,
        "reasoning": state.reasoning_coverage >= cfg.tau_reason,
        "consistency": state.consistency_score >= cfg.tau_cons,
    }
    sufficient = all(constraints.values())
    return {
        "sufficient": sufficient,
        "constraints": constraints,
    }


def should_stop(state: EvidenceState, cfg: SufficiencyConfig) -> bool:
    """Stopping rule.

    The agent stops when *both* of the following hold:

    1.  All sufficiency constraints are satisfied.
    2.  Marginal structural gain has fallen below ``cfg.marginal_gain_threshold``
        (no new information is being added).
    """
    info = assess_sufficiency(state, cfg)
    if not info["sufficient"]:
        return False
    return state.structural_gain < cfg.marginal_gain_threshold
