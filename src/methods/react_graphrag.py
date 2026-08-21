"""ReAct + GraphRAG baseline.

The agent interleaves Thought / Action / Observation steps using a textual
policy.  This isolates the *controller* contribution from the *agent*
contribution, as required by ``experiment.md`` §23.

Phase 1 implementation:
* The LLM is given a strict prompt asking for "Action: <NAME>".
* If the LLM does not return a known action we default to ``RETRIEVE`` (the
  conservative choice) so the agent always makes progress.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..controller.actions import Action, ActionExecutor
from ..data.hotpotqa import HotpotQAExample
from ..evidence.state import EvidenceState, Trajectory
from ..llm.base import LLMMessage
from .base import BaseMethod, MethodOutput


_ACTION_RE = re.compile(r"\b(RETRIEVE|EXPAND|BRIDGE|VERIFY|STOP)\b", re.I)


class ReActGraphRAGMethod(BaseMethod):
    name = "react_graphrag"

    def __init__(self, graph, embedder, llm, expand_hops: int = 2,
                  top_k_retrieve: int = 5, max_steps: int = 6, **kwargs):
        super().__init__(graph, embedder, llm)
        self.executor = ActionExecutor(graph, embedder,
                                         top_k_retrieve=top_k_retrieve,
                                         expand_hops=expand_hops)
        self.max_steps = max_steps

    def _decide(self, state: EvidenceState, example: HotpotQAExample) -> Action:
        prompt = (
            "You are a ReAct agent solving multi-hop questions over a knowledge graph.\n"
            "At each step you may choose one action:\n"
            "  RETRIEVE - semantic search for new evidence\n"
            "  EXPAND   - 1-hop expansion around current evidence\n"
            "  BRIDGE   - shortest-path search between two clusters\n"
            "  VERIFY   - consistency check\n"
            "  STOP     - finish acquisition\n\n"
            f"Question: {example.question}\n"
            f"Evidence nodes so far: {len(state.nodes)}\n"
            f"Evidence edges so far: {len(state.edges)}\n\n"
            "Pick the single best action. Respond on one line, for example:\n"
            "Action: RETRIEVE\n"
        )
        msgs = [
            LLMMessage("system", "You are a careful ReAct agent."),
            LLMMessage("user", prompt),
        ]
        out = self.llm.chat(msgs).text
        m = _ACTION_RE.search(out)
        if not m:
            # Conservative default: keep retrieving until something is found.
            return Action.RETRIEVE if not state.nodes else Action.STOP
        return Action(m.group(1).upper())

    def answer(self, example: HotpotQAExample) -> MethodOutput:
        trajectory = Trajectory(qid=example.qid, question=example.question,
                                 gold_answer=example.answer,
                                 method=self.name)
        state = EvidenceState(query=example.question)
        for i in range(self.max_steps):
            state.iteration = i
            action = self._decide(state, example)
            if action == Action.STOP:
                trajectory.add_step(state.as_step(action=Action.STOP.value,
                                                    action_target=[]))
                trajectory.stop_reason = "react_stop"
                break
            result = self.executor.run(action, state)
            trajectory.add_step(state.as_step(action=action.value,
                                                action_target=result.target,
                                                new_nodes=result.new_nodes,
                                                new_edges=result.new_edges,
                                                retrieval_score=result.score))
        else:
            trajectory.stop_reason = "max_steps_reached"
        trajectory.final_evidence = list(state.nodes)
        trajectory.total_cost = {
            "nodes": len(state.nodes),
            "edges": len(state.edges),
            "tokens": state.token_count,
            "retrieval_calls": max(0, len(trajectory.steps) - 1),
            "iterations": len(trajectory.steps),
        }
        pred = self._llm_answer(state, example)
        trajectory.final_answer = pred
        from ..eval.metrics import answer_metrics
        return MethodOutput(qid=example.qid, question=example.question,
                             gold_answer=example.answer,
                             predicted_answer=pred,
                             trajectory=trajectory,
                             metrics=answer_metrics(pred, example.answer))
