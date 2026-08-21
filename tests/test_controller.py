"""Integration tests for the controller, actions and methods."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controller.actions import Action, ActionExecutor
from src.controller.eac import EvidenceAcquisitionController, AblationConfig
from src.controller.oracle import OracleController
from src.data.graph_builder import GraphBuilder, build_graph_from_examples
from src.data.hotpotqa import HotpotQAExample
from src.evidence.sufficiency import SufficiencyConfig
from src.llm.dummy import DummyLLM
from src.methods.ea_graphrag import EAGraphRAGMethod
from src.methods.fixed_hop import FixedHopMethod
from src.methods.graphrag import GraphRAGMethod
from src.utils.embedding import DummyEmbedder


def _toy_graph() -> GraphBuilder:
    example = HotpotQAExample(
        qid="toy",
        question="Who directed Inception and what is their nationality?",
        answer="Christopher Nolan and United Kingdom",
        supporting_facts=[
            {"title": "Article about Inception", "sent_id": 1},
            {"title": "Biography of Christopher Nolan", "sent_id": 1},
        ],
        context=[
            ["Article about Inception",
             ["Inception is a science fiction film.",
              "Inception was directed by Christopher Nolan.",
              "The film stars Leonardo DiCaprio."]],
            ["Biography of Christopher Nolan",
             ["Christopher Nolan is a renowned filmmaker.",
              "Christopher Nolan is from the United Kingdom.",
              "He studied at University College London."]],
        ],
        type="bridge", level="medium",
    )
    return build_graph_from_examples([example]), example


def test_action_executor_retrieve_and_expand():
    builder, example = _toy_graph()
    embedder = DummyEmbedder(dim=32, seed=1)
    executor = ActionExecutor(builder, embedder, top_k_retrieve=2, expand_hops=1)
    state = __import__("src").evidence.state.EvidenceState(query=example.question)
    r = executor.run(Action.RETRIEVE, state)
    assert r.action == Action.RETRIEVE
    assert state.nodes, "RETRIEVE must add nodes"
    before = len(state.nodes)
    r = executor.run(Action.EXPAND, state)
    assert r.action == Action.EXPAND
    assert len(state.nodes) >= before


def test_eac_full_loop_runs():
    builder, example = _toy_graph()
    embedder = DummyEmbedder(dim=32, seed=2)
    llm = DummyLLM()
    executor = ActionExecutor(builder, embedder, top_k_retrieve=2, expand_hops=1)
    controller = EvidenceAcquisitionController(
        embedder=embedder, executor=executor,
        gold_titles=set(example.supporting_titles),
        title_to_node=builder.title_to_node,
        max_iterations=4,
    )
    state, trajectory = controller.run(example.question, example.qid,
                                        example.answer)
    assert len(trajectory.steps) >= 1
    assert trajectory.final_evidence == state.nodes
    assert trajectory.stop_reason != ""


def test_eac_ablation_still_runs():
    builder, example = _toy_graph()
    embedder = DummyEmbedder(dim=32, seed=3)
    llm = DummyLLM()
    method = EAGraphRAGMethod(builder, embedder, llm,
                                gold_titles=set(example.supporting_titles),
                                title_to_node=builder.title_to_node,
                                ablation=AblationConfig(use_structural=False,
                                                          use_consistency=False))
    out = method.answer(example)
    assert out.metrics["f1"] >= 0.0


def test_oracle_uses_gold_titles():
    builder, example = _toy_graph()
    embedder = DummyEmbedder(dim=32, seed=4)
    executor = ActionExecutor(builder, embedder, top_k_retrieve=2, expand_hops=1)
    oracle = OracleController(executor=executor,
                               gold_titles=set(example.supporting_titles),
                               title_to_node=builder.title_to_node,
                               max_iterations=3)
    state, traj = oracle.run(example.question, example.qid, example.answer)
    assert traj.stop_reason in {"oracle_stop", "max_iterations"}


def test_fixed_hop_method_runs():
    builder, example = _toy_graph()
    method = FixedHopMethod(builder, DummyEmbedder(dim=32), DummyLLM(), hops=1)
    out = method.answer(example)
    assert out.metrics["f1"] >= 0.0


def test_bm25_method_runs():
    builder, example = _toy_graph()
    from src.methods.bm25 import BM25Index, BM25Method
    idx = BM25Index.from_examples([example])
    method = BM25Method(builder, DummyEmbedder(dim=32), DummyLLM(), index=idx)
    out = method.answer(example)
    assert out.metrics["f1"] >= 0.0


def test_graphrag_method_runs():
    builder, example = _toy_graph()
    method = GraphRAGMethod(builder, DummyEmbedder(dim=32), DummyLLM())
    out = method.answer(example)
    assert out.metrics["f1"] >= 0.0


if __name__ == "__main__":
    test_action_executor_retrieve_and_expand()
    test_eac_full_loop_runs()
    test_eac_ablation_still_runs()
    test_oracle_uses_gold_titles()
    test_fixed_hop_method_runs()
    test_bm25_method_runs()
    test_graphrag_method_runs()
    print("All controller tests passed.")
