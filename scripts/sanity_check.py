"""Pre-flight checks for the EA-GraphRAG pipeline.

Runs without any LLM / GPU.  Useful as a smoke test before launching the
expensive experiments.

Checks performed:
  1. Imports for every module.
  2. Config file parses.
  3. Synthetic dataset is generated and graph is built.
  4. Unit-test assertions on the four evidence signals.
  5. A single end-to-end ``EAGraphRAGMethod.answer`` call against one
     synthetic example (uses the dummy LLM/embedder so it requires no
     external services).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    print("=== EA-GraphRAG sanity check ===")
    failed = 0

    # 1. Imports
    print("[1/5] importing modules ...", end=" ")
    try:
        import numpy  # noqa
        import yaml  # noqa
        from src.config import load_config
        from src.data.graph_builder import build_graph_from_examples
        from src.data.hotpotqa import load_synthetic, HotpotQAExample
        from src.evidence.signals import (semantic_relevance,
                                           structural_information_gain,
                                           reasoning_completeness,
                                           evidence_consistency)
        from src.evidence.sufficiency import assess_sufficiency
        from src.evidence.state import EvidenceState
        from src.controller.actions import Action, ActionExecutor
        from src.controller.eac import EvidenceAcquisitionController
        from src.controller.oracle import OracleController
        from src.methods.ea_graphrag import EAGraphRAGMethod
        from src.methods.fixed_hop import FixedHopMethod
        from src.methods.graphrag import GraphRAGMethod
        from src.methods.react_graphrag import ReActGraphRAGMethod
        from src.llm.dummy import DummyLLM
        from src.llm.api import APILLM, ollama_llm, vllm_llm, openai_llm
        from src.utils.embedding import DummyEmbedder, get_embedder
        print("OK")
    except Exception as exc:
        print(f"FAIL ({exc})")
        failed += 1

    # 2. Config
    print("[2/5] loading configs/default.yaml ...", end=" ")
    try:
        cfg = load_config()
        assert cfg["dataset"]["name"] in {"hotpotqa", "synthetic"}
        assert cfg["controller"]["max_iterations"] > 0
        print("OK")
    except Exception as exc:
        print(f"FAIL ({exc})")
        failed += 1

    # 3. Data + graph
    print("[3/5] building toy graph ...", end=" ")
    try:
        from scripts.build_sample_data import build
        examples_raw = build(n=6, seed=42)
        examples = [HotpotQAExample(qid=str(r["_id"]), **{
            k: r[k] for k in r if k != "_id"
        }) for r in examples_raw]
        graph = build_graph_from_examples(examples)
        assert graph.graph.number_of_nodes() > 0
        print(f"OK ({graph.graph.number_of_nodes()} nodes)")
    except Exception as exc:
        print(f"FAIL ({exc})")
        failed += 1

    # 4. Signal assertions
    print("[4/5] signal assertions ...", end=" ")
    try:
        from src.evidence.sufficiency import SufficiencyConfig
        suff_cfg = SufficiencyConfig(**cfg["controller"]["sufficiency"])
        s = EvidenceState(query="x")
        s.semantic_score = 0.9
        s.reasoning_coverage = 0.9
        s.consistency_score = 0.9
        info = assess_sufficiency(s, suff_cfg)
        assert info["sufficient"]
        assert reasoning_completeness({"a"}, {"a", "b"}) == 0.5
        gain = structural_information_gain(s, 0.0, 0.0, 1, 1)
        assert 0.0 <= gain <= 1.0
        print("OK")
    except Exception as exc:
        print(f"FAIL ({exc})")
        failed += 1

    # 5. End-to-end on one example
    print("[5/5] end-to-end EA-GraphRAG.answer() ...", end=" ")
    try:
        ex = examples[0]
        llm = DummyLLM()
        emb = DummyEmbedder(dim=64, seed=1)
        method = EAGraphRAGMethod(graph, emb, llm,
                                   gold_titles=set(ex.supporting_titles),
                                   title_to_node=graph.title_to_node,
                                   max_iterations=4)
        out = method.answer(ex)
        assert out.metrics["f1"] >= 0.0
        assert out.trajectory.stop_reason != ""
        print(f"OK (F1={out.metrics['f1']:.2f}, stop='{out.trajectory.stop_reason}', "
              f"nodes={len(out.trajectory.final_evidence)})")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"FAIL ({exc})")
        failed += 1

    print()
    if failed == 0:
        print("All checks passed.  You can now run `./run.sh`.")
        return 0
    print(f"{failed} check(s) failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
