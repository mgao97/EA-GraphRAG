"""Shared runner for Phase 1 experiments.

Provides a single entry point that:

1. Loads the dataset and the graph (either in-memory or pre-built).
2. Constructs the requested method.
3. Iterates over examples, writes trajectories, and aggregates metrics.
4. Returns a :class:`pandas`-free summary dict that :mod:`analyze_results`
   can serialise to CSV.

Methods supported:
    fixed_hop_{1,2,3,4}     – fixed-hop baseline
    graphrag                 – vanilla GraphRAG
    react_graphrag           – ReAct + GraphRAG
    ea_graphrag              – EA-GraphRAG (full)
    ea_graphrag_ablated      – EA-GraphRAG with one signal removed
    oracle                   – Oracle controller (uses gold supporting facts)
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, repo_root
from src.controller.eac import AblationConfig
from src.controller.oracle import OracleController
from src.controller.actions import ActionExecutor
from src.data.graph_builder import GraphBuilder, build_graph_from_examples
from src.data.hotpotqa import HotpotQAExample, load_hotpotqa, load_synthetic
from src.eval.metrics import answer_metrics
from src.eval.trajectory import TrajectoryWriter
from src.evidence.sufficiency import SufficiencyConfig
from src.llm.api import APILLM
from src.llm.dummy import DummyLLM
from src.methods.ea_graphrag import EAGraphRAGMethod
from src.methods.bm25 import BM25Method, BM25Index
from src.methods.fixed_hop import FixedHopMethod
from src.methods.graphrag import GraphRAGMethod
from src.methods.react_graphrag import ReActGraphRAGMethod
from src.utils.embedding import get_embedder
from src.utils.io import ensure_dir, read_json, write_json


def _load_examples(cfg) -> List[HotpotQAExample]:
    name = cfg["dataset"]["name"]
    if name == "synthetic":
        return load_synthetic(cfg["dataset"]["synthetic_path"],
                              limit=cfg["dataset"].get("limit"),
                              seed=cfg["project"]["seed"])
    return load_hotpotqa(cfg["dataset"]["path"],
                         limit=cfg["dataset"].get("limit"),
                         seed=cfg["project"]["seed"])


def _load_graph(cfg, examples: List[HotpotQAExample]) -> GraphBuilder:
    builder = build_graph_from_examples(examples,
                                         min_entity_len=cfg["graph"]["min_entity_len"],
                                         max_entities_per_passage=cfg["graph"]["max_entities_per_passage"])
    return builder


def _build_llm(cfg):
    backend = cfg["llm"]["backend"]
    if backend in {"openai", "api", "ollama", "vllm", "lmstudio"}:
        return APILLM(
            model=cfg["llm"].get("model") or "gpt-4o-mini",
            api_key=cfg["llm"].get("api_key"),
            base_url=cfg["llm"].get("base_url"),
            provider=cfg["llm"].get("provider") or backend,
        )
    return DummyLLM(mode=cfg["llm"].get("dummy_mode", "heuristic"))


def _build_method(name: str, cfg: Dict[str, Any], graph: GraphBuilder,
                   embedder, llm, example: Optional[HotpotQAExample] = None,
                   bm25_index: Optional[BM25Index] = None,
                   all_examples=None):
    ctrl = cfg["controller"]
    suff = SufficiencyConfig(
        tau_sem=ctrl["sufficiency"]["tau_sem"],
        tau_reason=ctrl["sufficiency"]["tau_reason"],
        tau_cons=ctrl["sufficiency"]["tau_cons"],
        marginal_gain_threshold=ctrl["sufficiency"]["marginal_gain_threshold"],
    )
    abl = AblationConfig(**ctrl["ablation"])
    if name == "bm25":
        return BM25Method(graph, embedder, llm,
                           index=bm25_index,
                           examples=all_examples,
                           top_k=cfg["retrieval"]["top_k_retrieve"])
    if name.startswith("fixed_hop_"):
        hops = int(name.split("_")[-1])
        return FixedHopMethod(graph, embedder, llm, hops=hops,
                               top_k_retrieve=cfg["retrieval"]["top_k_retrieve"])
    if name == "graphrag":
        return GraphRAGMethod(graph, embedder, llm,
                               top_k_retrieve=cfg["retrieval"]["top_k_retrieve"],
                               expand_hops=cfg["retrieval"]["expand_hops"])
    if name == "react_graphrag":
        return ReActGraphRAGMethod(graph, embedder, llm,
                                     top_k_retrieve=cfg["retrieval"]["top_k_retrieve"],
                                     expand_hops=cfg["retrieval"]["expand_hops"],
                                     max_steps=ctrl["max_iterations"])
    if name == "ea_graphrag":
        gold = set(example.supporting_titles) if example else set()
        return EAGraphRAGMethod(graph, embedder, llm,
                                 top_k_retrieve=cfg["retrieval"]["top_k_retrieve"],
                                 expand_hops=cfg["retrieval"]["expand_hops"],
                                 bridge_max_hops=cfg["retrieval"]["bridge_max_hops"],
                                 max_iterations=ctrl["max_iterations"],
                                 sufficiency=suff,
                                 ablation=abl,
                                 gold_titles=gold,
                                 title_to_node=graph.title_to_node)
    if name.startswith("ea_graphrag_ablated_"):
        # ea_graphrag_ablated_semantic etc.
        removed = name.split("_")[-1]
        abl_dict = dict(abl.__dict__)
        if removed in abl_dict:
            abl_dict[f"use_{removed}"] = False
        gold = set(example.supporting_titles) if example else set()
        return EAGraphRAGMethod(graph, embedder, llm,
                                 top_k_retrieve=cfg["retrieval"]["top_k_retrieve"],
                                 expand_hops=cfg["retrieval"]["expand_hops"],
                                 bridge_max_hops=cfg["retrieval"]["bridge_max_hops"],
                                 max_iterations=ctrl["max_iterations"],
                                 sufficiency=suff,
                                 ablation=AblationConfig(**abl_dict),
                                 gold_titles=gold,
                                 title_to_node=graph.title_to_node)
    if name == "oracle":
        if example is None:
            raise ValueError("oracle requires per-example gold")
        executor = ActionExecutor(graph, embedder,
                                    top_k_retrieve=cfg["retrieval"]["top_k_retrieve"],
                                    expand_hops=cfg["retrieval"]["expand_hops"],
                                    bridge_max_hops=cfg["retrieval"]["bridge_max_hops"])
        return OracleController(executor=executor,
                                 gold_titles=set(example.supporting_titles),
                                 title_to_node=graph.title_to_node,
                                 sufficiency=suff,
                                 max_iterations=ctrl["max_iterations"])
    raise ValueError(f"Unknown method: {name}")


def _run_oracle(controller, example: HotpotQAExample, llm) -> Dict[str, Any]:
    state, traj = controller.run(example.question, example.qid, example.answer)
    # Use the underlying method._format_context helper indirectly via a wrapper.
    # We synthesise a MethodOutput-style record.
    from src.methods.base import MethodOutput
    from src.eval.metrics import answer_metrics
    # Quick LLM call using the same prompt as base method.
    prompt = (
        "Use the following context to answer the question. "
        "If the answer is not contained in the context, say 'unknown'.\n\n"
        f"Context:\n{_format_state_context(state, example)}\n\n"
        f"Question: {example.question}\nAnswer:"
    )
    msgs = [("system", "You are a careful multi-hop QA assistant."),
            ("user", prompt)]
    from src.llm.base import LLMMessage
    msgs = [LLMMessage(m[0], m[1]) for m in msgs]
    resp = llm.chat(msgs)
    pred = resp.text.strip()
    traj.final_answer = pred
    metrics = answer_metrics(pred, example.answer)
    return MethodOutput(qid=example.qid, question=example.question,
                         gold_answer=example.answer,
                         predicted_answer=pred, trajectory=traj,
                         metrics=metrics)


def _format_state_context(state, example, max_sentences: int = 12) -> str:
    titles = set(example.supporting_titles)
    chunks: List[str] = []
    for title, sents in example.context:
        if title in titles:
            for sent in sents[: max_sentences - len(chunks)]:
                chunks.append(f"{title}: {sent}")
                if len(chunks) >= max_sentences:
                    break
        if len(chunks) >= max_sentences:
            break
    if not chunks:
        chunks = [state.texts.get(n, n) for n in state.nodes[:max_sentences]]
    return "\n".join(chunks) or "No context available."


def run_experiment(methods: Sequence[str], cfg_path: str, experiment: str,
                    output_csv: str, trajectory_dir: str,
                    n_questions: Optional[int] = None) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    examples = _load_examples(cfg)
    if n_questions:
        examples = examples[:n_questions]
    print(f"[runner] loaded {len(examples)} examples")
    graph = _load_graph(cfg, examples)
    print(f"[runner] graph: {graph.stats()}")
    embedder = get_embedder(
        cfg["embedding"]["backend"],
        cfg["embedding"].get("model"),
        dim=cfg["embedding"].get("dim", 256),
        base_url=cfg["embedding"].get("base_url"),
        api_key=cfg["embedding"].get("api_key"),
        api_key_env=cfg["embedding"].get("api_key_env", "OPENAI_API_KEY"),
        seed=cfg["project"]["seed"],
    )
    llm = _build_llm(cfg)
    # Build a shared BM25 index once per experiment so all bm25 method
    # calls share it.  We need at least one example to materialise the
    # corpus; if the corpus is empty the BM25 method is simply skipped.
    bm25_index = None
    if any(m == "bm25" for m in methods) and examples:
        try:
            bm25_index = BM25Index.from_examples(examples)
            print(f"[runner] BM25 index: {len(bm25_index.passages)} passages")
        except Exception as exc:
            print(f"[runner] could not build BM25 index: {exc}")
    writer = TrajectoryWriter(trajectory_dir)
    rows: List[Dict[str, Any]] = []
    method_summaries: Dict[str, Dict[str, Any]] = {}
    for method_name in methods:
        per_method_records: List[Dict[str, Any]] = []
        per_method_records_metrics: List[Dict[str, Any]] = []
        t0 = time.time()
        for ex_idx, ex in enumerate(examples):
            try:
                method = _build_method(method_name, cfg, graph, embedder, llm,
                                          example=ex, bm25_index=bm25_index,
                                          all_examples=examples)
            except ValueError as exc:
                print(f"[runner] could not build {method_name}: {exc}")
                continue
            if hasattr(llm, "set_gold"):
                llm.set_gold(ex.answer)
            if method_name == "oracle":
                out = _run_oracle(method, ex, llm)
            else:
                out = method.answer(ex)
            traj_path = writer.write(out.trajectory, experiment=experiment)
            row = {
                "method": method_name,
                "qid": ex.qid,
                "question": ex.question,
                "gold_answer": ex.answer,
                "predicted_answer": out.predicted_answer,
                "em": out.metrics["em"],
                "f1": out.metrics["f1"],
                "type": ex.type,
                "level": ex.level,
                "hop_count": ex.hop_count(),
                "nodes": out.trajectory.total_cost.get("nodes", 0),
                "edges": out.trajectory.total_cost.get("edges", 0),
                "tokens": out.trajectory.total_cost.get("tokens", 0),
                "retrieval_calls": out.trajectory.total_cost.get("retrieval_calls", 0),
                "iterations": out.trajectory.total_cost.get("iterations", 0),
                "stop_reason": out.trajectory.stop_reason,
                "trajectory_path": str(traj_path),
            }
            rows.append(row)
            per_method_records.append(row)
            per_method_records_metrics.append({"em": out.metrics["em"], "f1": out.metrics["f1"]})
        elapsed = time.time() - t0
        # Aggregate.
        em_mean = sum(r["em"] for r in per_method_records) / max(1, len(per_method_records))
        f1_mean = sum(r["f1"] for r in per_method_records) / max(1, len(per_method_records))
        cost = {
            "nodes": sum(r["nodes"] for r in per_method_records),
            "edges": sum(r["edges"] for r in per_method_records),
            "tokens": sum(r["tokens"] for r in per_method_records),
            "retrieval_calls": sum(r["retrieval_calls"] for r in per_method_records),
        }
        method_summaries[method_name] = {
            "em": em_mean,
            "f1": f1_mean,
            "n": len(per_method_records),
            "cost": cost,
            "elapsed_sec": elapsed,
        }
        print(f"[runner] {method_name:>22s} EM={em_mean:.3f} F1={f1_mean:.3f}  "
              f"nodes={cost['nodes']} tokens={cost['tokens']} "
              f"time={elapsed:.1f}s")
    if rows:
        ensure_dir(os.path.dirname(output_csv) or ".")
        keys = list(rows[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=keys)
            writer_csv.writeheader()
            for r in rows:
                writer_csv.writerow(r)
        print(f"[runner] wrote per-query records to {output_csv}")
    summary_path = output_csv.replace(".csv", "_summary.json")
    write_json(summary_path, method_summaries)
    print(f"[runner] wrote method summary to {summary_path}")
    return method_summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--experiment", default="e1")
    parser.add_argument("--output", default="results/e1_overall.csv")
    parser.add_argument("--trajectories", default="results/trajectories")
    parser.add_argument("--n_questions", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.methods, args.config, args.experiment,
                    args.output, args.trajectories,
                    n_questions=args.n_questions)


if __name__ == "__main__":
    main()
