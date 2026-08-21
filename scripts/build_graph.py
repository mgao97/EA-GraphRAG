"""Build the unified knowledge graph used by all baselines.

Reads the dataset (HotpotQA or synthetic) and constructs the entity graph
following the protocol in ``experiment.md`` §3.  The graph is serialised as
JSON so downstream scripts do not have to re-parse the original dataset.

Usage:
    python scripts/build_graph.py --dataset data/sample_hotpotqa.json \
        --output data/graph.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.graph_builder import build_graph_from_examples
from src.data.hotpotqa import load_hotpotqa, load_synthetic
from src.utils.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HotpotQA or synthetic JSON")
    parser.add_argument("--output", default="data/graph.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic or "sample_hotpotqa" in args.dataset:
        examples = load_synthetic(args.dataset, limit=args.limit, seed=args.seed)
    else:
        examples = load_hotpotqa(args.dataset, limit=args.limit, seed=args.seed)
    print(f"Loaded {len(examples)} examples")
    builder = build_graph_from_examples(examples)
    stats = builder.stats()
    print(f"Built graph: {stats}")
    serialised = {
        "nodes": [
            {"id": nid,
             "entity_name": builder.graph.nodes[nid].get("entity_name", ""),
             "text": builder.graph.nodes[nid].get("text", ""),
             "source_documents": sorted(builder.node_source.get(nid, []))}
            for nid in builder.graph.nodes()
        ],
        "edges": [
            {"source": u, "target": v,
             "relation": builder.graph[u][v].get("relation", "co_occurs"),
             "source_documents": sorted(builder.edge_source.get((u, v), []))}
            for u, v in builder.graph.edges()
        ],
        "title_to_node": builder.title_to_node,
        "stats": stats,
    }
    write_json(args.output, serialised)
    print(f"Wrote graph -> {args.output}")


if __name__ == "__main__":
    main()
