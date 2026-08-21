"""Analyse Phase 1 CSV results and produce the tables described in ``experiment.md``.

For each experiment (e1, e3, e5, e7) we:

* Aggregate per-method EM / F1 (and cost) into a summary table.
* For E3 we additionally compute the Pareto frontier.
* For E7 we group results by hop_count.
* For E5 we compute the deltas between the full model and the ablated variants.

The script writes:

    results/e1_overall_summary.csv
    results/e3_efficiency_summary.csv
    results/e5_ablation_summary.csv
    results/e7_complexity_summary.csv

It uses only the Python standard library so it always runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _read_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: r[k] for k in r})
    return rows


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def analyse_e1(csv_path: str) -> List[Dict[str, Any]]:
    rows = _read_csv(csv_path)
    bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        bucket[r["method"]].append(r)
    summary: List[Dict[str, Any]] = []
    for method, recs in bucket.items():
        n = len(recs)
        em = sum(_safe_float(r["em"]) for r in recs) / max(1, n)
        f1 = sum(_safe_float(r["f1"]) for r in recs) / max(1, n)
        nodes = sum(_safe_float(r["nodes"]) for r in recs) / max(1, n)
        edges = sum(_safe_float(r["edges"]) for r in recs) / max(1, n)
        tokens = sum(_safe_float(r["tokens"]) for r in recs) / max(1, n)
        calls = sum(_safe_float(r["retrieval_calls"]) for r in recs) / max(1, n)
        iters = sum(_safe_float(r["iterations"]) for r in recs) / max(1, n)
        summary.append({
            "method": method,
            "n": n,
            "em": round(em, 4),
            "f1": round(f1, 4),
            "avg_nodes": round(nodes, 2),
            "avg_edges": round(edges, 2),
            "avg_tokens": round(tokens, 2),
            "avg_retrieval_calls": round(calls, 2),
            "avg_iterations": round(iters, 2),
        })
    summary.sort(key=lambda r: -r["f1"])
    return summary


def analyse_e3(csv_path: str) -> List[Dict[str, Any]]:
    """Efficiency = F1 vs Evidence Cost.  Pareto frontier = methods not strictly
    dominated in (F1 ↑, cost ↓)."""
    summary = analyse_e1(csv_path)
    # Pareto: maximise f1, minimise avg_tokens.
    dominated = set()
    for i, a in enumerate(summary):
        for j, b in enumerate(summary):
            if i == j:
                continue
            if (b["f1"] >= a["f1"] and b["avg_tokens"] <= a["avg_tokens"] and
                    (b["f1"] > a["f1"] or b["avg_tokens"] < a["avg_tokens"])):
                dominated.add(i)
    for idx, row in enumerate(summary):
        row["on_pareto_frontier"] = idx not in dominated
    return summary


def analyse_e5(csv_path: str) -> List[Dict[str, Any]]:
    """Ablation: report F1 deltas relative to the full model."""
    summary = analyse_e1(csv_path)
    full_f1 = None
    for row in summary:
        if row["method"] == "ea_graphrag":
            full_f1 = row["f1"]
            break
    if full_f1 is None:
        full_f1 = max(r["f1"] for r in summary) if summary else 0.0
    for row in summary:
        row["delta_f1_vs_full"] = round(row["f1"] - full_f1, 4) if full_f1 else 0.0
        row["cost_increase"] = ""
    return summary


def analyse_e7(csv_path: str) -> Dict[str, List[Dict[str, Any]]]:
    rows = _read_csv(csv_path)
    # Group by method and hop bucket.
    hop_buckets = {"1-hop": [], "2-hop": [], "3-hop": [], "4-hop+": []}

    def bucket(h: int) -> str:
        if h <= 1:
            return "1-hop"
        if h == 2:
            return "2-hop"
        if h == 3:
            return "3-hop"
        return "4-hop+"

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        h = int(_safe_float(r.get("hop_count", 1)))
        grouped[r["method"]][bucket(h)].append(r)
    table: List[Dict[str, Any]] = []
    for method, hops in grouped.items():
        for hb, recs in hops.items():
            n = len(recs)
            if n == 0:
                continue
            table.append({
                "method": method,
                "hop_bucket": hb,
                "n": n,
                "em": round(sum(_safe_float(r["em"]) for r in recs) / n, 4),
                "f1": round(sum(_safe_float(r["f1"]) for r in recs) / n, 4),
                "avg_tokens": round(sum(_safe_float(r["tokens"]) for r in recs) / n, 2),
                "avg_iterations": round(sum(_safe_float(r["iterations"]) for r in recs) / n, 2),
            })
    return {"by_hop": table}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    base = Path(args.results)
    if (base / "e1_overall.csv").exists():
        s1 = analyse_e1(str(base / "e1_overall.csv"))
        _write_csv(str(base / "e1_overall_summary.csv"), s1)
        print(f"E1 summary -> {base/'e1_overall_summary.csv'}")
        for r in s1:
            print(f"  {r['method']:>22s}  F1={r['f1']:.3f}  EM={r['em']:.3f}")
    if (base / "e3_efficiency.csv").exists():
        s3 = analyse_e3(str(base / "e3_efficiency.csv"))
        _write_csv(str(base / "e3_efficiency_summary.csv"), s3)
        print(f"E3 summary -> {base/'e3_efficiency_summary.csv'}")
        for r in s3:
            print(f"  {r['method']:>22s}  F1={r['f1']:.3f}  tokens={r['avg_tokens']:.1f} "
                  f"pareto={r['on_pareto_frontier']}")
    if (base / "e5_ablation.csv").exists():
        s5 = analyse_e5(str(base / "e5_ablation.csv"))
        _write_csv(str(base / "e5_ablation_summary.csv"), s5)
        print(f"E5 summary -> {base/'e5_ablation_summary.csv'}")
        for r in s5:
            print(f"  {r['method']:>36s}  F1={r['f1']:.3f}  ΔF1={r['delta_f1_vs_full']:+.3f}")
    if (base / "e7_complexity.csv").exists():
        s7 = analyse_e7(str(base / "e7_complexity.csv"))
        _write_csv(str(base / "e7_complexity_summary.csv"), s7["by_hop"])
        print(f"E7 summary -> {base/'e7_complexity_summary.csv'}")
        for r in s7["by_hop"]:
            print(f"  {r['method']:>22s}  {r['hop_bucket']:>7s}  F1={r['f1']:.3f}  "
                  f"tokens={r['avg_tokens']:.1f}")


if __name__ == "__main__":
    main()
