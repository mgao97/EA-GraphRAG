"""Generate matplotlib PDF figures from the Phase 1 results.

If matplotlib is not available, the script prints a clear message and exits
gracefully so the rest of the pipeline still works.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _fig_path(name: str) -> str:
    return str(ROOT / "results" / "figures" / name)


def make_figures(results_dir: str = "results"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure generation. "
              "Install with `pip install matplotlib` to enable.")
        return

    base = Path(results_dir)
    out = base / "figures"
    out.mkdir(parents=True, exist_ok=True)
    # E1 overall bar chart.
    p = base / "e1_overall.csv"
    if p.exists():
        rows = _read_csv(str(p))
        methods: Dict[str, List[float]] = {}
        for r in rows:
            methods.setdefault(r["method"], []).append(_safe_float(r["f1"]))
        names = list(methods.keys())
        f1s = [sum(methods[n]) / max(1, len(methods[n])) for n in names]
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(names)), f1s)
        plt.xticks(range(len(names)), names, rotation=30, ha="right")
        plt.ylabel("F1")
        plt.title("E1: Overall QA performance")
        plt.tight_layout()
        plt.savefig(_fig_path("overall_performance.pdf"))
        plt.close()
        print("  -> overall_performance.pdf")
    # E3 cost vs F1 scatter.
    p = base / "e3_efficiency.csv"
    if p.exists():
        rows = _read_csv(str(p))
        methods = {}
        for r in rows:
            d = methods.setdefault(r["method"], {"f1": [], "tokens": []})
            d["f1"].append(_safe_float(r["f1"]))
            d["tokens"].append(_safe_float(r["tokens"]))
        plt.figure(figsize=(6, 5))
        for name, d in methods.items():
            f1m = sum(d["f1"]) / max(1, len(d["f1"]))
            tm = sum(d["tokens"]) / max(1, len(d["tokens"]))
            plt.scatter(tm, f1m, s=80)
            plt.annotate(name, (tm, f1m), fontsize=8, xytext=(5, 5),
                         textcoords="offset points")
        plt.xlabel("Average tokens (evidence cost)")
        plt.ylabel("F1")
        plt.title("E3: Accuracy vs Evidence Cost")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(_fig_path("accuracy_vs_cost.pdf"))
        plt.close()
        print("  -> accuracy_vs_cost.pdf")
    # E5 ablation bar chart.
    p = base / "e5_ablation_summary.csv"
    if p.exists():
        rows = _read_csv(str(p))
        rows.sort(key=lambda r: r.get("method", ""))
        names = [r["method"] for r in rows]
        f1s = [_safe_float(r["f1"]) for r in rows]
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(names)), f1s)
        plt.xticks(range(len(names)), names, rotation=30, ha="right")
        plt.ylabel("F1")
        plt.title("E5: Ablation")
        plt.tight_layout()
        plt.savefig(_fig_path("ablation.pdf"))
        plt.close()
        print("  -> ablation.pdf")
    # E7 complexity plot.
    p = base / "e7_complexity_summary.csv"
    if not p.exists():
        p = base / "e7_complexity.csv"
    if p.exists():
        rows = _read_csv(str(p))
        bucket_order = {"1-hop": 1, "2-hop": 2, "3-hop": 3, "4-hop+": 4}
        methods = {}
        for r in rows:
            d = methods.setdefault(r["method"], {})
            d.setdefault(r["hop_bucket"], []).append(_safe_float(r["f1"]))
        plt.figure(figsize=(7, 5))
        for name, d in methods.items():
            xs, ys = [], []
            for hb, vals in d.items():
                xs.append(bucket_order.get(hb, 0))
                ys.append(sum(vals) / max(1, len(vals)))
            xs, ys = zip(*sorted(zip(xs, ys)))
            plt.plot(xs, ys, marker="o", label=name)
        plt.xticks([1, 2, 3, 4], ["1-hop", "2-hop", "3-hop", "4-hop+"])
        plt.xlabel("Reasoning hops")
        plt.ylabel("F1")
        plt.title("E7: F1 vs reasoning complexity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(_fig_path("complexity.pdf"))
        plt.close()
        print("  -> complexity.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    make_figures(args.results)


if __name__ == "__main__":
    main()
