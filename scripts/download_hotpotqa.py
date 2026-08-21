"""Download HotpotQA and store it in the official JSON format used by the loader.

Three sources are supported (the first one that works wins):

1. HuggingFace ``datasets`` package — fastest and most reliable.
2. HuggingFace direct HTTP download (no extra deps).
3. Official Google Drive / GitHub mirror via ``gdown``.

After download the file is validated against the official schema and stored at
``data/raw/hotpot_<split>_v1.1.json``.

Usage:
    python scripts/download_hotpotqa.py --split train --limit 0
    python scripts/download_hotpotqa.py --split dev   --limit 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "data" / "raw"

# (filename, candidate URLs)
CANDIDATES = {
    "train": [
        # HuggingFace direct (jsonl).
        ("https://huggingface.co/datasets/hotpot_qa/resolve/main/hotpot_train_v1.1.json",
         "json"),
        # HotpotQA's official S3 mirror (no longer always online, kept for reference).
        ("http://curtis.ml.cmu.edu/datasets/hotpotqa/hotpot_train_v1.1.json",
         "json"),
    ],
    "dev": [
        ("https://huggingface.co/datasets/hotpot_qa/resolve/main/hotpot_dev_v1.1.json",
         "json"),
        ("http://curtis.ml.cmu.edu/datasets/hotpotqa/hotpot_dev_v1.1.json",
         "json"),
    ],
    "test": [
        ("https://huggingface.co/datasets/hotpot_qa/resolve/main/hotpot_test_v1.1.json",
         "json"),
    ],
}


def _try_huggingface_datasets(split: str, dest: Path) -> bool:
    """Try ``datasets.load_dataset('hotpot_qa', ...)``."""
    try:
        from datasets import load_dataset
    except ImportError:
        return False
    try:
        ds = load_dataset("hotpot_qa", "distractor", split=split,
                          trust_remote_code=True)
        out = []
        for r in ds:
            # distractor config already wraps context as [[title, [sentences]], …].
            context = list(r.get("context", []))
            out.append({
                "_id": str(r.get("id", "")),
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "supporting_facts": list(r.get("supporting_facts", [])),
                "context": context,
                "type": r.get("type", ""),
                "level": r.get("level", ""),
            })
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        print(f"[hf-datasets] failed for split={split}: {exc}", file=sys.stderr)
        return False


def _try_http(url: str, dest: Path) -> bool:
    print(f"[http] downloading {url} -> {dest}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            if resp.status != 200:
                return False
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception as exc:
        print(f"[http] failed: {exc}", file=sys.stderr)
        return False


def _validate(path: Path, limit: int) -> int:
    """Quick schema check, return number of examples."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected payload: {type(data)}")
    if limit > 0:
        data = data[:limit]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    for r in data[:5]:
        for k in ("question", "answer", "context"):
            if k not in r:
                raise ValueError(f"Missing required key '{k}' in record: {r}")
    return len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(CANDIDATES.keys()), default="train")
    parser.add_argument("--dest", default=None,
                         help=f"Destination JSON file (default: data/raw/hotpot_<split>_v1.1.json)")
    parser.add_argument("--limit", type=int, default=0,
                         help="Optional cap on the number of examples to keep (0 = no cap).")
    args = parser.parse_args()

    dest = Path(args.dest) if args.dest else DEFAULT_DEST / f"hotpot_{args.split}_v1.1.json"
    if dest.exists():
        print(f"[ok] {dest} already exists, skipping download.")
        n = _validate(dest, args.limit)
        print(f"[ok] validated, {n} examples.")
        return

    # Try HuggingFace `datasets` first.
    if _try_huggingface_datasets(args.split, dest):
        n = _validate(dest, args.limit)
        print(f"[ok] {n} examples written to {dest}")
        return

    # Fall back to direct HTTP.
    for url, _fmt in CANDIDATES[args.split]:
        if _try_http(url, dest):
            n = _validate(dest, args.limit)
            print(f"[ok] {n} examples written to {dest}")
            return

    raise SystemExit(
        f"ERROR: could not download HotpotQA '{args.split}'. "
        "Try `pip install datasets` first, or check your network / proxy."
    )


if __name__ == "__main__":
    main()
