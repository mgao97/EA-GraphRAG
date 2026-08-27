"""
统一评测脚本: 读取 HippoRAG 的 raw pred, 用与 GraphRAG/LogicRAG/LightRAG/EA
完全相同的 normalize(小写+去标点+合并空白, 不去冠词) + em + f1 重算,
输出到 results/baseline_hipporag_<dataset>.csv (列: id,question,gold,pred,em,f1)。
保证 HippoRAG 与其他 baseline 在采样、语料、指标口径上完全一致。
"""
import os, csv, string, sys
from collections import Counter

ROOT = "/home/user/GSK/mgao/EA-GraphRAG"

def normalize(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())

def em(pred, gold):
    return int(normalize(pred) == normalize(gold))

def f1(pred, gold):
    pred_t = normalize(pred).split()
    gold_t = normalize(gold).split()
    if not pred_t or not gold_t:
        return int(pred_t == gold_t)
    common = Counter(pred_t) & Counter(gold_t)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_t)
    rec = num_same / len(gold_t)
    return 2 * prec * rec / (prec + rec)

def score(dataset):
    # HippoRAG 串行 wrapper 会把每个数据集结果备份到 outputs/<ds>_n50/qa_results.csv
    src = os.path.join(ROOT, "baseline/hipporag/outputs", f"{dataset}_n50/qa_results.csv")
    if not os.path.exists(src):
        # 兼容: 若尚未备份, 直接读共享 outputs/qa_results.csv (仅当它是该数据集时)
        alt = os.path.join(ROOT, "baseline/hipporag/outputs", "qa_results.csv")
        print(f"[warn] {dataset}: {src} missing; fallback to shared qa_results.csv (only valid if it is {dataset})", flush=True)
        src = alt
    if not os.path.exists(src):
        print(f"[skip] {dataset}: {src} not found", flush=True)
        return
    rows = []
    with open(src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    out_rows = []
    for r in rows:
        gold = r.get("gold", "")
        pred = r.get("pred", r.get("answer", ""))
        out_rows.append({
            "id": r.get("id", ""),
            "question": r.get("question", ""),
            "gold": gold,
            "pred": pred,
            "em": em(pred, gold),
            "f1": round(f1(pred, gold), 4),
        })
    dst = os.path.join(ROOT, "results", f"baseline_hipporag_{dataset}.csv")
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "gold", "pred", "em", "f1"])
        w.writeheader()
        w.writerows(out_rows)
    n = len(out_rows)
    avg_em = sum(r["em"] for r in out_rows) / n
    avg_f1 = sum(r["f1"] for r in out_rows) / n
    print(f"[done] {dataset}: n={n} EM={avg_em:.4f} F1={avg_f1:.4f} -> {dst}", flush=True)

if __name__ == "__main__":
    datasets = sys.argv[1:] or ["musique", "2wikimultihopqa", "hotpotqa"]
    for d in datasets:
        score(d)
