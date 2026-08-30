#!/usr/bin/env bash
# 等待 hipporag 本次统一实验的 30 题结果落盘后，转换并补全 summary。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/user/.conda/envs/grag/bin/python
SRC="$ROOT/baseline/hipporag/outputs/hotpotqa_sample/qa_results.csv"
DST="$ROOT/results/baseline_hipporag_hotpotqa_sample_minimax_api.csv"
Q30="$ROOT/data/raw/hotpotqa_sample/questions.json"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

for i in $(seq 1 120); do
  if [ -f "$SRC" ]; then
    # 校验确实是统一 30 题（question 匹配数 >= 1）
    ok=$($PY -c "import csv,json,sys
q=[x['question'] for x in json.load(open('$Q30'))]
try:
    rows=list(csv.DictReader(open('$SRC')))
    print(1 if sum(1 for r in rows if r['question'] in q)>=1 else 0)
except Exception:
    print(0)")
    if [ "$ok" = "1" ]; then
      log "hipporag unified QA result detected."
      break
    fi
  fi
  sleep 30
done

sleep 10

log "converting hipporag result -> unified CSV"
$PY - <<'PY'
import csv, string, os
def norm(s):
    s=(s or "").lower()
    s="".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())
def em(p,g): return int(norm(p)==norm(g))
def f1(p,g):
    p,g=norm(p).split(),norm(g).split()
    if not p or not g: return float(p==g)
    common={}
    for t in g: common[t]=common.get(t,0)+1
    c=0
    for t in p:
        if common.get(t,0)>0: c+=1; common[t]-=1
    if c==0: return 0.0
    prec,rec=c/len(p),c/len(g)
    return 2*prec*rec/(prec+rec)
src=os.environ["SRC"]; dst=os.environ["DST"]
rows=list(csv.DictReader(open(src)))
with open(dst,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["id","question","gold","pred","em","f1"])
    for r in rows:
        g=r.get("gold",""); p=r.get("pred","")
        w.writerow([r.get("id",""), r.get("question",""), g, p, em(p,g), round(f1(p,g),4)])
ems=[em(r["pred"],r["gold"]) for r in rows]
f1s=[f1(r["pred"],r["gold"]) for r in rows]
print(f"hipporag -> {len(rows)} rows | EM={sum(ems)/len(ems):.4f} F1={sum(f1s)/len(f1s):.4f}")
PY

log "rebuilding summary"
$PY - "$ROOT/results" <<'PY'
import csv, os, sys
res=sys.argv[1]
ds="hotpotqa_sample"
targets=[
 ("naive",    f"baseline_naive_{ds}_minimax.csv"),
 ("lightrag", f"baseline_lightrag_{ds}_minimax_api.csv"),
 ("graphrag", f"baseline_graphrag_{ds}_minimax_api.csv"),
 ("hipporag", f"baseline_hipporag_{ds}_minimax_api.csv"),
 ("logicrag", f"baseline_logicrag_{ds}_minimax_api.csv"),
]
def norm(s):
    s=(s or "").lower(); s="".join(c for c in s if c not in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"); return " ".join(s.split())
def em(p,g): return int(norm(p)==norm(g))
def f1(p,g):
    p,g=norm(p).split(),norm(g).split()
    if not p or not g: return float(p==g)
    common={}
    for t in g: common[t]=common.get(t,0)+1
    c=sum(1 for t in p if common.get(t,0)>0 and (common.__setitem__(t,common[t]-1) or True))
    return 0.0 if c==0 else 2*(c/len(p))*(c/len(g))/(c/len(p)+c/len(g))
lines=["# Unified Baseline Summary (dataset=hotpotqa_sample, n=30, LLM=MiniMax-M3 via :30001)","",
       "| baseline | EM | F1 | rows |","| --- | --- | --- | --- |"]
out=[["baseline","em","f1","rows"]]
for name,fn in targets:
    p=os.path.join(res,fn)
    if not os.path.exists(p):
        lines.append(f"| {name} | - | - | - |"); continue
    rows=list(csv.DictReader(open(p)))
    ems=[em(r.get("pred",""),r.get("gold","")) for r in rows]
    f1s=[f1(r.get("pred",""),r.get("gold","")) for r in rows]
    e=sum(ems)/len(ems); f=sum(f1s)/len(f1s)
    lines.append(f"| {name} | {e:.4f} | {f:.4f} | {len(rows)} |")
    out.append([name,f"{e:.4f}",f"{f:.4f}",len(rows)])
open(os.path.join(res,"baseline_unified_hotpotqa_sample_summary.md"),"w").write("\n".join(lines)+"\n")
with open(os.path.join(res,"baseline_unified_hotpotqa_sample_summary.csv"),"w",newline="") as f:
    csv.writer(f).writerows(out)
print("\n".join(lines))
PY

log "done"
