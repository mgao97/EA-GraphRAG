import csv, os, glob, sys
ROOT="/home/user/GSK/mgao/EA-GraphRAG"
# 目标文件: (label, csv, em_col, f1_col)
targets = [
  ("GraphRAG","baseline_graphrag_musique.csv"),
  ("GraphRAG","baseline_graphrag_2wikimultihopqa.csv"),
  ("GraphRAG","baseline_graphrag_hotpotqa.csv"),
  ("LightRAG","baseline_lightrag_musique.csv"),
  ("LightRAG","baseline_lightrag_2wikimultihopqa.csv"),
  ("LightRAG","baseline_lightrag_hotpotqa.csv"),
  ("LogicRAG","baseline_logicrag_musique.csv"),
  ("LogicRAG","baseline_logicrag_2wikimultihopqa.csv"),
  ("LogicRAG","baseline_logicrag_hotpotqa.csv"),
  ("HippoRAG","baseline_hipporag_musique.csv"),
  ("HippoRAG","baseline_hipporag_2wikimultihopqa.csv"),
  ("HippoRAG","baseline_hipporag_hotpotqa.csv"),
  ("EA(optimized)","ea_only_musique.csv"),
  ("EA(optimized)","ea_only_2wikimultihopqa.csv"),
  ("EA(optimized)","ea_only_hotpotqa.csv"),
  ("EA(v1)","ea_v1_only_musique.csv"),
  ("EA(v1)","ea_v1_only_2wikimultihopqa.csv"),
  ("EA(v1)","ea_v1_only_hotpotqa.csv"),
]
def load_em_f1(path):
    r=list(csv.DictReader(open(path)))
    cols=r[0].keys()
    # 找em/f1列
    emc=[c for c in cols if c.lower() in ("em","exact_match")][0]
    f1c=[c for c in cols if c.lower() in ("f1","f1_score")][0]
    n=len(r)
    em=sum(int(float(x[emc])) for x in r)/n
    f1=sum(float(x[f1c]) for x in r)/n
    return n,round(em,4),round(f1,4)
missing=[]
rows=[]
for label,fn in targets:
    p=os.path.join(ROOT,"results",fn)
    if not os.path.exists(p):
        missing.append(fn); continue
    n,em,f1=load_em_f1(p)
    rows.append((label,fn,n,em,f1))
print(f"=== 已就绪 {len(rows)}/{len(targets)} ===")
if missing:
    print("缺失:",missing)
# 打印表格
print(f"{'Model':<14}{'Dataset':<18}{'n':>4}{'EM':>8}{'F1':>8}")
for label,fn,n,em,f1 in sorted(rows):
    ds=fn.split('_',1)[1].rsplit('_',1)[0].replace('2wikimultihopqa','2wiki')
    print(f"{label:<14}{ds:<18}{n:>4}{em:>8}{f1:>8}")
