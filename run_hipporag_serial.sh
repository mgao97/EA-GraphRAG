#!/bin/bash
# 串行跑HippoRAG三数据集, 每个跑完立即备份qa_results.csv, 避免并行覆盖
ROOT=/home/user/GSK/mgao/EA-GraphRAG
DIR=$ROOT/baseline/hipporag
cd $DIR
PY=/home/user/.conda/envs/sls/bin/python
for ds in musique 2wikimultihopqa hotpotqa; do
  echo "[hippo-serial] start $ds"
  bash run.sh $ds 50 > $ROOT/logs/r_hipporag_$ds.log 2>&1
  # 跑完立即备份
  mkdir -p outputs/${ds}_n50
  cp outputs/qa_results.csv outputs/${ds}_n50/qa_results.csv
  echo "[hipp. freed] backed up $ds -> outputs/${ds}_n50/qa_results.csv"
done
echo "[hippo-serial] ALL DONE"
