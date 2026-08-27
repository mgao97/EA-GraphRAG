#!/bin/bash
ROOT=/home/user/GSK/mgao/EA-GraphRAG
cd $ROOT
# 等 GraphRAG 2wiki 建索引进程结束
echo "[waiter] waiting for GraphRAG 2wiki index (pid 2013295) to finish..."
while kill -0 2013295 2>/dev/null; do sleep 30; done
echo "[waiter] GraphRAG 2wiki index done. Sleeping 20s for port 11438 release..."
sleep 20
# 启动 EA v1 三数据集
echo "[ea-v1] launching v1 on 3 datasets (n=50)..."
for ds in musique 2wikimultihopqa hotpotqa; do
  nohup /home/user/.conda/envs/graphrag/bin/python run_ea_only.py $ds 50 v1 \
    > $ROOT/logs/r_ea_v1_${ds}.log 2>&1 &
  echo "[ea-v1] started $ds pid=$!"
  sleep 5
done
echo "[ea-v1] ALL launched. Results -> results/ea_v1_<ds>.csv"
