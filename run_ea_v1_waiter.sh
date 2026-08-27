#!/bin/bash
ROOT=/home/user/GSK/mgao/EA-GraphRAG
cd $ROOT
# 等 EA optimized 三进程全部结束
echo "[v1-waiter] waiting for EA optimized 3 procs to finish..."
while ps -eo args 2>/dev/null | grep -q "run_ea_only.py .* configs/optimized_11435.yaml" && ! ps -eo args 2>/dev/null | grep "run_ea_only.py" | grep -v grep | grep -q "optimized_11435"; do
  sleep 30
done
# 更稳妥: 直接等三个optimized csv都出现且进程不在
while ps -eo args 2>/dev/null | grep "optimized_11435.yaml" | grep -v grep | grep -q "run_ea_only"; do sleep 30; done
echo "[v1-waiter] EA optimized done. Launching EA v1 with correct config..."
for ds in musique 2wikimultihopqa hotpotqa; do
  setsid /home/user/.conda/envs/graphrag/bin/python run_ea_only.py $ds 50 configs/ea_v1_11438.yaml \
    > logs/r_ea_v1_${ds}.log 2>&1 < /dev/null &
  disown
  echo "[v1-waiter] started EA v1 $ds pid=$!"
  sleep 3
done
echo "[v1-waiter] EA v1 ALL launched."
