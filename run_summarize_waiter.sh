#!/bin/bash
ROOT=/home/user/GSK/mgao/EA-GraphRAG
cd $ROOT
while true; do
  status=$(/home/user/.conda/envs/graphrag/bin/python -c "
import summarize_final as s
print(f'{len(s.rows)}/{len(s.targets)}')
" 2>/dev/null)
  echo "[sum-waiter] $(date +%H:%M) ready=$status"
  if [ "$status" = "18/18" ]; then
    echo "=== ALL DONE, final table ==="
    /home/user/.conda/envs/graphrag/bin/python summarize_final.py
    break
  fi
  sleep 300
done
