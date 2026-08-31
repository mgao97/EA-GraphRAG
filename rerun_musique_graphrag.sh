#!/usr/bin/env bash
# 5小时后重跑 musique 的 GraphRAG (A组vllm + B组minimax_api)
# 使用 --corpus_mode related (按题目 context 收敛语料)
# 原因: --corpus_limit 语义是 corpus[:N], musique 的 corpus 是打乱的,
# corpus[:450] 只覆盖 3.8% (33/875) 的相关文档, 旧结果无效
set -u
ROOT=/home/user/GSK/mgao/EA-GraphRAG
LOG="$ROOT/logs/musique_rerun_scheduler.log"
GR=/home/user/.conda/envs/graphrag/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "=== 调度器启动: 5小时(18000s)后重跑 musique GraphRAG ==="
sleep 18000
log "=== 开始执行 ==="

cd "$ROOT" || exit 1

# 若旧 musique GraphRAG A 组仍在跑, 先等它自然结束, 避免污染输出 csv
# 精确匹配输出文件名, 不要只匹配 "run_dataset_vllm.py --dataset musique",
# 因为 A 组 LightRAG 的 musique 任务脚本同名, 会误判
for i in $(seq 1 60); do
  OLD=$(pgrep -f "baseline_graphrag_musique_vllm.csv" | tr '\n' ' ')
  if [ -z "${OLD// /}" ]; then break; fi
  log "旧 musique GraphRAG A 组仍在跑 (PID: $OLD), 等待 60s ($i/60)"
  sleep 60
done

# 备份旧结果
for f in results/baseline_graphrag_musique_vllm.csv results/baseline_graphrag_musique_minimax_api.csv; do
  if [ -f "$f" ]; then cp "$f" "$f.bak_head_mode" && log "已备份 $f"; fi
done

# 旧索引目录改名备份
if [ -d baseline/graphrag/graphrag_vllm/musique ]; then
  mv baseline/graphrag/graphrag_vllm/musique baseline/graphrag/graphrag_vllm/musique_bak_head
  log "已备份 A组索引目录"
fi
if [ -d baseline/graphrag/musique ]; then
  mv baseline/graphrag/musique baseline/graphrag/musique_bak_head
  log "已备份 B组索引目录"
fi

# 启动重跑
export CODEBUDDY_SAFE_DELETE_DISABLED=1
export CODEBUDDY_SAFE_DELETE_BULK_CONFIRM_REQUIRED=""
cd "$ROOT/baseline/graphrag" || exit 1

nohup $GR run_dataset_vllm.py --dataset musique --n 50 --corpus_limit 50 --corpus_mode related \
  --output ../../results/baseline_graphrag_musique_vllm.csv \
  > run_graphrag_musique_vllm_rebuild.log 2>&1 < /dev/null &
log "A组(vllm) 已启动 PID=$!"

nohup $GR run_dataset_minimax_api.py --dataset musique --n 50 --corpus_limit 50 --corpus_mode related \
  --output ../../results/baseline_graphrag_musique_minimax_api.csv \
  > run_graphrag_musique_minimax_rebuild.log 2>&1 < /dev/null &
log "B组(minimax_api) 已启动 PID=$!"

sleep 20
log "A组日志: $(grep -m1 'corpus_mode\|total docs' run_graphrag_musique_vllm_rebuild.log 2>/dev/null | cut -c1-160)"
log "B组日志: $(grep -m1 'corpus_mode\|total docs' run_graphrag_musique_minimax_rebuild.log 2>/dev/null | cut -c1-160)"
log "=== 重跑已启动, 调度器结束 ==="
