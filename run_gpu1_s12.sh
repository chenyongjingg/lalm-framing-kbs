#!/bin/bash
# GPU1 S12 运行 orchestrator（2026-08-14）
# 文本模态澄清重跑。smoke 先行（生成 6 + 评分）→ full（72 + 144 评分）。
# CUDA_VISIBLE_DEVICES=1。增量评分缓存（scorers_cache/s12_*）。
set -u
cd /root/lalm_framing_revision_v6 || exit 1
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs
LOG=logs/gpu1_s12.log
HB=logs/gpu1_s12.hb
echo "[$(date '+%F %T')] S12 start" >> "$LOG"
( while :; do echo "$(date '+%F %T') RUNNING" > "$HB"; sleep 300; done ) &
HBPID=$!
# 1) smoke（6 单元：生成 + 12 条评分 + 全分析路径）
/root/.venv/bin/python gpu1_s12_text_clarify.py --smoke >> "$LOG" 2>&1
RC1=$?
if [ $RC1 -ne 0 ]; then
    echo "[$(date '+%F %T')] S12 smoke FAIL rc=$RC1" >> "$LOG"
    kill $HBPID 2>/dev/null; exit $RC1
fi
echo "[$(date '+%F %T')] S12 smoke PASS" >> "$LOG"
# 2) full（72 单元：E4B 生成 72 + 144 响应评分）
/root/.venv/bin/python gpu1_s12_text_clarify.py >> "$LOG" 2>&1
RC=$?
kill $HBPID 2>/dev/null
echo "[$(date '+%F %T')] S12 exit rc=$RC" >> "$LOG"
exit $RC
