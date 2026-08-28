#!/bin/bash
# GPU1 S10b 运行 orchestrator（2026-08-14）
# smoke 先行验证 → full（含 7 评分器）。CUDA_VISIBLE_DEVICES=1。
set -u
cd /root/lalm_framing_revision_v6 || exit 1
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs
LOG=logs/gpu1_s10b.log
HB=logs/gpu1_s10b.hb
echo "[$(date '+%F %T')] S10b start" >> "$LOG"
( while :; do echo "$(date '+%F %T') RUNNING" > "$HB"; sleep 300; done ) &
HBPID=$!
# 1) smoke（6 单元，无评分）——验证抽样/澄清生成/停滞分类端到端
/root/.venv/bin/python gpu1_s10b.py --smoke >> "$LOG" 2>&1
RC1=$?
if [ $RC1 -ne 0 ]; then
    echo "[$(date '+%F %T')] S10b smoke FAIL rc=$RC1" >> "$LOG"
    kill $HBPID 2>/dev/null; exit $RC1
fi
echo "[$(date '+%F %T')] S10b smoke PASS" >> "$LOG"
# 2) full（72 单元 + 7 评分器评分）
/root/.venv/bin/python gpu1_s10b.py >> "$LOG" 2>&1
RC=$?
kill $HBPID 2>/dev/null
echo "[$(date '+%F %T')] S10b exit rc=$RC" >> "$LOG"
exit $RC
