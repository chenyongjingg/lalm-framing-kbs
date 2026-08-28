#!/bin/bash
# GPU1 S9 完整运行 orchestrator（2026-08-14）
# CUDA_VISIBLE_DEVICES=1；smoke 已通过；本脚本跑 full + --with-audio。
set -u
cd /root/lalm_framing_revision_v6 || exit 1
export CUDA_VISIBLE_DEVICES=1
LOG=logs/gpu1_s9_full.log
HB=logs/gpu1_s9.hb
mkdir -p logs
echo "[$(date '+%F %T')] S9 full start" >> "$LOG"
# heartbeat 守护（后台）
( while :; do echo "$(date '+%F %T') RUNNING" > "$HB"; sleep 300; done ) &
HBPID=$!
/root/.venv/bin/python gpu1_s9_cross_family.py --with-audio >> "$LOG" 2>&1
RC=$?
kill $HBPID 2>/dev/null
echo "[$(date '+%F %T')] S9 exit rc=$RC" >> "$LOG"
exit $RC
