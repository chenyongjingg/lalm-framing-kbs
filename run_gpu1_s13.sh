#!/bin/bash
# GPU1 S13 运行 orchestrator（2026-08-14）
# 评分器 test-retest（300 抽样 × 3 评分器 × 2 pass）。smoke → full。
# CUDA_VISIBLE_DEVICES=1。增量缓存 scorers_cache/s13_*。
set -u
cd /root/lalm_framing_revision_v6 || exit 1
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs
LOG=logs/gpu1_s13.log
HB=logs/gpu1_s13.hb
echo "[$(date '+%F %T')] S13 start" >> "$LOG"
( while :; do echo "$(date '+%F %T') RUNNING" > "$HB"; sleep 300; done ) &
HBPID=$!
# 1) smoke（6 抽样 × 2 pass，全分析路径）
/root/.venv/bin/python gpu1_s13_test_retest.py --smoke >> "$LOG" 2>&1
RC1=$?
if [ $RC1 -ne 0 ]; then
    echo "[$(date '+%F %T')] S13 smoke FAIL rc=$RC1" >> "$LOG"
    kill $HBPID 2>/dev/null; exit $RC1
fi
echo "[$(date '+%F %T')] S13 smoke PASS" >> "$LOG"
# 2) full（300 × 2 pass）
/root/.venv/bin/python gpu1_s13_test_retest.py >> "$LOG" 2>&1
RC=$?
kill $HBPID 2>/dev/null
echo "[$(date '+%F %T')] S13 exit rc=$RC" >> "$LOG"
exit $RC
