#!/bin/bash
# GPU1 S11 运行 orchestrator（2026-08-14）
# E2B 全量 3600 文本响应跨族核验。smoke 先行 → full。
# CUDA_VISIBLE_DEVICES=1。增量评分缓存（scorers_cache/），崩溃可续跑。
set -u
cd /root/lalm_framing_revision_v6 || exit 1
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs
LOG=logs/gpu1_s11.log
HB=logs/gpu1_s11.hb
echo "[$(date '+%F %T')] S11 start" >> "$LOG"
( while :; do echo "$(date '+%F %T') RUNNING" > "$HB"; sleep 300; done ) &
HBPID=$!
# 1) smoke（6 行，三评分器各 6 条 + 全分析路径）
/root/.venv/bin/python gpu1_s11_e2b_cross_family.py --smoke >> "$LOG" 2>&1
RC1=$?
if [ $RC1 -ne 0 ]; then
    echo "[$(date '+%F %T')] S11 smoke FAIL rc=$RC1" >> "$LOG"
    kill $HBPID 2>/dev/null; exit $RC1
fi
echo "[$(date '+%F %T')] S11 smoke PASS" >> "$LOG"
# 2) full（3600 行 × qwen32/judge_big/judge_small）
/root/.venv/bin/python gpu1_s11_e2b_cross_family.py >> "$LOG" 2>&1
RC=$?
kill $HBPID 2>/dev/null
echo "[$(date '+%F %T')] S11 exit rc=$RC" >> "$LOG"
exit $RC
