#!/bin/bash
# GPU1 串行批次：S36（强制解码补全）→ S37（ShieldGemma 交叉一致）
cd /root/lalm_framing_revision_v6
LOG=run_gpu1_s36_s37.log
echo "[wrap] $(date -u) ===== start S36 =====" >> $LOG
CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s36_forced_complete.py >> $LOG 2>&1
echo "[wrap] $(date -u) S36 rc=$? (2026-08-16)" >> $LOG
echo "[wrap] $(date -u) ===== start S37 =====" >> $LOG
CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s37_shieldgemma_cross.py >> $LOG 2>&1
echo "[wrap] $(date -u) S37 rc=$? (2026-08-16)" >> $LOG
echo "[wrap] $(date -u) ===== done =====" >> $LOG
