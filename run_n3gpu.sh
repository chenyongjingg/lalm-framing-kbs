#!/bin/bash
# run_n3gpu.sh — N3-GPU：主链 E2B 解码采样鲁棒性（修复 import os + category 后重跑）
cd /root/lalm_framing_revision_v6 || exit 9
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u)] N3-GPU 开始"
/root/.venv/bin/python s_e2b_sampling.py --all > logs/e2b_sampling.log 2>&1
echo "[$(date -u)] N3-GPU rc=$?  (log: logs/e2b_sampling.log)"
