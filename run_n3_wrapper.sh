#!/bin/bash
# run_n3_wrapper.sh — 等 N1 wrapper 全部完成后，串行执行 N3：
#   N3-GPU：主链 E2B 解码采样鲁棒性（生成+评分+分析，探索性）
#   N3-CPU：MSRF 良性 FPR 多阈值曲线（纯 CPU，基于 N1 数据）
cd /root/lalm_framing_revision_v6 || exit 9
echo "[$(date -u)] run_n3_wrapper 启动，等 run_n1_wrapper 退出..."
while pgrep -f "run_n1_wrapper.sh" >/dev/null 2>&1; do sleep 60; done
echo "[$(date -u)] N1 已退出，开始 N3"

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -u)] N3-GPU 开始"
/root/.venv/bin/python s_e2b_sampling.py --all > logs/e2b_sampling.log 2>&1
echo "[$(date -u)] N3-GPU rc=$?"

echo "[$(date -u)] N3-CPU 开始"
/root/.venv/bin/python s_msrf_curve.py > logs/msrf_curve.log 2>&1
echo "[$(date -u)] N3-CPU rc=$?  (log: logs/msrf_curve.log)"

echo "[$(date -u)] run_n3_wrapper 全部完成"
