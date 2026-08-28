#!/bin/bash
# run_n1_wrapper.sh — 等 S40 全部退出后，串行执行 N1（良性误报率扫描 + MSRF FPR）
# 和 S40 官方分析（s40b_analyze.py）。GPU 纪律：N1 各阶段串行单模型驻留。
cd /root/lalm_framing_revision_v6 || exit 9
echo "[$(date -u)] run_n1_wrapper 启动，等 S40 退出..."
# 等 run_s40_wrapped / s40_benign_control 进程消失
while pgrep -f "run_s40_wrapped.py" >/dev/null 2>&1; do sleep 60; done
while pgrep -f "s40_benign_control.py" >/dev/null 2>&1; do sleep 60; done
echo "[$(date -u)] S40 已退出，开始 N1 + s40b"

# 1) S40 官方分析（纯 CPU）
/root/.venv/bin/python s40b_analyze.py > logs/s40b_analyze.log 2>&1
echo "[$(date -u)] s40b_analyze rc=$?  (log: logs/s40b_analyze.log)"

# 2) N1 良性响应误报率扫描（串行，各单模型）
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for stage in shieldgemma wildguard harmbench strongreject msrf analyze; do
  echo "[$(date -u)] N1 stage=$stage 开始"
  /root/.venv/bin/python s_benign_fpr.py --$stage >> logs/benign_fpr.log 2>&1
  echo "[$(date -u)] N1 stage=$stage rc=$?"
done
echo "[$(date -u)] run_n1_wrapper 全部完成"
