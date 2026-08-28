#!/usr/bin/env bash
# run_benign_control.sh — 良性对照 wait-gate 启动器
#
# 纪律：
#  - 只等生产 stage_p2c_adaptive.py 完全退出后才启动（单卡，绝不并发抢 GPU）。
#  - 再用 GPU util <10% 双保险确认空闲。
#  - setsid 后台运行 s40，心跳 logs/benign_control.hb。
#  - 幂等：s40 有 resume（响应/评分 jsonl 增量），重复启动安全。
#
# 用法： bash run_benign_control.sh [--n-queries N] [--templates 0,1,2]

cd /root/lalm_framing_revision_v6 || exit 3
NQ=${1:-100}
TPL=${2:-0,1,2}
LOG=logs/benign_control_launch.log
: > "$LOG"

echo "[$(date -u +%FT%TZ)] 等待生产 p2c_adaptive 退出..." | tee -a "$LOG"
while pgrep -f "stage_p2c_adaptive.py" >/dev/null 2>&1; do
  sleep 300
done
echo "[$(date -u +%FT%TZ)] p2c_adaptive 已退出，等待 GPU 空闲稳定..." | tee -a "$LOG"
for i in $(seq 1 30); do
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${UTIL:-99}" -lt 10 ] && [ "${MEM:-24000}" -lt 4000 ]; then
    echo "[$(date -u +%FT%TZ)] GPU 空闲 (util=${UTIL}% mem=${MEM}MiB)" | tee -a "$LOG"
    break
  fi
  sleep 60
done
# 兜底：等满 30 轮仍忙 → 绝不启动，abort（防与用户重跑的生产进程撞卡）
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${UTIL:-99}" -ge 10 ] || [ "${MEM:-24000}" -ge 4000 ]; then
  echo "[$(date -u +%FT%TZ)] 致命：等满 30 轮 GPU 仍忙 (util=${UTIL}% mem=${MEM}MiB)，"
       "abort 不启动。s40 resume 幂等，可稍后重跑本脚本。" | tee -a "$LOG"
  exit 3
fi

echo "[$(date -u +%FT%TZ)] 启动 s40_benign_control.py (n_queries=${NQ}, templates=${TPL})" | tee -a "$LOG"
setsid /root/.venv/bin/python s40_benign_control.py \
  --n-queries "$NQ" --templates "$TPL" --model gemma_4_e2b \
  >> logs/benign_control_run.log 2>&1 &
echo "[$(date -u +%FT%TZ)] pid=$!" | tee -a "$LOG"
