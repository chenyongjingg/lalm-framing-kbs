#!/bin/bash
# 2026-08-24 v4：与主链错峰的外部基线补跑守护。
# 序列：仅当 (a) run/p2c.complete 出现（主链 p2c 已完成）或
# (b) 主链已退出且 p2c 曾启动过（run/p2c.pid 存在，链死于 p2c 中）
# 才启动 GradSafe + WildGuard + Evaluate。ShieldGemma 已由 v2 launcher 完成。
# 遵守铁律：绝不与主链 p2c（GPU 长板）抢资源；不打断运行中的分片。
# 另：另一 agent 正在服务器协同（intent-LoRA 已修 NaN 重训中），本守护
# 只在 p2c 结束后接续，绝不与其它进程同时抢 GPU。
LOG=/root/lalm_framing_revision_v6/logs/baseline_after_p2c.log
PYLOG=/root/lalm_framing_revision_v6/logs/baseline_after_p2c_py.log
{
  cd /root/lalm_framing_revision_v6
  echo "[$(date +%F\ %T)] v4-launcher 启动：等待主链 p2c 完成"
  while [ ! -f run/p2c.complete ]; do
    if ! pgrep -f "lalm_framing_revision_v6/pipeline.sh" >/dev/null 2>&1; then
      if [ -f run/p2c.pid ]; then
        echo "[$(date +%F\ %T)] 主链已退出且 p2c 曾启动（run/p2c.pid 存在），按 (b) 触发补跑"
        break
      fi
      echo "[$(date +%F\ %T)] 主链已退出但 p2c 未启动，暂不补跑（等新链）"
      # 继续等：可能链将重启
      sleep 120
      continue
    fi
    sleep 60
  done
  echo "[$(date +%F\ %T)] 触发条件满足，sleep 120 让 GPU 稳定"
  sleep 120
  echo "[$(date +%F\ %T)] === GradSafe + WildGuard + Evaluate 补跑（p2c 后）==="
  /root/.venv/bin/python stage_p2_baselines.py --config pipeline_config.yaml --resume \
      --gradsafe-infer --wildguard-infer --evaluate > "$PYLOG" 2>&1
  echo "rc=$?"
  echo "=== 尾部 ==="; tail -12 "$PYLOG"
  echo "[$(date +%F\ %T)] v4-launcher 补跑结束"
} >> "$LOG" 2>&1
