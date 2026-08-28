#!/bin/bash
# 2026-08-24 v3：GradSafe 显存修复后补跑 GradSafe + WildGuard + Evaluate
# ShieldGemma 已由 launcher 完成（16200 条，shieldgemma_scores.jsonl），跳过 --infer。
# 后台独立运行，setsid 脱离 ssh。外层日志与 python 输出分离，避免互相截断。
LOG=/root/lalm_framing_revision_v6/logs/gsw_rerun_launcher.log
PYLOG=/root/lalm_framing_revision_v6/logs/gsw_rerun.log
{
  cd /root/lalm_framing_revision_v6
  echo "[$(date +%F\ %T)] === GradSafe + WildGuard + Evaluate 补跑启动 ==="
  /root/.venv/bin/python stage_p2_baselines.py --config pipeline_config.yaml --resume \
      --gradsafe-infer --wildguard-infer --evaluate > "$PYLOG" 2>&1
  echo "rc=$?"
  echo "=== 尾部 ==="; tail -15 "$PYLOG"
  echo "[$(date +%F\ %T)] 补跑结束"
} >> "$LOG" 2>&1
