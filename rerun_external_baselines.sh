#!/bin/bash
# 2026-08-24 v2：主链 p2c 完成后重跑全部外部基线（GradSafe 真实 + ShieldGemma + WildGuard）
# 背景：
#  - 11:45/11:47 ShieldGemma/WildGuard 因 huggingface.co 不可达加载失败；stage_p2_baselines.py
#    已打 offline 补丁（config network.offline=true → HF_HUB_OFFLINE=1 纯本地缓存，零网络）。
#  - GradSafe 真实推理在 CPU bf16 上不可行（~100h+），主链 gradsafe 已被替换为 GPU 版
#    （gradsafe_real.load_model + device_map="auto"）；此处补跑。
# 触发：run/p2c.complete 出现（主链 p2c 完成）或 pipeline.sh 退出。
# 安全窗口：p2c 之后主链只剩 G2/F/R（CPU 为主），GPU 空闲可给基线推理。
LOG=/root/lalm_framing_revision_v6/logs/baseline_rerun_launcher.log
{
  cd /root/lalm_framing_revision_v6
  echo "[$(date +%F\ %T)] 启动：等待主链 p2c 完成"
  while [ ! -f run/p2c.complete ] && pgrep -f "lalm_framing_revision_v6/pipeline.sh" >/dev/null; do
    sleep 60
  done
  echo "[$(date +%F\ %T)] 触发条件满足，sleep 120 让主链释放 GPU"
  sleep 120
  echo "[$(date +%F\ %T)] === 外部基线一次跑完: --gradsafe-infer --infer --wildguard-infer --evaluate ==="
  /root/.venv/bin/python stage_p2_baselines.py --config pipeline_config.yaml --resume \
      --gradsafe-infer --infer --wildguard-infer --evaluate > logs/baseline_rerun.log 2>&1
  echo "baseline rc=$?"
  echo "=== baseline_rerun.log tail ==="; tail -12 logs/baseline_rerun.log
  echo "[$(date +%F\ %T)] 外部基线重跑完成"
} >> "$LOG" 2>&1
