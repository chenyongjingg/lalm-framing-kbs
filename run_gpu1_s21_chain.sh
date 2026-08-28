#!/bin/bash
# run_gpu1_s21_chain.sh — S20d 完成后无缝启动 S21（GPU1 保忙）
# 用法：ssh -f ... "setsid nohup bash run_gpu1_s21_chain.sh >> logs/gpu1_s21_chain.log 2>&1 < /dev/null &"
cd ~/lalm_framing_revision_v6 || exit 1
log=logs/gpu1_s21_chain.log
echo "[chain $(date -u +%H:%M)] 等待 S20d 退出" >> "$log"
for i in $(seq 1 360); do
  if ! pgrep -f 'gpu1_s20d_e4b_text_bench' > /dev/null 2>&1; then
    break
  fi
  sleep 60
done
sleep 20
echo "[chain $(date -u +%H:%M)] S20d 已退出，启动 S21" >> "$log"
CUDA_VISIBLE_DEVICES=1 setsid nohup /root/.venv/bin/python gpu1_s21_e2b_text_bench.py --scorers strongreject,harmbench --batch 4 >> logs/gpu1_s21.log 2>&1 < /dev/null &
echo "[chain $(date -u +%H:%M)] S21_LAUNCHED_PID=$!" >> "$log"
