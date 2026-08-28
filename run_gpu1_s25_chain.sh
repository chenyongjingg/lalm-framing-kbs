#!/bin/bash
# run_gpu1_s25_chain.sh — S20d 与 S21 均完成后无缝启动 S25（E4B 音频 strongreject+harmbench，GPU1 保忙）
# S24 模态效应（audio>>text）需官方基准评分器腿支撑；S25 镜像 S20d 到音频侧。
# 用法：ssh -f ... "setsid nohup bash run_gpu1_s25_chain.sh >> logs/gpu1_s25_chain.log 2>&1 < /dev/null &"
cd ~/lalm_framing_revision_v6 || exit 1
log=logs/gpu1_s25_chain.log
echo "[chain $(date -u +%H:%M)] 等待 S20d 与 S21 均退出（上限 8h）" >> "$log"
for i in $(seq 1 480); do
  if ! pgrep -f 'gpu1_s20d_e4b_text_bench' > /dev/null 2>&1 \
     && ! pgrep -f 'gpu1_s21_e2b_text_bench' > /dev/null 2>&1; then
    break
  fi
  sleep 60
done
if pgrep -f 'gpu1_s25_e4b_audio_bench' > /dev/null 2>&1; then
  echo "[chain $(date -u +%H:%M)] S25 已在运行，退出" >> "$log"
  exit 0
fi
sleep 20
echo "[chain $(date -u +%H:%M)] S20d/S21 均已退出，启动 S25" >> "$log"
CUDA_VISIBLE_DEVICES=1 setsid nohup /root/.venv/bin/python gpu1_s25_e4b_audio_bench.py --scorers strongreject,harmbench --batch 4 >> logs/gpu1_s25.log 2>&1 < /dev/null &
echo "[chain $(date -u +%H:%M)] S25_LAUNCHED_PID=$!" >> "$log"
