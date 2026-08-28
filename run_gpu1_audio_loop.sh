#!/bin/bash
# run_gpu1_audio_loop.sh — S20d 与 S21 都结束后，E4B 音频增量 catch-up 循环（GPU1 保忙直到 E4B≥10700）
# 顺序：S20d →(run_gpu1_s21_chain.sh)→ S21 → 本循环（等 S20d+S21 都退出才跑第一轮）
# 每轮：S20b judge_big+judge_small 音频 catch-up → S17a qwen32 音频 catch-up → 睡 1500s
# 每轮前再查一次 S20d/S21 是否复活，避免并发抢占 GPU1。
# E4B≥10700 退出（最终 catch-up + S17 Part B 由 8f4ea1da cron 接管）。
cd ~/lalm_framing_revision_v6 || exit 1
log=logs/gpu1_audio_loop.log
echo "[audio-loop $(date -u +%H:%M)] 启动，等待 S20d 与 S21 都退出" >> "$log"
# 等待 S20d 与 S21 进程都不存在
while pgrep -f 'gpu1_s20d_e4b_text_bench' > /dev/null 2>&1 || \
      pgrep -f 'gpu1_s21_e2b_text_bench' > /dev/null 2>&1; do
  sleep 60
done
sleep 30
echo "[audio-loop $(date -u +%H:%M)] S20d+S21 已退出，开始音频 catch-up 循环" >> "$log"
RND=0
while :; do
  N=$(wc -l < responses/P1_PILOT/gemma_4_e4b_responses.jsonl 2>/dev/null || echo 0)
  if [ "$N" -ge 10700 ]; then
    echo "[audio-loop $(date -u +%H:%M)] E4B=$N ≥10700，退出（cron 接管）" >> "$log"
    break
  fi
  # 每轮前再次确认 S20d/S21 未复活
  if pgrep -f 'gpu1_s20d_e4b_text_bench' > /dev/null 2>&1 || \
     pgrep -f 'gpu1_s21_e2b_text_bench' > /dev/null 2>&1; then
    echo "[audio-loop $(date -u +%H:%M)] 检测到 S20d/S21 运行中，跳过本轮" >> "$log"
    sleep 1500
    continue
  fi
  RND=$((RND+1))
  echo "[audio-loop $(date -u +%H:%M)] 轮#$RND 开始（E4B=$N）" >> "$log"
  CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s20b_e4b_text_judges.py \
    --judge both --modality audio --batch-size 4 >> logs/gpu1_audio_loop_s20b.log 2>&1
  echo "[audio-loop $(date -u +%H:%M)] 轮#$RND S20b 音频完成" >> "$log"
  CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s17_e4b_audio_qwen32.py \
    --modality audio >> logs/gpu1_audio_loop_s17a.log 2>&1
  echo "[audio-loop $(date -u +%H:%M)] 轮#$RND S17a qwen32 音频完成" >> "$log"
  sleep 1500
done
echo "[audio-loop $(date -u +%H:%M)] 结束" >> "$log"
