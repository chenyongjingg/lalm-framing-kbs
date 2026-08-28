#!/bin/bash
# run_gpu1_ext20h.sh — 扩展 GPU1 流水线至 GPU0(E4B) 完成（≥20h）
# 替换 run_gpu1_audio_loop.sh。顺序（相位内严格串行，防并发抢占 GPU1）：
#   0. CPU 分析（立即，不占 GPU）：S30 FDR、S31 功效（校准 S28 的 n_queries）
#   1. 等待 S20d+S21 都退出（每轮 re-check，防并发抢占 GPU1）
#   2. GPU 相位 A：S28 Qwen2-Audio 异族生成复现（最高价值，优先给满预算）
#   3. GPU 相位 B：S29 音频确定性大样本复核
#   4. GPU 相位 C：S25 E4B 音频 official bench（strongreject+harmbench，增量）
#   5. CPU 相位：S32 DS 潜在标签主效应（需 S20d 的 strongreject/harmbench 文本
#      caches 完成——相位 1 之后必然已就绪）
#   6. 填充循环：S20b 音频 judge catch-up + S17a qwen32 音频 catch-up
#      + S25 音频 bench catch-up + sleep，直到 E4B≥10700（GPU0 完成）
# 纪律：绝不写 .complete/.done/账本；只写 results/gpu1_pipeline/+report/。
cd ~/lalm_framing_revision_v6 || exit 1
log=logs/gpu1_ext20h.log
start=$(date +%s)
echo "[ext20h $(date -u +%H:%M)] 启动（运行至 E4B≥10700 或安全上限 30h）" >> "$log"

# ---- 相位 0：CPU 分析（立即，不占 GPU1）----
echo "[ext20h $(date -u +%H:%M)] 相位0 CPU: S30+S31" >> "$log"
/root/.venv/bin/python gpu1_s30_fdr.py >> logs/gpu1_ext20h_s30.log 2>&1 \
  || echo "[ext20h] S30 失败" >> "$log"
/root/.venv/bin/python gpu1_s31_power.py >> logs/gpu1_ext20h_s31.log 2>&1 \
  || echo "[ext20h] S31 失败" >> "$log"
echo "[ext20h $(date -u +%H:%M)] S30/S31 完成" >> "$log"

# ---- 相位 1：等待 S20d+S21（GPU1 被它们占用中）----
echo "[ext20h $(date -u +%H:%M)] 等待 S20d+S21 退出" >> "$log"
while pgrep -f 'gpu1_s20d_e4b_text_bench' > /dev/null 2>&1 || \
      pgrep -f 'gpu1_s21_e2b_text_bench' > /dev/null 2>&1; do
  sleep 120
done
sleep 30
echo "[ext20h $(date -u +%H:%M)] S20d+S21 已退出" >> "$log"
# 相位 1b：等待 run_gpu1_s25_chain.sh 触发的 S25 首轮完全退出（上限 14h），
# 避免 S25 评分与 S28 生成并发抢 GPU1。
echo "[ext20h $(date -u +%H:%M)] 等待 S25 首轮（s25_chain）退出" >> "$log"
S25_WAIT=0
while pgrep -f 'gpu1_s25_e4b_audio_bench' > /dev/null 2>&1 && [ $S25_WAIT -lt 840 ]; do
  sleep 120
  S25_WAIT=$((S25_WAIT + 2))
done
sleep 20
echo "[ext20h $(date -u +%H:%M)] S25 首轮已退出（或 14h 上限），开始 GPU 相位" >> "$log"

# ---- 相位 2：S28 Qwen2 异族生成复现（n_queries 由 S31 校准）----
# 幂等：s28_hetero_audio.json 已产出则跳过（修复后重启 orchestrator 时 S28 可补跑）
if [ -f results/gpu1_pipeline/s28_hetero_audio.json ]; then
  echo "[ext20h $(date -u +%H:%M)] S28 产物已存在，跳过" >> "$log"
else
  NQ=100
  if [ -f results/gpu1_pipeline/s31_power.json ]; then
    # 偏好更强功效：MDE<=0.02 的最大 n_q（实测=100），退而 MDE<=0.04 的最大，再退 100。
    NQ=$(python3 -c "
import json
d = json.load(open('results/gpu1_pipeline/s31_power.json'))
mde = d.get('mde_80', [])
strong = [m['n_queries'] for m in mde
          if m.get('mde_0_80') is not None and m['mde_0_80'] <= 0.02]
mod = [m['n_queries'] for m in mde
       if m.get('mde_0_80') is not None and m['mde_0_80'] <= 0.04]
print(max(strong) if strong else (max(mod) if mod else 100))
")
  fi
  echo "[ext20h $(date -u +%H:%M)] S28 启动 n_q=$NQ（S31 校准，预算 420min）" >> "$log"
  CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s28_hetero_audio.py \
    --n-queries "$NQ" --max-min 420 --batch 4 \
    >> logs/gpu1_ext20h_s28.log 2>&1 || echo "[ext20h] S28 失败" >> "$log"
  echo "[ext20h $(date -u +%H:%M)] S28 完成" >> "$log"
fi

# ---- 相位 3：S29 音频确定性大样本复核 ----
if [ -f results/gpu1_pipeline/s29_determinism_audio.json ]; then
  echo "[ext20h $(date -u +%H:%M)] S29 产物已存在，跳过" >> "$log"
else
  echo "[ext20h $(date -u +%H:%M)] S29 启动" >> "$log"
  CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s29_determinism_audio.py \
    --n 120 >> logs/gpu1_ext20h_s29.log 2>&1 \
    || echo "[ext20h] S29 失败" >> "$log"
  echo "[ext20h $(date -u +%H:%M)] S29 完成" >> "$log"
fi

# ---- 相位 4：S25 E4B 音频 official bench（增量 catch-up）----
echo "[ext20h $(date -u +%H:%M)] S25 启动" >> "$log"
CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s25_e4b_audio_bench.py \
  --scorers strongreject,harmbench --batch 4 \
  >> logs/gpu1_ext20h_s25.log 2>&1 || echo "[ext20h] S25 失败" >> "$log"
echo "[ext20h $(date -u +%H:%M)] S25 完成" >> "$log"

# ---- 相位 5：S32 DS 潜在标签主效应（CPU）----
echo "[ext20h $(date -u +%H:%M)] S32 启动" >> "$log"
/root/.venv/bin/python gpu1_s32_ds_main_effect.py >> logs/gpu1_ext20h_s32.log 2>&1 \
  || echo "[ext20h] S32 失败" >> "$log"
echo "[ext20h $(date -u +%H:%M)] S32 完成" >> "$log"

# ---- 相位 6：填充循环（音频 catch-up 三腿 + sleep，直至 E4B≥10700）----
RND=0
while :; do
  now=$(date +%s)
  el=$(( (now - start) / 60 ))
  N=$(wc -l < responses/P1_PILOT/gemma_4_e4b_responses.jsonl 2>/dev/null || echo 0)
  # GPU0 完成 = stage_p1_pilot.py 退出（含评分/effects 收尾）；E4B≥10700 为回退
  if ! pgrep -f 'stage_p1_pilot.py' > /dev/null 2>&1 || [ "$N" -ge 10700 ]; then
    echo "[ext20h $(date -u +%H:%M)] GPU0 完成（stage_p1_pilot 退出或 E4B=$N≥10700），退出" >> "$log"
    break
  fi
  if [ "$el" -ge 1800 ]; then
    echo "[ext20h $(date -u +%H:%M)] 安全上限 30h 到（el=${el}min），退出" >> "$log"
    break
  fi
  if pgrep -f 'gpu1_s20d_e4b_text_bench' > /dev/null 2>&1 || \
     pgrep -f 'gpu1_s21_e2b_text_bench' > /dev/null 2>&1; then
    echo "[ext20h $(date -u +%H:%M)] S20d/S21 复活，跳过本轮" >> "$log"
    sleep 300
    continue
  fi
  RND=$((RND + 1))
  echo "[ext20h $(date -u +%H:%M)] 填充轮#$RND（E4B=$N, el=${el}min）" >> "$log"
  CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s20b_e4b_text_judges.py \
    --judge both --modality audio --batch-size 4 \
    >> logs/gpu1_ext20h_s20b.log 2>&1
  echo "[ext20h $(date -u +%H:%M)] 轮#$RND S20b 音频完成" >> "$log"
  CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s17_e4b_audio_qwen32.py \
    --modality audio >> logs/gpu1_ext20h_s17a.log 2>&1
  echo "[ext20h $(date -u +%H:%M)] 轮#$RND S17a qwen32 音频完成" >> "$log"
  CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s25_e4b_audio_bench.py \
    --scorers strongreject,harmbench --batch 4 \
    >> logs/gpu1_ext20h_s25fill.log 2>&1
  echo "[ext20h $(date -u +%H:%M)] 轮#$RND S25 音频 bench 完成" >> "$log"
  sleep 1200
done
echo "[ext20h $(date -u +%H:%M)] 结束" >> "$log"
