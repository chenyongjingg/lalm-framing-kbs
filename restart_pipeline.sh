#!/bin/bash
# 重启 v6.5 pipeline（在 P0 手动重跑完成后调用）
# 用法: bash restart_pipeline.sh
ROOT=/root/lalm_framing_revision_v6
LOG=$ROOT/logs/pipeline_restart.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# 1. 确保 P0 已完成且评分器 ≥3（v6.5.11-fix 2026-08-08：原 ≥5 是 v6.2 遗留，
#    v6.5 §4.1 收缩为 HarmBench-cls/StrongREJECT/异构 judge A（Gemma-4-E2B-it）/dual_judge 主，
#    P0 验证 n_scorers=3，原条件导致 v6.5 无法重启）
if [ -f "$ROOT/run/p0.complete" ]; then
  NS=$(grep -o '"n_scorers": *[0-9]*' "$ROOT/gates/P0_scorers.json" 2>/dev/null | head -1 | grep -o '[0-9]*')
  log "P0.complete 存在，n_scorers=$NS"
  if [ -z "$NS" ] || [ "$NS" -lt 3 ]; then
    log "⚠ P0 评分器 <3，不重启（需要先重跑 P0）"
    exit 1
  fi
else
  log "P0.complete 不存在（P0 未完成或仍在跑），不重启"
  exit 1
fi

# 2. 确保没有 pipeline 实例在跑（精确匹配，避免匹配到 restart_pipeline.sh 自身）
if ps aux | grep -E "bash [^r].*pipeline\.sh|tmux new-session.*pipeline\.sh" | grep -v grep >/dev/null; then
  log "⚠ pipeline.sh 仍在运行，先 kill"
  pkill -9 -f "bash /root/lalm_framing_revision_v6/pipeline.sh"
  sleep 3
fi

# 3. 清掉可能的残留锁
rm -f "$ROOT/logs/pipeline.lock" "$ROOT/logs/pipeline.pid"

# 4. 启动新 pipeline（tmux 会话 revision_v6）
cd "$ROOT"
tmux kill-session -t revision_v6 2>/dev/null
tmux new-session -d -s revision_v6 "bash $ROOT/pipeline.sh 2>&1 | tee -a $ROOT/logs/pipeline_main.log"
log "pipeline 已重启（tmux revision_v6）"
echo "RESTARTED"
