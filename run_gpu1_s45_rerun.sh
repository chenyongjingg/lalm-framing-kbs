#!/bin/bash
# GPU1 S4+S5 修复后重跑（R102-续⑤，2026-08-12 用户批准）
# 纪律：CUDA_VISIBLE_DEVICES=1 隔离 GPU0(4459)；每阶段 smoke→full；
#   setsid + 心跳 logs/gpu1_s45_rerun.hb + stderr 落 logs/gpu1_*.log；
#   不写 4459 账本/不写 .complete/done；只写 results/gpu1_pipeline/ + report/；
#   结果由 30min 同步守护进程自动 commit/push（本脚本不碰 git）。
cd ~/lalm_framing_revision_v6 || exit 1
export CUDA_VISIBLE_DEVICES=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
PY=/root/.venv/bin/python
HB=logs/gpu1_s45_rerun.hb
LOGDIR=logs
: > "$HB"
echo "$(date -u +%FT%TZ) s45 rerun start" >> "$HB"

run_stage() {
    local script="$1"; local name="$2"; shift 2
    local stage_log="$LOGDIR/gpu1_${name}.log"
    echo "$(date -u +%FT%TZ) stage $name start (smoke)" >> "$HB"
    "$PY" "${script}.py" --smoke >> "$stage_log" 2>&1 &
    local spid=$!
    while kill -0 "$spid" 2>/dev/null; do
        echo "$(date -u +%FT%TZ) stage $name smoke running (pid=$spid)" >> "$HB"
        sleep 600
    done
    wait "$spid"; local src=$?
    if [ "$src" -ne 0 ]; then
        echo "$(date -u +%FT%TZ) stage $name SMOKE FAILED rc=$src → skip full, continue" >> "$HB"
        return 1
    fi
    echo "$(date -u +%FT%TZ) stage $name smoke OK → full" >> "$HB"
    "$PY" "${script}.py" "$@" >> "$stage_log" 2>&1 &
    spid=$!
    while kill -0 "$spid" 2>/dev/null; do
        echo "$(date -u +%FT%TZ) stage $name full running (pid=$spid)" >> "$HB"
        sleep 600
    done
    wait "$spid"; local frc=$?
    echo "$(date -u +%FT%TZ) stage $name full done rc=$frc" >> "$HB"
    return $frc
}

run_stage gpu1_s4_convergence_full s4
run_stage gpu1_s5_zh_audio s5
echo "$(date -u +%FT%TZ) s45 rerun complete" >> "$HB"
exit 0
