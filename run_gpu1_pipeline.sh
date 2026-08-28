#!/bin/bash
# GPU1 空闲期测量验证流水线 orchestrator（2026-08-12，用户批准）
# 纪律：等待确认重跑退出 → S1(zh 评分器收敛) → S2(顺序确定性) → S3(ShieldGemma 基准)。
#   - CUDA_VISIBLE_DEVICES=1 隔离 GPU0 主运行（4459）
#   - 每阶段 smoke → full；失败不阻断后续阶段（独立产物）
#   - setsid + 心跳（logs/gpu1_pipeline.hb）+ stderr 落盘（logs/gpu1_*.log）
#   - 不写 4459 账本、不写 .complete/done；只写 results/gpu1_pipeline/ + report/
#   - 结果由 30min 同步守护进程自动 commit/push（本脚本不碰 git）
cd ~/lalm_framing_revision_v6 || exit 1
export CUDA_VISIBLE_DEVICES=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
PY=/root/.venv/bin/python
HB=logs/gpu1_pipeline.hb
LOGDIR=logs
: > "$HB"
echo "$(date -u +%FT%TZ) orchestrator start" >> "$HB"

# ── 等待确认重跑（validate_batch_text）完全退出 ──
wait_confirm() {
    echo "$(date -u +%FT%TZ) waiting for validate_batch_confirm to exit..." >> "$HB"
    while :; do
        if grep -q "python EXITED" logs/validate_batch_confirm.hb 2>/dev/null; then
            rc=$(grep "python EXITED" logs/validate_batch_confirm.hb \
                 | tail -1 | awk '{print $NF}')
            echo "$(date -u +%FT%TZ) confirm exited rc=$rc → start pipeline" >> "$HB"
            return 0
        fi
        # 进程消失但未写 EXITED（launcher 被误杀）→ 不再空等
        if ! pgrep -f "validate_batch_text.py" >/dev/null 2>&1; then
            echo "$(date -u +%FT%TZ) WARN confirm process gone without EXITED marker; proceed" >> "$HB"
            return 0
        fi
        sleep 60
    done
}

# ── 单阶段：smoke → full ──
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

wait_confirm
run_stage gpu1_s1_zh_convergence s1
run_stage gpu1_s2_determinism     s2
run_stage gpu1_s3_guard_bench     s3
run_stage gpu1_s4_convergence_full s4
run_stage gpu1_s5_zh_audio         s5
echo "$(date -u +%FT%TZ) pipeline complete" >> "$HB"
exit 0
