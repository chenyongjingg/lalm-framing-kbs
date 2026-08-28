#!/bin/bash
# GPU1 S8：下游阶段预检（CPU，2026-08-12 用户批准）
# 目的：在 4459 完成前，默认入口 dry-run 各下游阶段，验证代码路径
# import+解析+到达 gate 检查不崩溃（抓「完成后才发现下游崩」的 bug）。
# 纯 CPU/数据层；每项 timeout 180s 兜底（多数应因 gate 未满足而优雅早退）；
# 任何 rc≠0 或 traceback → 记录为待修清单（不阻塞 GPU1 流水线）。
cd ~/lalm_framing_revision_v6 || exit 1
PY=/root/.venv/bin/python
OUT=results/gpu1_pipeline/s8_preflight.jsonl
: > "$OUT"

run_one() {
    local name="$1"; shift
    local log=/tmp/s8_${name}.log
    timeout 180 "$PY" "$@" > "$log" 2>&1
    local rc=$?
    local tail1
    tail1=$(tail -1 "$log" 2>/dev/null | cut -c1-160)
    echo "{\"entry\":\"$name\",\"rc\":$rc,\"tail\":\"$tail1\"}" >> "$OUT"
    echo "[S8] $name rc=$rc | $tail1"
}

echo "=== S8 下游阶段预检 $(date -u +%FT%TZ) ==="
run_one stage_r_artifact stage_r_artifact.py --config pipeline_config.yaml
run_one gate_g1 gate_g1.py --config pipeline_config.yaml
run_one posthoc posthoc_p1_agreement.py
run_one recalc_v64 recalc_v64.py --config pipeline_config.yaml
run_one fix_bootstrap_ci fix_bootstrap_ci.py --config pipeline_config.yaml

echo "=== S8 done: $OUT ==="
