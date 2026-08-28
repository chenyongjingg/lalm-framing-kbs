#!/bin/bash
# auto_chain_v661.sh — D→P0→P1-PILOT→评分 自动串联（v6.6 修复版执行链）
# 用法：bash auto_chain_v661.sh <stage>   stage ∈ {p0, p1_pilot, score}
# 每次调用只跑一个阶段（GPU 串行），由外部循环/wait 控制
# v6.6.1-fix：注释声明 stage ∈ {p0, p1_pilot, score} 但 case 无 score 分支，
# 与审查"声明≠实现"纪律冲突。score 阶段的职责已由 stage_p1_pilot.py 内部
# 评分循环承担（同进程内 4 评分器顺序执行），无需独立分支；此处将声明
# 收窄为实际支持的分支，并加显式拒绝提示，杜绝误导。
set -u
ROOT=/root/lalm_framing_revision_v6
PY=/root/.venv/bin/python
cd "$ROOT" || exit 3

STAGE="${1:-}"
case "$STAGE" in
  p0)
    echo "[$(date '+%F %T')] P0 启动（4 评分器 + 异构交叉验证 + 中文适用性）"
    "$PY" stage_p0_measure.py --config pipeline_config.yaml --resume >> logs/stage_p0.log 2>&1
    echo "[$(date '+%F %T')] P0 退出 rc=$?"
    ;;
  p1_pilot)
    echo "[$(date '+%F %T')] P1-PILOT 启动（新 D 数据全量重推）"
    "$PY" stage_p1_pilot.py --config pipeline_config.yaml --resume >> logs/stage_p1_pilot.log 2>&1
    echo "[$(date '+%F %T')] P1-PILOT 退出 rc=$?"
    ;;
  score)
    # 说明：评分由 stage_p1_pilot.py 内建评分循环执行（同进程顺序跑 4 评分器）。
    # 不提供独立 score 分支——避免与主脚本抢模型显存。如需单独补跑评分，
    # 请直接调用 stage_p1_pilot.py --config pipeline_config.yaml --resume --score-only。
    echo "[$(date '+%F %T')] score 无独立分支：评分内建于 p1_pilot 阶段，直接运行 p1_pilot 即可"
    exit 0
    ;;
  *)
    echo "用法: $0 {p0|p1_pilot}"
    exit 1
    ;;
esac
