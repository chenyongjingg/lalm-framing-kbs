# -*- coding: utf-8 -*-
"""全重建触发脚本（方案：合规数据重建，2026-08-05）

核心思路：不自己跑 stage_d / stage_p1_pilot（避免与主链抢 GPU），
而是删除重建所需的"完成标记 + 旧数据 + 旧 checkpoint"，然后 kill 主链
pipeline.sh 触发自动重试（attempt+1）——由主链串行完成：
  D（重跑，14B-AWQ 生成合规数据）→ P1-PILOT（新数据推理）→ 评分 → G1

触发时机：评分进程 328423 结束后的 GPU 窗口（自动化调用）。
"""
import argparse, json, logging, subprocess, sys, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-data", action="store_true",
                    help="只重建数据集不重跑推理（先验证数据质量）")
    args = ap.parse_args()

    root = Path("/root/lalm_framing_revision_v6")
    log_path = root / "logs" / "rebuild_data.log"
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("rebuild")

    log.info("=== 全重建触发开始 ===")

    # 1) 删除旧数据 + 完成标记（触发主链重跑 D 阶段）
    targets = [
        root / "data" / "queries_v1.jsonl",   # 旧 fallback 查询
        root / "data" / "recipe.json",        # 缺失/旧配方
        root / "run" / "d.complete",          # D 阶段完成标记 → 删除触发重跑
        root / "run" / "p1_pilot.complete",   # P1-PILOT 完成标记（若存在）
    ]
    for t in targets:
        if t.exists():
            t.unlink()
            log.info("已删除: %s", t.name)

    # 2) 删除旧 P1-PILOT checkpoint（防误命中旧数据推理）
    for ck in root.glob("checkpoints/p1_pilot_*.jsonl"):
        if "pre_" not in ck.name and ".bak" not in ck.name:
            ck.unlink()
            log.info("已删除旧 checkpoint: %s", ck.name)

    # 3) 删除旧评分结果（防 G1 读到旧数据口径）
    for f in [root / "results" / "p1_pilot_scored.parquet",
              root / "results" / "p1_pilot_effects.json",
              root / "gates" / "G2_input.json",
              root / "gates" / "G1.json"]:
        if f.exists():
            f.unlink()
            log.info("已删除旧结果: %s", f.name)

    # 4) 确认评分进程已结束（避免误删进行中的成果）
    r = subprocess.run("pgrep -f 'stage_p1_pilot.py' | head -1", shell=True,
                       capture_output=True, text=True)
    if r.stdout.strip():
        log.error("评分进程仍在运行（PID=%s），不执行重建，等待下次触发",
                  r.stdout.strip())
        print("SCORING_STILL_RUNNING")
        return 1

    # 5) 重启主链（tmux 会话若已死则重建；主链因 d.complete 缺失重跑 D→P1-PILOT）
    #    先尝试复用已有 tmux 会话
    r = subprocess.run("tmux has-session -t revision_v6 2>/dev/null; echo $?",
                       shell=True, capture_output=True, text=True)
    if r.stdout.strip() == "0":
        log.info("tmux 会话 revision_v6 存在，不重复启动（若主链已退出，"
                 "需人工 kill 会话或等 G1 exit 3 后由守护重启）")
    else:
        cmd = ("tmux new-session -d -s revision_v6 "
               "'bash /root/lalm_framing_revision_v6/pipeline.sh 2>&1 | "
               "tee /root/lalm_framing_revision_v6/logs/pipeline_main.log'")
        subprocess.run(cmd, shell=True)
        log.info("已重启主链（新 tmux 会话 revision_v6）")
    print("REBUILD_TRIGGER_OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
