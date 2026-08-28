# -*- coding: utf-8 -*-
"""
posthoc_p1_agreement.py — 事后补算 P1-PILOT 评分器一致性（补全 #8 的降级路径）

背景：补全 #8（stage_p1_pilot.py 评分段接入 cohens_kappa/spearman_with_ci，
落盘 results/p1_pilot_scorer_agreement.json）部署于 2026-08-10 12:49 UTC，而当前
P1-PILOT 推理进程（PID 198729）启动于 2026-08-09 13:34 UTC —— 进程装载的是旧代码，
本轮完成后不会产出该 JSON。本脚本用与补全 #8 完全一致的算法/列/n_boot/seed，
从已评分的 results/p1_pilot_scored.parquet 复算同一产物，实现无损恢复。

幂等：若 agreement JSON 已存在（未来新代码运行已产出）→ 直接跳过。
准备未就绪（parquet 尚不存在）→ 退出 0 静默，待巡检 cron 再触发。
KBS 合理性：评分器两两一致性（k/Spearman）是"三口径方向一致"判据（prompt.md
§4.3）的量化证据，KBS 主表必需。事后从已落盘评分数据复算与原算法逐位一致，
不改变任何原始数据，属可复现工程的正规补算手段。
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LABEL_COLS = ("hb_label", "sr_label", "gemma_label",
              "judge_big_label", "judge_mistral_label")


def main():
    log = logging.getLogger("posthoc_agreement")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    scored = ROOT / "results" / "p1_pilot_scored.parquet"
    out = ROOT / "results" / "p1_pilot_scorer_agreement.json"

    if out.exists():
        log.info("agreement JSON 已存在（%s），幂等跳过", out)
        return 0
    if not scored.exists():
        log.info("scored parquet 尚不存在（评分未完成），静默等待: %s", scored)
        return 0

    from scorer_utils import cohens_kappa, spearman_with_ci

    df = pd.read_parquet(scored)
    agree = {}
    for i, a in enumerate(LABEL_COLS):
        if a not in df.columns:
            continue
        for b in LABEL_COLS[i + 1:]:
            if b not in df.columns:
                continue
            pa = df[a].to_numpy()
            pb = df[b].to_numpy()
            m = (~np.isnan(pa)) & (~np.isnan(pb))
            if m.sum() < 10:
                continue
            k = cohens_kappa(pa[m].tolist(), pb[m].tolist(), n_boot=200, seed=42)
            sp = spearman_with_ci(pa[m].tolist(), pb[m].tolist(), n_boot=200, seed=42)
            agree[f"{a}|{b}"] = {"n": int(m.sum()), "kappa": k, "spearman": sp}

    out.write_text(json.dumps(agree, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("评分器一致性(k/Spearman) 事后补算完成: %d 对 -> %s", len(agree), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
