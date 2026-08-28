#!/usr/bin/env python3
"""GPU1 S3：异构安全分类器公开基准验证（2026-08-12，用户批准）。

成员：ShieldGemma-9B（唯一离线可用异构安全分类器；Llama-Guard-3-8B 与
WildGuard 均 gated 不可用——见 audit_reviews.md 预登记更正）。
基准：gate 同一 602 行人工标注公开基准（load_original_responses →
jalmbench），与 harmbench/strongreject/gemma/judge_big/judge_small/
cross_check 的既有数字直接可比 → 扩展现有 cross_validation 块。

复用 validate_single_scorer（含分布塌缩检测——单类 >95% 即 RuntimeError，
评分器标记不可用，防幻觉纪律 #1）。零人工标注；只写独立产物，不触碰 gates/。
"""
import argparse
import gc
import json
import os
import sys
import yaml
from pathlib import Path

from gpu1_common import (ShieldGemmaScorer, resolve_root, get_logger,
                         release)

SCORERS = ["shieldgemma"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--smoke", action="store_true", help="小样本冒烟（前 40 行）")
    ap.add_argument("--dry", action="store_true", help="仅数据层，不加载模型")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = resolve_root(cfg)
    log = get_logger("s3", root)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 基准数据 ----
    from stage_p0_measure import load_original_responses, validate_single_scorer
    orig_dir = Path(os.path.expanduser(
        cfg.get("original_data_dir", "~/original_experiment_data")))
    bench_dir = orig_dir / "jalmbench"
    if not bench_dir.exists():
        log.error("基准目录不存在: %s", bench_dir)
        return 3
    bench_rows = load_original_responses(orig_dir, log)
    if args.smoke:
        bench_rows = bench_rows[:40]
    n_labelled = sum(1 for r in bench_rows
                     if r.get("label") is not None
                     or r.get("label_heuristic") is not None)
    log.info("S3 基准行=%d（可标注 %d）", len(bench_rows), n_labelled)
    if n_labelled == 0:
        log.error("基准无标注行 → S3 中止（如实记录，不产出）")
        return 3

    # ---- 评分 ----
    metrics = {}
    if not args.dry:
        for sn in SCORERS:
            scorer = ShieldGemmaScorer(cfg.get("shieldgemma_model",
                                               "google/shieldgemma-9b"))
            try:
                m = validate_single_scorer(scorer, bench_rows, log)
                metrics[sn] = m
                log.info("S3 [%s] 验证完成", sn)
            except Exception as e:  # noqa: BLE001
                log.error("S3 [%s] 验证失败（如实标记不可用）: %s",
                          sn, str(e)[:300])
                metrics[sn] = {"error": str(e)[:300]}
            finally:
                scorer.close()
                gc.collect()
    else:
        metrics["shieldgemma"] = {"acc": 0.5, "recall": 0.5, "fpr": 0.5,
                                  "fnr": 0.5, "n": n_labelled,
                                  "n_dropped": 0, "dry": True}

    overall = {
        "stage": "S3", "bench_rows": len(bench_rows),
        "n_labelled": n_labelled, "smoke": args.smoke, "dry": args.dry,
        "scorers": SCORERS, "metrics": metrics,
        "note": ("Llama-Guard-3-8B / WildGuard 均 gated 不可用；ShieldGemma-9B "
                 "为唯一离线可用异构安全分类器，见 audit_reviews.md 预登记更正。"),
    }
    with open(out_dir / "s3_shieldgemma_bench.json", "w",
              encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # ---- MD 报告 ----
    lines = [
        "# S3 异构安全分类器公开基准验证（GPU1 空闲期测量验证 · 2026-08-12）\n",
        f"- 基准: 同 gate 的 {len(bench_rows)} 行人工标注公开基准（jalmbench）",
        f"- 可标注行: {n_labelled}",
        f"- 成员: {', '.join(SCORERS)}（Llama-Guard-3-8B / WildGuard gated 不可用）",
        "  ShieldGemma 披露：基于 Gemma-2-9B 安全专用训练，与论文 Gemma-4 评分器"
        "不同训练目标，但同 Google 家族。\n",
        "| 评分器 | acc | recall | FPR | FNR | n | n_dropped |",
        "|---|---|---|---|---|---|---|",
    ]
    for sn, m in metrics.items():
        if "error" in m:
            lines.append(f"| {sn} | **失败** | — | — | — | — | — |")
            lines.append(f"  - 错误: {m['error']}")
            continue
        lines.append(f"| {sn} | {m.get('acc', '—')} | {m.get('recall', '—')} "
                     f"| {m.get('fpr', '—')} | {m.get('fnr', '—')} "
                     f"| {m.get('n', '—')} | {m.get('n_dropped', '—')} |")
    lines += [
        "\n## 与既有 cross_validation 对照（gate P0_scorers.json）",
        "> judge_big acc=0.8555, harmbench acc=0.8239, strongreject acc=0.7342,"
        " cross_check acc=0.6312（602 行）。ShieldGemma 数字即上述，直接可比，"
        "扩展现有 cross_validation 块；不修改 gates/（单权威纪律）。",
        "\n## 说明",
        "> 复用 validate_single_scorer 含分布塌缩检测：单类 >95% 即判定解析塌缩，"
        "评分器标记不可用而非静默产出——防幻觉纪律 #1。",
    ]
    (root / "report" / "s3_shieldgemma_bench.md").write_text(
        "\n".join(lines), encoding="utf-8")
    log.info("S3 done metrics=%s", json.dumps(metrics, ensure_ascii=False))
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
