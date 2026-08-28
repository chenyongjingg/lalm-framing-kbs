#!/usr/bin/env python3
"""GPU1 S2：顺序路径确定性复验（2026-08-12，用户批准）。

~20 单元 × (E4B+E2B) 顺序生成两遍 → 字节一致率。
科学定位：弃 B（非逐字节等价）后，同一顺序路径在同输入下是否逐字节确定？
  - 一致率 100% → 顺序路径确定，弃 B 差异源于「批量 vs 顺序」而非顺序路径抖动。
  - 出现分歧 → 顺序路径本身不确定，属论文新增实证，须重审。
零人工标注；只写 results/gpu1_pipeline/s2_determinism.{json,md}；不写账本/done。
"""
import argparse
import gc
import json
import sys
import yaml

from gpu1_common import (load_generation_model, build_texts, infer_single_prod,
                         load_cells, stratify, release, resolve_root,
                         get_logger, sha256_hex)

MODELS = ["gemma_4_e4b", "gemma_4_e2b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--smoke", action="store_true", help="小样本冒烟（n=3）")
    ap.add_argument("--dry", action="store_true", help="仅数据层，不加载模型")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = resolve_root(cfg)
    log = get_logger("s2", root)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_new = cfg.get("gpu", {}).get("max_new_tokens", 512)
    n = 3 if args.smoke else args.n

    log.info("S2 start n=%d seed=%d dry=%s", n, args.seed, args.dry)
    cells, _ = load_cells(cfg, root, log)
    sample = stratify(cells, n, args.seed)
    log.info("S2 抽样单元=%d", len(sample))

    per_model = []
    for model_name in MODELS:
        recs = []
        if not args.dry:
            mconf = cfg["models"][model_name]
            model, tok = load_generation_model(model_name, mconf, cfg, log)
            texts = build_texts(sample, tok)
            for run in (1, 2):
                resp = [None] * len(texts)
                for i, t in enumerate(texts):
                    try:
                        resp[i] = infer_single_prod(model, tok, t, max_new)
                    except Exception as e:  # noqa: BLE001
                        log.warning("[%s] run%d idx=%d 失败: %s",
                                    model_name, run, i, str(e)[:200])
                if run == 1:
                    run1 = resp
                else:
                    run2 = resp
            release(model, tok)
        else:
            run1 = ["<dry>" + c["prompt"][:20] for c in sample]
            run2 = run1[:]

        n_eq = n_mis = n_err = 0
        for i, c in enumerate(sample):
            if run1[i] is None or run2[i] is None:
                n_err += 1
                eq = False
            else:
                eq = (run1[i] == run2[i])
                if eq:
                    n_eq += 1
                else:
                    n_mis += 1
            recs.append({
                "model": model_name,
                "query_id": c["query_id"], "lang": c["lang"],
                "condition": c["condition"], "template_idx": c["template_idx"],
                "run1_sha256": sha256_hex(run1[i]),
                "run2_sha256": sha256_hex(run2[i]),
                "byte_equal": eq, "run1_len": len(run1[i]) if run1[i] else -1,
                "run2_len": len(run2[i]) if run2[i] else -1,
            })
        per_model.append({
            "model": model_name, "n": len(recs), "n_byte_equal": n_eq,
            "n_mismatch": n_mis, "n_errors": n_err,
            "byte_equal_rate": (n_eq / len(recs) if recs else 0.0),
        })
        log.info("[%s] run1==run2: equal=%d mismatch=%d err=%d",
                 model_name, n_eq, n_mis, n_err)

    # 汇总
    all_eq = sum(m["n_byte_equal"] for m in per_model)
    all_n = sum(m["n"] for m in per_model)
    overall = {
        "stage": "S2", "n_cells": len(sample), "n_models": len(MODELS),
        "n_byte_equal_total": all_eq,
        "byte_equal_rate_total": (all_eq / all_n if all_n else 0.0),
        "per_model": per_model,
        "seed": args.seed, "dry": args.dry, "smoke": args.smoke,
        "verdict": ("顺序路径确定" if all_eq == all_n else
                    "顺序路径存在分歧——需重审"),
    }
    with open(out_dir / "s2_determinism.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    # MD 报告
    lines = [
        "# S2 顺序路径确定性复验（GPU1 空闲期测量验证 · 2026-08-12）\n",
        f"- 单元数: {len(sample)}（stratifed 抽样，seed={args.seed}）",
        f"- 模型: {', '.join(MODELS)}（顺序逐条 greedy 生成两遍）",
        f"- 字节一致率: {overall['byte_equal_rate_total']:.4f} "
        f"({all_eq}/{all_n})",
        f"- 判定: **{overall['verdict']}**\n",
        "| 模型 | 单元 | 逐字节一致 | 分歧 | 错误 | 一致率 |",
        "|---|---|---|---|---|---|",
    ]
    for m in per_model:
        lines.append(f"| {m['model']} | {m['n']} | {m['n_byte_equal']} | "
                     f"{m['n_mismatch']} | {m['n_errors']} | "
                     f"{m['byte_equal_rate']:.4f} |")
    lines += [
        "\n## 说明",
        "> 与弃 B（批量 vs 顺序非逐字节等价）正交：此处验证**同一顺序路径**两遍"
        "是否确定。100% 一致 → 顺序路径确定，弃 B 差异归因于批量路径。",
        "> 零人工标注；无金标准体系下以字节一致率佐证路径确定性。",
    ]
    (root / "report" / "s2_determinism.md").write_text(
        "\n".join(lines), encoding="utf-8")
    log.info("S2 done verdict=%s", overall["verdict"])
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
