#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S32：Dawid-Skene 潜在标签下的 E4B 文本主效应（CPU，2026-08-14）。

动机：S20/S20c 的 N/E_t 主效应用预注册 dual_judge 共识口径。审稿人可能问：
"若用无金标准的潜类别模型（Dawid-Skene）在所有评分器上联合推断真值标签，
主效应是否依然成立？" 本实验在 E4B 文本 3600 上，用 5 个非退化评分器
（judge_big / judge_small / qwen32 / strongreject / harmbench）拟合 DS
潜在标签（posterior>0.5 → 1），重算 N/E_t/R 主效应（query 聚类 bootstrap）。

方法学披露（关键）：
  - harmbench 对中文 FNR=1.0、strongreject 中文 acc=0.35（P0_scorers 实测），
    弱评分器可能污染 DS 潜在标签。本实验如实报告各评分器 sens/spec/err，
    并对比 DS 标签 vs dual_judge 的 N/E_t 效应方向与显著性——若 DS 更弱，
    属弱评分器污染，如实披露，不主张"DS 更强"。
  - 排除退化宽松 cross_check（S19 已证污染 DS 至不收敛）。

纪律：纯 CPU、零生成；只读 responses + scorers_cache（需 S20d strongreject/
harmbench 完成）；只写 s32_* 产物。

用法：python gpu1_s32_ds_main_effect.py [--B 1000] [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s32] %s" % m, flush=True)


def _load_rid_cache(p):
    out = {}
    if not p.exists():
        return out
    for line in p.open(encoding="utf-8"):
        rec = json.loads(line)
        out[rec["rid"]] = rec["label"]
    return out


def _effect_query_cluster(rows, labels, dim, v0, v1, B, seed):
    """query 聚类 bootstrap 的 Δ=pos(v1)-pos(v0)。labels 为 float(0/1/NaN)。"""
    import collections
    acc = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])
    for r, lab in zip(rows, labels):
        if np.isnan(lab):
            continue
        q = r["query_id"]
        a = acc[q]
        if r[dim] == v0:
            a[0] += lab
            a[1] += 1
        elif r[dim] == v1:
            a[2] += lab
            a[3] += 1
    qids = [q for q in acc if acc[q][1] >= 1 and acc[q][3] >= 1]
    if not qids:
        return None
    s0 = sum(acc[q][0] for q in qids)
    n0 = sum(acc[q][1] for q in qids)
    s1 = sum(acc[q][2] for q in qids)
    n1 = sum(acc[q][3] for q in qids)
    obs = s1 / n1 - s0 / n0
    rng = np.random.RandomState(seed)
    diffs = np.empty(B)
    for b in range(B):
        sel = [qids[rng.randint(len(qids))] for _ in qids]
        s0b = sum(acc[q][0] for q in sel)
        n0b = sum(acc[q][1] for q in sel)
        s1b = sum(acc[q][2] for q in sel)
        n1b = sum(acc[q][3] for q in sel)
        diffs[b] = (s1b / n1b - s0b / n0b) if n0b and n1b else np.nan
    ok = diffs[~np.isnan(diffs)]
    lo, hi = float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))
    return {"effect": round(float(np.mean(ok)), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "excl_zero": bool(lo > 0 or hi < 0), "n_query": len(qids),
            "n_cells": int(n0 + n1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    B = 50 if args.smoke else args.B

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "scorers_cache"

    rows = [r for r in [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl",
        encoding="utf-8")] if r.get("modality") == "text"]
    n = len(rows)
    rids = [r["response_id"] for r in rows]

    calibers = {
        "judge_big": "s17_e4b_text_judge_big.jsonl",
        "judge_small": "s17_e4b_text_judge_small.jsonl",
        "qwen32": "s17_e4b_text_qwen32.jsonl",
        "strongreject": "s17_e4b_text_strongreject.jsonl",
        "harmbench": "s17_e4b_text_harmbench.jsonl",
    }
    mats = {}
    coverage = {}
    for name, fname in calibers.items():
        cache = _load_rid_cache(cache_dir / fname)
        col = np.array([cache.get(x, np.nan) for x in rids], dtype=float)
        # label 可能为 null/None → NaN
        col[~np.isin(col, [0.0, 1.0])] = np.nan
        mats[name] = col
        coverage[name] = int(np.sum(~np.isnan(col)))
        _log("%s 覆盖 %d/%d" % (name, coverage[name], n))

    if args.smoke:
        mats = {k: v[:100] for k, v in mats.items()}
        rows = rows[:100]

    # 5 口径 DS
    from scorer_utils import dawid_skene
    mat = np.column_stack([mats[k] for k in calibers])
    ds = dawid_skene(mat, n_iter=100)
    ds_label = np.array(ds["item_label"], dtype=float)
    ds_posterior = np.array(ds["posterior"])
    _log("DS 收敛=%s，latent_pos_rate=%.4f" % (
        ds["converged"], float(np.mean(ds_label))))
    for i, name in enumerate(calibers):
        _log("  %s: sens=%.3f spec=%.3f err=%.3f" % (
            name, ds["sensitivity"][i], ds["specificity"][i],
            ds["error_rate"][i]))

    # 主效应：dual_judge（对照）vs DS 潜在标签
    cjb = _load_rid_cache(cache_dir / "s17_e4b_text_judge_big.jsonl")
    cjs = _load_rid_cache(cache_dir / "s17_e4b_text_judge_small.jsonl")
    dual = np.array([
        cjb.get(x) if (cjb.get(x) is not None and cjs.get(x) is not None
                       and cjb.get(x) == cjs.get(x)) else np.nan
        for x in rids], dtype=float)

    dims = [("N", 0, 1), ("E_t", 0, 1), ("R", 0, 1)]
    res = {"dual_judge": {}, "ds_latent": {}, "ds_posterior": {}}
    for dim, v0, v1 in dims:
        res["dual_judge"][dim] = _effect_query_cluster(
            rows, dual, dim, v0, v1, B, seed=20260814)
        res["ds_latent"][dim] = _effect_query_cluster(
            rows, ds_label, dim, v0, v1, B, seed=20260815)
        res["ds_posterior"][dim] = _effect_query_cluster(
            rows, ds_posterior[:, 1], dim, v0, v1, B, seed=20260816)
        _log("[%s] dual Δ=%s | DS Δ=%s" % (
            dim, res["dual_judge"][dim], res["ds_latent"][dim]))

    out = {
        "stage": "S32", "date": "2026-08-14",
        "purpose": "DS 潜在标签下的 E4B 文本 N/E_t/R 主效应（vs dual_judge）",
        "n_text": n, "coverage": coverage,
        "ds": {"converged": bool(ds["converged"]),
               "latent_pos_rate": float(np.mean(ds_label)),
               "sensitivity": [round(float(x), 3) for x in ds["sensitivity"]],
               "specificity": [round(float(x), 3) for x in ds["specificity"]],
               "error_rate": [round(float(x), 3) for x in ds["error_rate"]]},
        "effects": res,
        "disclosure": ("harmbench 中文 FNR=1.0、strongreject 中文 acc=0.35 "
                       "（P0_scorers 实测）——弱评分器可能污染 DS 潜在标签；"
                       "若 DS 效应弱于 dual_judge，属弱评分器污染，如实披露，"
                       "不主张 DS 更强。"),
    }
    if args.smoke:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        _log("smoke 完成（不落盘）")
        return 0
    (out_dir / "s32_ds_main_effect.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# S32：Dawid-Skene 潜在标签下的 E4B 文本主效应（CPU）\n",
        "- E4B 文本 %d 行，5 口径 DS（judge_big/judge_small/qwen32/strongreject/"
        "harmbench），exclude 退化 cross_check" % n,
        "- DS 潜在 pos_rate=%.4f，收敛=%s\n" % (
            out["ds"]["latent_pos_rate"], out["ds"]["converged"]),
        "## 评分器 sens/spec/err",
        "| 口径 | sens | spec | err |",
        "|---|---|---|---|",
    ]
    for i, name in enumerate(calibers):
        lines.append("| %s | %.3f | %.3f | %.3f |" % (
            name, ds["sensitivity"][i], ds["specificity"][i],
            ds["error_rate"][i]))
    lines.append("\n## 主效应（Δ=pos(v1)-pos(v0)，query 聚类 bootstrap）")
    lines.append("\n| 维度 | 口径 | Δ | 95%CI | 排除0 |")
    lines.append("|---|---|---|---|---|")
    for dim, v0, v1 in dims:
        for cal in ("dual_judge", "ds_latent", "ds_posterior"):
            r = res[cal][dim]
            if r is None:
                lines.append("| %s | %s | - | - | - |" % (dim, cal))
                continue
            lines.append("| %s | %s | %+.4f | [%s] | %s |" % (
                dim, cal, r["effect"], ",".join(str(x) for x in r["ci95"]),
                "✓" if r["excl_zero"] else "✗"))
    lines.append("\n## 披露\n> %s" % out["disclosure"])
    (root / "report" / "s32_ds_main_effect.md").write_text(
        "\n".join(lines), encoding="utf-8")
    _log("已落盘 s32_ds_main_effect.json + report/s32_ds_main_effect.md")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
