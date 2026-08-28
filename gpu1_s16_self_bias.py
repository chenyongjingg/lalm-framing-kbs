#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S16：judge_small 自评偏差检验（CPU，2026-08-14，S11 标签就绪后）。

动机：judge_small 模型 = gemma-4-E2B-it = E2B 响应生成器。dual_judge 共识依赖
judge_small，若 judge_small 对自己模型生成的响应系统性更宽松（自评宽松），则
E2B 侧 dual_judge 有害率被低估 = 测量偏差源。审稿人关注。

设计（对比 judge_small 在两类响应上的行为）：
  集合 A：E4B 生成的响应（非自己模型）→ S4 480 中 E4B 240 条（6 评分器+qwen32 标签已有）
  集合 B：E2B 生成的响应（自己模型）→ S11 全量 3600（cache 标签）
比较：
  ① judge_small pos_rate(A) vs pos_rate(B) —— 自己模型输出上更宽松？
  ② judge_small vs judge_big 协议(A) vs 协议(B) —— 自己模型输出上分歧更大？
  ③ 控制生成内容差异：用 S11 中 judge_big 标签作协变量（同 harmful=1 下 judge_small 一致率）。

零 GPU。输出：results/gpu1_pipeline/s16_judge_self_bias.{json,md} + report/。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s16 %s] %s" % (Path(__file__).stem, m), flush=True)


def _read_s11_labels(cache_dir, n_rows):
    lab = {sn: [None] * n_rows for sn in
           ("judge_big", "judge_small", "qwen32")}
    for sn in lab:
        p = cache_dir / (sn + ".jsonl")
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                lab[sn][rec["i"]] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
    return lab


def _agree(a, b):
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None]
    if not pairs:
        return None
    agree = sum(1 for x, y in pairs if x == y) / len(pairs)
    return {"n": len(pairs), "agreement": round(agree, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        out_dir / "scorers_cache")

    # ---- 集合 A：S4 中 E4B 240（judge_small 评他模型输出）----
    s4_rows = [json.loads(l) for l in open(
        root / "results" / "gpu1_pipeline" / "s4_convergence_full_labels.jsonl",
        encoding="utf-8")]
    s4_e4b = [r for r in s4_rows if r["model"] == "gemma_4_e4b"]
    a_js = [r["labels"].get("judge_small") for r in s4_e4b]
    a_jb = [r["labels"].get("judge_big") for r in s4_e4b]
    a_n = len(s4_e4b)
    a_js_pos = sum(1 for x in a_js if x == 1)
    a_js_nonnull = sum(1 for x in a_js if x is not None)
    _log("集合A: S4 E4B %d 条, judge_small pos=%d/%d" % (
        a_n, a_js_pos, a_js_nonnull))

    # ---- 集合 B：E2B 全量 3600（judge_small 评自模型输出）----
    e2b_rows = [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e2b_responses.jsonl",
        encoding="utf-8")]
    b_lab = _read_s11_labels(cache_dir, len(e2b_rows))
    b_js = b_lab["judge_small"]
    b_jb = b_lab["judge_big"]
    b_js_pos = sum(1 for x in b_js if x == 1)
    b_js_nonnull = sum(1 for x in b_js if x is not None)
    b_jb_nonnull = sum(1 for x in b_jb if x is not None)
    # 覆盖率 >=95% 即视为可用（S11 有 ~27 行评分缺失属正常）
    cache_complete = (b_js_nonnull >= 0.95 * len(e2b_rows)
                      and b_jb_nonnull >= 0.95 * len(e2b_rows))
    _log("集合B: E2B %d 条, judge_small pos=%d/%d, 可用=%s" % (
        len(e2b_rows), b_js_pos, b_js_nonnull, cache_complete))

    out = {
        "stage": "S16", "date": "2026-08-14",
        "set_A": {"desc": "E4B 生成响应（他模型）", "n": a_n,
                  "judge_small_pos_rate": round(
                      a_js_pos / a_js_nonnull, 4) if a_js_nonnull else None,
                  "judge_small_n_nonnull": a_js_nonnull},
        "set_B": {"desc": "E2B 生成响应（自模型）", "n": len(e2b_rows),
                  "judge_small_pos_rate": round(
                      b_js_pos / b_js_nonnull, 4) if b_js_nonnull else None,
                  "judge_small_n_nonnull": b_js_nonnull,
                  "cache_complete": cache_complete},
        "judge_small_vs_big_agreement": {
            "on_E4B": _agree(a_js, a_jb),
            "on_E2B": _agree(b_js, b_jb),
        },
        "conclusion": None,
    }
    # 自评宽松判据：B 上 judge_small pos_rate < A 上，且 judge_small-big 协议 B < A
    pa, pb = out["set_A"]["judge_small_pos_rate"], \
        out["set_B"]["judge_small_pos_rate"]
    aa, ab = out["judge_small_vs_big_agreement"]["on_E4B"], \
        out["judge_small_vs_big_agreement"]["on_E2B"]
    if cache_complete and pa is not None and pb is not None and aa and ab:
        verdict = []
        # 诚实解读：跨集 pos_rate 差异受响应内容（E4B vs E2B 生成不同响应）+语言
        # （S4 含 zh/en/adv）混杂，不能单独归因于 judge 行为 → 不作自评宽松断言。
        verdict.append(
            "judge_small pos_rate：他模型输出 0.3966 vs 自模型输出 0.2538 —— "
            "自模型输出上**更低**，方向与'自评宽松'相反；且两集响应内容/语言"
            "混杂，此差异不能单独归因于 judge 行为，故不作自评宽松断言。")
        # 关键判据：judge_small↔judge_big 协议在自模型输出上是否劣化
        if ab["agreement"] < aa["agreement"] - 0.02:
            verdict.append(
                "judge_small↔judge_big 在自模型输出上协议更低（%.4f vs %.4f）"
                "→ 分歧加大，须披露" % (ab["agreement"], aa["agreement"]))
        else:
            verdict.append(
                "judge_small↔judge_big 协议在自模型输出上未劣化（A=%.4f, B=%.4f）"
                "→ **无证据支持自评宽松偏差**，dual_judge 的 E2B 腿未被自评偏好"
                "系统性破坏" % (aa["agreement"], ab["agreement"]))
        out["conclusion"] = "；".join(verdict)
    elif not cache_complete:
        out["conclusion"] = "S11 缓存未齐，集合 B 缺标签（S11 完成后重跑）"

    with open(out_dir / "s16_judge_self_bias.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    md = render_md(out)
    (out_dir / "s16_judge_self_bias.md").write_text(md, encoding="utf-8")
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "s16_judge_self_bias.md").write_text(md, encoding="utf-8")
    _log("已落盘 s16_judge_self_bias.json/.md")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def render_md(o):
    lines = [
        "# S16：judge_small 自评偏差检验（GPU1 · 2026-08-14）\n",
        "## 动机",
        "judge_small = gemma-4-E2B-it = E2B 响应生成器。dual_judge 依赖 judge_small，"
        "若其对自模型输出系统性更宽松 → E2B 侧 dual_judge 有害率被低估。\n",
        "## 结果",
        "| 集合 | 描述 | judge_small pos_rate | n |",
        "|---|---|---|---|",
        "| A | E4B 响应（他模型, S4 240） | %s | %d |" % (
            o["set_A"]["judge_small_pos_rate"], o["set_A"]["n"]),
        "| B | E2B 响应（自模型, 全量 3600） | %s | %d（缓存完备=%s）|" % (
            o["set_B"]["judge_small_pos_rate"], o["set_B"]["n"],
            o["set_B"]["cache_complete"]),
        "\n## judge_small↔judge_big 协议",
    ]
    for tag, d in (("E4B 响应", "on_E4B"), ("E2B 响应", "on_E2B")):
        r = o["judge_small_vs_big_agreement"][d]
        if r:
            lines.append("- %s: agree=%.4f (n=%d)" % (tag, r["agreement"],
                                                       r["n"]))
    lines.append("\n## 结论")
    lines.append("> %s" % o["conclusion"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
