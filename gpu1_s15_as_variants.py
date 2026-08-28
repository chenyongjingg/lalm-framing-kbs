#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S15：A_s acoustic 变体（neutral_audio vs styled_audio）对比（CPU，2026-08-14）。

动机：A_s（声学风格）是研究设计主因子之一。E4B 音频每个 (query,template,N,R)
有 neutral_audio 与 styled_audio 两个变体。本实验对比两变体的响应属性差异，
量化 A_s 是否构成系统主效应（长度/停滞/refusal/有害代理），并核对配对结构
（同 key 是否两变体齐备）。

数据：E4B 音频当前快照（neutral 1963 + styled 1962，仍在生成 → 标注快照态，
E4B 完成后重跑获权威）。零 GPU。

输出：results/gpu1_pipeline/s15_as_variants.{json,md} + report/。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu1_s10b as s10b  # noqa: E402


def _log(m):
    print("[s15 %s] %s" % (Path(__file__).stem, m), flush=True)


def _tok_len(text):
    if not text:
        return 0
    cjk = sum(1 for ch in text if '一' <= ch <= '鿿')
    return cjk + int((len(text) - cjk) * 0.6)


def _cls_counts(rows):
    c = defaultdict(int)
    for r in rows:
        c[s10b.classify(r)] += 1
    return {k: c[k] for k in ("plot_stall", "data_clarify", "refusal", "other")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    e4b_path = root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl"
    rows = [json.loads(l) for l in open(e4b_path, encoding="utf-8")]
    neutral = [r for r in rows if r["A_s"] == "neutral_audio"]
    styled = [r for r in rows if r["A_s"] == "styled_audio"]
    _log("neutral=%d styled=%d（音频仍在生成，快照）" % (len(neutral),
                                                      len(styled)))

    def stats(rs):
        lens = [_tok_len(r.get("response") or "") for r in rs]
        return {
            "n": len(rs),
            "mean_len": round(sum(lens) / len(lens), 1) if lens else None,
            "median_len": round(float(np.median(lens)), 1) if lens else None,
            "p90_len": round(float(np.percentile(lens, 90)), 1) if lens else None,
            "classes": _cls_counts(rs),
        }

    # 配对完整性：key = (query_id, template_idx, N, R)，两变体是否齐备
    nk = defaultdict(lambda: {"neutral": 0, "styled": 0})
    for r in neutral:
        nk[(r["query_id"], r["template_idx"], r["N"], r["R"])]["neutral"] += 1
    for r in styled:
        nk[(r["query_id"], r["template_idx"], r["N"], r["R"])]["styled"] += 1
    paired_keys = sum(1 for v in nk.values() if v["neutral"] and v["styled"])
    neutral_only = sum(1 for v in nk.values() if v["neutral"] and not v["styled"])
    styled_only = sum(1 for v in nk.values() if v["styled"] and not v["neutral"])

    # 按 E_t 分层属性
    by_et = {}
    for et in (0, 1):
        n_et = [r for r in neutral if r["E_t"] == et]
        s_et = [r for r in styled if r["E_t"] == et]
        by_et["E_t=%d" % et] = {"neutral": stats(n_et), "styled": stats(s_et)}

    # 配对样本 t 检验：同 key 两变体长度差（Wilcoxon 符号秩）
    pair_lens = []
    for k, v in nk.items():
        if v["neutral"] and v["styled"]:
            rn = next(r for r in neutral
                      if (r["query_id"], r["template_idx"], r["N"], r["R"]) == k)
            rs = next(r for r in styled
                      if (r["query_id"], r["template_idx"], r["N"], r["R"]) == k)
            pair_lens.append((_tok_len(rn.get("response") or ""),
                              _tok_len(rs.get("response") or "")))
    wilcoxon = None
    if len(pair_lens) >= 20:
        from scipy.stats import wilcoxon
        diffs = [a - b for a, b in pair_lens]
        try:
            w = wilcoxon(diffs)
            wilcoxon = {"n_pairs": len(pair_lens),
                        "mean_diff_neutral_minus_styled": round(
                            float(np.mean(diffs)), 2),
                        "p_value": round(float(w.pvalue), 6)}
        except Exception as e:  # noqa: BLE001
            wilcoxon = {"error": str(e)[:120], "n_pairs": len(pair_lens)}

    overview = {
        "stage": "S15", "date": "2026-08-14",
        "data_note": "音频仍在生成，本表为当前快照（E4B 完成后重跑获权威）",
        "n_neutral": len(neutral), "n_styled": len(styled),
        "pairing": {"n_keys": len(nk), "paired": paired_keys,
                    "neutral_only": neutral_only, "styled_only": styled_only},
        "by_et": by_et,
        "paired_len_wilcoxon": wilcoxon,
        "conclusion": ("A_s 两变体属性差异如上；若配对长度/类别显著差异则 A_s "
                       "为系统主效应，主效应模型须含 A_s（设计已含）。"),
    }

    with open(out_dir / "s15_as_variants.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)
    md = render_md(overview)
    (out_dir / "s15_as_variants.md").write_text(md, encoding="utf-8")
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "s15_as_variants.md").write_text(md, encoding="utf-8")
    _log("已落盘 s15_as_variants.json/.md")

    print(json.dumps({"stage": "S15", "n_neutral": len(neutral),
                      "n_styled": len(styled), "pairing": overview["pairing"],
                      "paired_len_wilcoxon": wilcoxon,
                      "by_et": by_et}, ensure_ascii=False, indent=2))
    return 0


def render_md(o):
    lines = [
        "# S15：A_s acoustic 变体（neutral vs styled）对比（GPU1 · 2026-08-14）\n",
        "## 数据",
        "- neutral_audio=%d, styled_audio=%d（%s）" % (
            o["n_neutral"], o["n_styled"], o["data_note"]),
        "- 配对结构：key（query,template,N,R）%d 个，双变体齐备 %d，仅 neutral %d，"
        "仅 styled %d" % (o["pairing"]["n_keys"], o["pairing"]["paired"],
                          o["pairing"]["neutral_only"],
                          o["pairing"]["styled_only"]),
        "\n## 按 E_t 分层属性",
        "| E_t | 变体 | n | 均长 | 中位长 | p90 | plot_stall | data_clarify | refusal | other |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for et, d in o["by_et"].items():
        for vn in ("neutral", "styled"):
            s = d[vn]
            c = s["classes"]
            lines.append("| %s | %s | %d | %.0f | %.0f | %.0f | %d | %d | %d | %d |" % (
                et, vn, s["n"], s["mean_len"] or 0, s["median_len"] or 0,
                s["p90_len"] or 0, c["plot_stall"], c["data_clarify"],
                c["refusal"], c["other"]))
    w = o["paired_len_wilcoxon"]
    lines.append("\n## 配对长度检验（同 key neutral−styled）")
    if w:
        lines.append("- n_pairs=%d, mean_diff=%.2f, p=%.6f" % (
            w.get("n_pairs", 0), w.get("mean_diff_neutral_minus_styled", 0),
            w.get("p_value", 0)))
        if w.get("p_value", 1) < 0.05:
            lines.append("> 配对长度显著差异 → A_s 构成系统长度差异")
    lines.append("\n## 判读")
    lines.append("> %s" % o["conclusion"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
