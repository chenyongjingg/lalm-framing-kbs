#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S27：模态效应在官方基准评分器下的稳健性（CPU · 2026-08-14/15）。

动机：S24 确立 audio>>text 模态主效应（dual Δ≈+0.14，qwen32 Δ≈+0.21），但该结论
依赖 judge 类口径。审稿人必问："若用官方基准（StrongREJECT / HarmBench）二分类标签，
音频放大是否依然成立？" 本实验用 S20d（E4B 文本）+ S25（E4B 音频）产出的
strongreject / harmbench 两条官方腿，重算模态主效应，检验稳健性边界。

方法（与 S24 同构，口径对齐）：
  - E4B 全量 responses（E4B 完成后运行，权威值）。
  - 每官方评分器：text / neutral_audio / styled_audio 单模态有害率，
    两两模态效应（query 聚类 bootstrap 95%CI + Fisher OR/p），
    以及 text vs all_audio 合并对比。
  - 覆盖率如实披露（E4B 全量下 strongreject/harmbench 双模态覆盖）。

纪律：纯 CPU、零生成；只读 responses + s17_e4b_text/audio_{strongreject,harmbench}
缓存；只写 s27_* 产物。权威运行（E4B 完成 + S20d 文本 / S25 音频官方腿全覆盖后）。

用法：python gpu1_s27_modality_official.py [--B 1000] [--smoke]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s27 %s] %s" % (Path(__file__).stem, m), flush=True)


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _comb_effect(rows, labels, is0, is1, B=1000, seed=20260815, name=""):
    """text vs all_audio 合并对比（A_s 精确匹配无法覆盖多水平，故独立实现）。"""
    from gpu1_s24_modality_effect import _fisher
    by_q = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_q[r["query_id"]].append(i)
    qids = sorted(by_q)
    rng = np.random.RandomState(seed)
    nq = len(qids)
    g0 = [labels[i] for i, r in enumerate(rows)
          if is0(r) and not np.isnan(labels[i])]
    g1 = [labels[i] for i, r in enumerate(rows)
          if is1(r) and not np.isnan(labels[i])]
    p0 = float(np.mean(g0)) if g0 else None
    p1 = float(np.mean(g1)) if g1 else None
    deltas = np.empty(B)
    for b in range(B):
        sel = []
        for _ in range(nq):
            sel.extend(by_q[qids[rng.randint(nq)]])
        s0 = [labels[i] for i in sel if is0(rows[i]) and not np.isnan(labels[i])]
        s1 = [labels[i] for i in sel if is1(rows[i]) and not np.isnan(labels[i])]
        if not s0 or not s1:
            deltas[b] = np.nan
            continue
        deltas[b] = np.mean(s1) - np.mean(s0)
    ok = deltas[~np.isnan(deltas)]
    ci = ([round(float(np.percentile(ok, 2.5)), 4),
           round(float(np.percentile(ok, 97.5)), 4)]
          if len(ok) >= B // 2 else None)
    h1 = int(sum(g1)) if g1 else 0
    h0 = int(sum(g0)) if g0 else 0
    odds, p = _fisher(h1, len(g1) - h1, h0, len(g0) - h0)
    return {"n0": len(g0), "n1": len(g1),
            "pos0": round(p0, 4) if p0 is not None else None,
            "pos1": round(p1, 4) if p1 is not None else None,
            "delta": round(p1 - p0, 4) if (p0 is not None and p1 is not None) else None,
            "ci95": ci, "or_fisher": odds, "p_fisher": p, "name": name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    B = 20 if args.smoke else args.B

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "scorers_cache"
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    from gpu1_s24_modality_effect import _read_rid_cache, _modality_effect
    rows = [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl",
        encoding="utf-8")]
    rid = [r["response_id"] for r in rows]

    audio_mask = np.array([str(r.get("A_s", "")).endswith("audio") for r in rows])
    text_mask = ~audio_mask
    n_text = int(np.sum(text_mask))
    n_aud = int(np.sum(audio_mask))
    _log("E4B 总=%d text=%d audio=%d（权威全量）" % (len(rows), n_text, n_aud))

    def _lbl(fn):
        d = _read_rid_cache(cache_dir / fn)
        return np.array([d.get(x, np.nan) for x in rid], dtype=float)

    pairs = [("text", "neutral_audio"), ("text", "styled_audio"),
             ("neutral_audio", "styled_audio")]

    out = {
        "stage": "S27",
        "date": "2026-08-15",
        "purpose": "模态效应在官方基准评分器（strongreject / harmbench）下的稳健性",
        "authoritative": "E4B 完成后运行（E4B 全量 + S20d 文本 + S25 音频官方腿全覆盖）",
        "design": "与 S24 同构：query 聚类 bootstrap 95%CI；两两模态 + text vs all_audio",
    }

    rates, effects, coverage = {}, {}, {}
    for sn in ["strongreject", "harmbench"]:
        txt = _lbl("s17_e4b_text_%s.jsonl" % sn)
        aud = _lbl("s17_e4b_audio_%s.jsonl" % sn)
        merged = np.where(audio_mask, aud, txt)
        n_txt = int(np.sum(~np.isnan(txt) & text_mask))
        n_aud_cov = int(np.sum(~np.isnan(aud) & audio_mask))
        coverage[sn] = {
            "text_n": n_txt, "audio_n": n_aud_cov,
            "text_coverage": round(n_txt / n_text, 4) if n_text else None,
            "audio_coverage": round(n_aud_cov / n_aud, 4) if n_aud else None,
        }
        _log("[%s] 覆盖 text %d/%d audio %d/%d" % (sn, n_txt, n_text,
                                                   n_aud_cov, n_aud))
        r = {}
        for lvl in ["text", "neutral_audio", "styled_audio"]:
            v = [merged[i] for i, rr in enumerate(rows)
                 if rr.get("A_s") == lvl and not np.isnan(merged[i])]
            r[lvl] = {"n": len(v), "pos": round(float(np.mean(v)), 4) if v else None}
        rates[sn] = r
        e = {}
        for a0, a1 in pairs:
            e["%s_vs_%s" % (a0, a1)] = _modality_effect(
                rows, merged, a0, a1, B=B, name="%s_%s_vs_%s" % (sn, a0, a1))
        e["text_vs_all_audio"] = _comb_effect(
            rows, merged,
            lambda r: r.get("A_s") == "text",
            lambda r: str(r.get("A_s", "")).endswith("audio"),
            B=B, name="%s_text_vs_all_audio" % sn)
        effects[sn] = e
        _log("[%s] text_vs_styled Δ=%s CI=%s; text_vs_all Δ=%s" % (
            sn, e["text_vs_styled_audio"]["delta"],
            e["text_vs_styled_audio"]["ci95"],
            e["text_vs_all_audio"]["delta"]))
    out["coverage"] = coverage
    out["modality_rates"] = rates
    out["modality_effects"] = effects

    if args.smoke:
        print(json.dumps(_json_safe(out), ensure_ascii=False, indent=2))
        _log("smoke 完成（不落盘）")
        return 0

    with open(out_dir / "s27_modality_official.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(out), f, ensure_ascii=False, indent=2)
    _render_md(out, report_dir / "s27_modality_official.md")
    _log("已落盘 s27_modality_official.json + report/s27_modality_official.md")
    return 0


def _render_md(o, path):
    L = ["# S27：模态效应 × 官方基准评分器稳健性（GPU1 · 2026-08-15）\n",
         "\n**状态**：权威（E4B 全量 + S20d/S25 官方腿全覆盖）｜ 口径：strongreject / harmbench 二分类\n",
         "\n## 1. 单模态有害率（官方评分器）\n",
         "| 评分器 | text | neutral_audio | styled_audio |\n",
         "|---|---|---|---|\n"]
    for sn, r in o["modality_rates"].items():
        L.append("| %s | %s | %s | %s |\n" % (
            sn, r["text"]["pos"], r["neutral_audio"]["pos"],
            r["styled_audio"]["pos"]))
    L.append("\n## 2. 模态效应（Δ = pos(au) − pos(text)，query 聚类 95%CI）\n")
    L.append("| 评分器 | 对比 | n0 | n1 | pos0 | pos1 | Δ | 95%CI | Fisher OR(p) |\n")
    L.append("|---|---|---|---|---|---|---|---|---|\n")
    for sn, e in o["modality_effects"].items():
        for cmpname, me in e.items():
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" % (
                sn, cmpname, me["n0"], me["n1"], me["pos0"], me["pos1"],
                me["delta"], me["ci95"],
                ("%s(%s)" % (me["or_fisher"], me["p_fisher"]))
                if me["or_fisher"] is not None else "-"))
    L.append("\n## 3. 覆盖率\n")
    L.append("| 评分器 | text n | text cov | audio n | audio cov |\n")
    L.append("|---|---|---|---|---|\n")
    for sn, c in o["coverage"].items():
        L.append("| %s | %s | %s | %s | %s |\n" % (
            sn, c["text_n"], c["text_coverage"], c["audio_n"],
            c["audio_coverage"]))
    L.append("\n## 判读\n")
    L.append("> S24 模态效应（dual/qwen32/judge）在官方基准评分器下是否稳健（符号、量级、"
             "显著性）；若 strongreject/harmbench 亦显示 audio>>text，则为多模态放大声明"
             "提供跨评分器证据。\n")
    path.write_text("".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
