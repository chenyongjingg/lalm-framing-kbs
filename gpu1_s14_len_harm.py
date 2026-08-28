#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S14：响应长度-有害率关系 + 生成属性交叉表（CPU，2026-08-14）。

动机（R3 协议修订）：主效应分析含 +len 协变量，以排除"篇幅→可判害"代理
效应（长响应更易被判有害的混杂）。本实验量化响应 token 长度与有害率的关系：
  - 若有害率随长度单调上升 → 长度是混杂，必须 +len 协变量（支撑 R3）；
  - 若长度与有害率无关 → 无代理效应。
同时给出生成属性（长度/停滞/refusal）跨 E_t×template 交叉表，支撑缺陷归因。

数据源（零 GPU）：
  - S4 480 响应（E4B+E2B 各 240，6 评分器 + qwen32 标签已存在）——立即。
  - E2B 全量 3600（S11 增量缓存标签 scorers_cache/{judge_big,judge_small,qwen32}.jsonl）——
    S11 完成后自动可用（本脚本重复运行幂等，缓存齐即补算 E2B 段）。
  - E4B 文本 3600（长度/停滞属性交叉表，无需标签）。

输出：results/gpu1_pipeline/s14_len_harm.{json,md} + report/。
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
    print("[s14 %s] %s" % (Path(__file__).stem, m), flush=True)


def _tok_len(text):
    """近似 token 长度：中文字符≈1 token，ASCII 词≈0.6×字符数（保守代理）。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if '一' <= ch <= '鿿')
    ascii_chars = len(text) - cjk
    return cjk + int(ascii_chars * 0.6)


def _load_s11_labels(cache_dir, rows):
    """从 S11 增量缓存重建 E2B 行级标签。返回 {i: {scorer: label}}。"""
    out = {}
    for sn in ("judge_big", "judge_small", "qwen32"):
        p = cache_dir / (sn + ".jsonl")
        if not p.exists():
            _log("  [S11 缓存] %s 缺失（S11 未完成）" % sn)
            continue
        for line in p.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                out.setdefault(rec["i"], {})[sn] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
    return out


def _load_s4_labels(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    # labels 字段含 6 评分器 + qwen32
    return rows


def _bins(values, labels):
    """按分位数分桶长度 → 有害率。"""
    v = np.array(values, dtype=float)
    lab = np.array(labels)
    mask = lab != -1  # -1 = 无有效标签
    if mask.sum() < 20:
        return None
    qs = np.quantile(v[mask], [0.2, 0.4, 0.6, 0.8])
    edges = [-np.inf] + list(qs) + [np.inf]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = mask & (v >= a) & (v < b)
        n = int(sel.sum())
        if n == 0:
            continue
        out.append({
            "len_range": [round(float(a), 1) if a > -np.inf else None,
                          round(float(b), 1) if b < np.inf else None],
            "n": n,
            "mean_len": round(float(v[sel].mean()), 1),
            "harm_rate": round(float(lab[sel].mean()), 4),
        })
    return out


def _cross_prop(rows):
    """生成属性交叉表：E_t(或 condition) × template → 长度/类别分布。"""
    keyname = "E_t" if "E_t" in rows[0] else "condition"
    tab = defaultdict(lambda: {"lens": [], "cls": defaultdict(int)})
    for r in rows:
        k = (r[keyname], r["template_idx"])
        tab[k]["lens"].append(_tok_len(r.get("response") or ""))
        tab[k]["cls"][s10b.classify(r)] += 1
    out = {}
    for (g, t), d in sorted(tab.items(), key=lambda kv: str(kv[0][0])):
        n = len(d["lens"])
        out["%s=%s/t%d" % (keyname, g, t)] = {
            "n": n,
            "mean_len": round(sum(d["lens"]) / n, 1),
            "median_len": round(float(np.median(d["lens"])), 1),
            "classes": {c: d["cls"][c] for c in
                        ("plot_stall", "data_clarify", "refusal", "other")},
        }
    return out


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

    # ---- 1. E4B 文本属性交叉表（无需标签） ----
    e4b_path = root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl"
    e4b_rows = [json.loads(l) for l in open(e4b_path, encoding="utf-8")]
    e4b_text = [r for r in e4b_rows if r["A_s"] == "text"]
    _log("E4B 文本=%d（权威 100%%）" % len(e4b_text))
    cross_e4b_text = _cross_prop(e4b_text)

    # ---- 2. S4 480（已标标签）长度-有害率 ----
    s4_path = root / "results" / "gpu1_pipeline" / "s4_convergence_full_labels.jsonl"
    s4_rows = _load_s4_labels(s4_path)
    _log("S4 响应=%d（6 评分器 + qwen32 标签已存在）" % len(s4_rows))
    s4 = {"n": len(s4_rows)}
    for tag, src in (("dual_judge", None), ("qwen32", None)):
        pass  # 占位，下重构
    # 直接构建：dual_judge 共识 + qwen32
    dj = []
    qw = []
    lens = []
    for r in s4_rows:
        lb = r.get("labels", {})
        b, s = lb.get("judge_big"), lb.get("judge_small")
        d = int(b) if (b is not None and b == s) else -1
        dj.append(d)
        q = lb.get("qwen32")
        qw.append(int(q) if q is not None else -1)
        lens.append(_tok_len(r.get("response") or ""))
    s4["len_bins_dual_judge"] = _bins(lens, np.array(dj))
    s4["len_bins_qwen32"] = _bins(lens, np.array(qw))
    # 相关性：长度 × 有害（点二列相关）
    dj_arr = np.array(dj)
    if (dj_arr != -1).sum() >= 30:
        from scipy.stats import pointbiserialr
        m = dj_arr != -1
        pbs_dj = pointbiserialr(np.array(lens)[m], dj_arr[m])
        s4["pointbiserial_len_harm_dj"] = {
            "r": round(float(pbs_dj.statistic), 4),
            "p": round(float(pbs_dj.pvalue), 6)}
    s4["cross_properties"] = _cross_prop(s4_rows)

    # ---- 3. E2B 全量 3600（S11 缓存标签，可用即补算） ----
    e2b_path = root / "responses" / "P1_PILOT" / "gemma_4_e2b_responses.jsonl"
    e2b_rows = [json.loads(l) for l in open(e2b_path, encoding="utf-8")]
    e2b = {"n": len(e2b_rows), "cache_complete": False}
    s11 = _load_s11_labels(cache_dir, e2b_rows)
    _log("S11 缓存已恢复 %d/%d 行标签" % (len(s11), len(e2b_rows)))
    e2b_lens, e2b_dj, e2b_qw = [], [], []
    n_labelled = 0
    for i, r in enumerate(e2b_rows):
        e2b_lens.append(_tok_len(r.get("response") or ""))
        rec = s11.get(i, {})
        b, s, q = rec.get("judge_big"), rec.get("judge_small"), rec.get("qwen32")
        d = int(b) if (b is not None and b == s) else -1
        e2b_dj.append(d)
        e2b_qw.append(int(q) if q is not None else -1)
        if b is not None and s is not None and q is not None:
            n_labelled += 1
    e2b["n_labelled"] = n_labelled
    # 覆盖率 >=95%（dual_judge 完整行）即计算（S11 正常有少量缺失行）
    if n_labelled >= 0.95 * len(e2b_rows):
        e2b["cache_complete"] = True
        e2b["len_bins_dual_judge"] = _bins(e2b_lens, np.array(e2b_dj))
        e2b["len_bins_qwen32"] = _bins(e2b_lens, np.array(e2b_qw))
        dja = np.array(e2b_dj)
        if (dja != -1).sum() >= 30:
            from scipy.stats import pointbiserialr
            m = dja != -1
            pbs = pointbiserialr(np.array(e2b_lens)[m], dja[m])
            e2b["pointbiserial_len_harm_dj"] = {
                "r": round(float(pbs.statistic), 4),
                "p": round(float(pbs.pvalue), 6)}
    e2b["cross_properties"] = _cross_prop(e2b_rows)
    _log("E2B 缓存完备=%s（%d/%d）" % (e2b["cache_complete"], n_labelled,
                                      len(e2b_rows)))

    overview = {
        "stage": "S14", "date": "2026-08-14",
        "purpose": ("响应长度-有害率关系（R3 +len 协变量论证）；"
                    "生成属性跨 E_t×template 交叉表"),
        "len_metric": "近似 token（中文=1，ASCII=0.6×字符）",
        "s4_480": s4,
        "e2b_3600": e2b,
        "e4b_text_3600_cross": cross_e4b_text,
        "conclusion": _conclude(s4, e2b),
    }

    with open(out_dir / "s14_len_harm.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)
    md = render_md(overview)
    (out_dir / "s14_len_harm.md").write_text(md, encoding="utf-8")
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "s14_len_harm.md").write_text(md, encoding="utf-8")
    _log("已落盘 s14_len_harm.json/.md")

    print(json.dumps({"stage": "S14", "s4_pointbiserial_dj":
                      s4.get("pointbiserial_len_harm_dj"),
                      "e2b_cache_complete": e2b["cache_complete"],
                      "e2b_pointbiserial_dj":
                      e2b.get("pointbiserial_len_harm_dj"),
                      "conclusion": overview["conclusion"]},
                     ensure_ascii=False, indent=2))
    return 0


def _conclude(s4, e2b):
    notes = []
    for tag, d in (("S4", s4), ("E2B", e2b)):
        pbs = d.get("pointbiserial_len_harm_dj")
        if pbs:
            notes.append({
                "source": tag, "pointbiserial_r": pbs["r"], "p": pbs["p"],
                "read": ("长度与有害率显著相关（r=%.2f, p=%.4g）→ 长度是混杂，"
                         "主效应须 +len 协变量（R3）" % (pbs["r"], pbs["p"])
                         if pbs["p"] < 0.05 else
                         "长度与有害率无显著相关 → 无篇幅代理效应")})
    return notes


def render_md(o):
    lines = [
        "# S14：响应长度-有害率关系 + 生成属性交叉表（GPU1 · 2026-08-14）\n",
        "## 目的",
        "主效应分析含 +len 协变量（R3），排除'篇幅→可判害'代理效应。本实验量化"
        "长度与有害率关系：显著相关 → 必须 +len；无关 → 无代理效应。\n",
        "## S4 480 响应（6 评分器 + qwen32 标签已存在）",
    ]
    b = o["s4_480"]
    lines.append("- 点二列相关(长度 × dual_judge 有害): %s" % (
        json.dumps(b.get("pointbiserial_len_harm_dj"), ensure_ascii=False)))
    for src_name, src in (("S4", b), ("E2B 全量", o["e2b_3600"])):
        lines.append("\n## %s 长度分桶 → 有害率" % src_name)
        if src.get("cache_complete"):
            bb = src.get("len_bins_dual_judge")
            if bb:
                lines.append("| 长度段 | n | 均长 | dual_judge 有害率 |")
                lines.append("|---|---|---|---|")
                for x in bb:
                    lo, hi = x["len_range"]
                    lines.append("| %s-%s | %d | %.0f | %.4f |" % (
                        lo if lo is not None else "~",
                        hi if hi is not None else "~", x["n"], x["mean_len"],
                        x["harm_rate"]))
            else:
                lines.append("（无有效 dual_judge 标签）")
        else:
            lines.append("（S11 缓存未齐：%d/%d，S11 完成后重跑补算）" % (
                src.get("n_labelled", 0), src["n"]))
    lines.append("\n## E4B 文本生成属性交叉表（E_t × template，权威 3600）")
    lines.append("| 组 | n | 均长 | 中位长 | plot_stall | data_clarify | refusal | other |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k, d in o["e4b_text_3600_cross"].items():
        c = d["classes"]
        lines.append("| %s | %d | %.0f | %.0f | %d | %d | %d | %d |" % (
            k, d["n"], d["mean_len"], d["median_len"], c["plot_stall"],
            c["data_clarify"], c["refusal"], c["other"]))
    lines.append("\n## 结论")
    for n in o["conclusion"]:
        lines.append("- [%s] %s" % (n["source"], n["read"]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
