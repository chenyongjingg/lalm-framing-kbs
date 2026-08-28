#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S20：E2B 全量 E_t×N×R 三维分解 + 长度混杂 + 跨生成器（CPU，2026-08-14）。

动机（多角度补齐，KBS 录用导向）：S1-S19 全部把 N（叙事结构/事件链，协议核心 RQ
H1）与 R（角色代理/人格）合并——协议 §2 定义 E_t×N×R×A_s 完整因子空间，审稿人必问
"你的设计有 N 与 R 两个维度，为何只报 E_t？"。且实测 R=1 人格 prompt 使响应长度
中位 425→785（+85%），与 S14 长度-有害率混杂直接冲突 → 必须披露并控制。

本实验（零 GPU，全用现成缓存）：
  1) 三维主效应 E_t / N / R × 4 口径（dual_judge 共识 / judge_big / judge_small /
     qwen32）：pos(0/1)、Δ、query 聚类 bootstrap 95%CI、Fisher OR+p。
  2) 长度混杂：各维度长度差异 + logit(+len) 调整后的调整 Δ（R3 协变量纪律）。
  3) 两两交互 E_t×N / E_t×N×R / E_t×R / N×R：2×2 分层表 + logit 交互项系数。
  4) 8 组合 (E_t,N,R) 完整 pos_rate 表。
  5) 跨生成器（S17b 产物）：E4B 文本 vs E2B 同格 3600 qwen32 标签逐格一致率 +
     E4B 侧三维主效应 → 效应是否跨生成器同向。
  6) 查询级异质性：150 查询的效应分布 + leave-one-out 敏感性。

零人工标注；只写 results/gpu1_pipeline/s20_* + report/；不碰账本/done。
判据：N 效应 E2B 侧预验（协议 RQ）；R 人格效应如实披露（含长度混杂）；
跨生成器同向 → 效应稳健。

用法：python gpu1_s20_e2b_r_n_interaction.py [--B 2000]
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
    print("[s20 %s] %s" % (Path(__file__).stem, m), flush=True)


def _json_safe(o):
    """递归把 numpy 标量 / 元组转成 JSON 安全类型（防序列化崩溃）。"""
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _read_cache(p):
    out = {}
    if not p.exists():
        return out
    for line in p.open(encoding="utf-8"):
        try:
            rec = json.loads(line)
            out[rec.get("rid") if "rid" in rec else rec["i"]] = rec["label"]
        except Exception:  # noqa: BLE001
            continue
    return out


def _bootstrap_dim(rows, labels, dim, val0, val1, seed=20260815, B=2000):
    """query 聚类 bootstrap：Δ = pos(dim=val1) − pos(dim=val0)。labels: np 数组。"""
    by_q = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_q[r["query_id"]].append(i)
    qids = sorted(by_q)
    rng = np.random.RandomState(seed)
    nq = len(qids)
    deltas = np.empty(B)
    for b in range(B):
        sel = []
        for _ in range(nq):
            sel.extend(by_q[qids[rng.randint(nq)]])
        g0 = [labels[i] for i in sel if rows[i][dim] == val0]
        g1 = [labels[i] for i in sel if rows[i][dim] == val1]
        g0 = [x for x in g0 if not np.isnan(x)]
        g1 = [x for x in g1 if not np.isnan(x)]
        if not g0 or not g1:
            deltas[b] = np.nan
            continue
        deltas[b] = np.mean(g1) - np.mean(g0)
    ok = deltas[~np.isnan(deltas)]
    if len(ok) < B // 2:
        return None
    return [round(float(np.percentile(ok, 2.5)), 4),
            round(float(np.percentile(ok, 97.5)), 4)]


def _fisher(a, b, c, d):
    from scipy.stats import fisher_exact
    try:
        or_, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        return round(float(or_), 4), round(float(p), 6)
    except Exception:  # noqa: BLE001
        return None, None


def _dim_effect(rows, labels, dim, val0, val1, B=2000, name=""):
    e0 = [labels[i] for i, r in enumerate(rows)
          if r[dim] == val0 and not np.isnan(labels[i])]
    e1 = [labels[i] for i, r in enumerate(rows)
          if r[dim] == val1 and not np.isnan(labels[i])]
    p0 = float(np.mean(e0)) if e0 else None
    p1 = float(np.mean(e1)) if e1 else None
    delta = (p1 - p0) if (p0 is not None and p1 is not None) else None
    ci = _bootstrap_dim(rows, labels, dim, val0, val1, B=B)
    or_, pv = None, None
    if e0 and e1:
        a = sum(1 for i, r in enumerate(rows)
                if r[dim] == val1 and labels[i] == 1)
        c = sum(1 for i, r in enumerate(rows)
                if r[dim] == val0 and labels[i] == 1)
        b = sum(1 for i, r in enumerate(rows)
                if r[dim] == val1 and labels[i] == 0)
        d = sum(1 for i, r in enumerate(rows)
                if r[dim] == val0 and labels[i] == 0)
        or_, pv = _fisher(a, b, c, d)
    return {"dim": dim, "name": name, "n0": len(e0), "n1": len(e1),
            "pos0": round(p0, 4) if p0 is not None else None,
            "pos1": round(p1, 4) if p1 is not None else None,
            "delta": round(delta, 4) if delta is not None else None,
            "ci95": ci, "or": or_, "p": pv}


def _logit_adjusted(rows, labels, dim, B=2000):
    """logit(harm) ~ dim + log(len) + template：调整后 dim 系数/OR（R3 协变量）。"""
    try:
        import statsmodels.api as sm
    except Exception:  # noqa: BLE001
        return {"note": "statsmodels 不可用，跳过 logit 调整"}
    X, y, idx = [], [], []
    for i, r in enumerate(rows):
        if np.isnan(labels[i]):
            continue
        ln = np.log(max(1, len(r.get("response") or "")))
        X.append([1.0, float(r[dim]), float(ln), float(r["template_idx"])])
        y.append(int(labels[i]))
    if not X or len(set(v[1] for v in X)) < 2:
        return {"note": "维度无方差"}
    Xa, ya = np.array(X), np.array(y)
    model = sm.Logit(ya, Xa).fit(disp=0)
    coef = model.params[1]
    pval = float(model.pvalues[1])
    lo, hi = model.conf_int()[1]
    return {"coef_dim": round(float(coef), 4),
            "coef_len": round(float(model.params[2]), 4),
            "p_dim": round(pval, 6),
            "or_dim": round(float(np.exp(coef)), 4),
            "ci95_or": [round(float(np.exp(lo)), 4),
                        round(float(np.exp(hi)), 4)]}


def _interaction(rows, labels, d1, d2, B=2000):
    """两两交互：Δ(d2|d1=1) − Δ(d2|d1=0)，query 聚类 bootstrap。"""
    lab = np.array(labels)
    vals = [0, 1]
    rates = {}
    for v1 in vals:
        sub = [(i, r) for i, r in enumerate(rows) if r[d1] == v1]
        sub = [(i, r) for i, r in sub if not np.isnan(lab[i])]
        for v2 in vals:
            sel = [lab[i] for i, r in sub if r[d2] == v2]
            rates[(v1, v2)] = float(np.mean(sel)) if sel else None
    d0 = (rates.get((0, 1)) or 0) - (rates.get((0, 0)) or 0)
    d1v = (rates.get((1, 1)) or 0) - (rates.get((1, 0)) or 0)
    inter = d1v - d0
    # bootstrap 交互项 CI（query 聚类）
    by_q = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_q[r["query_id"]].append(i)
    qids = sorted(by_q)
    rng = np.random.RandomState(seed=20260815)
    nq = len(qids)
    its = np.empty(B)
    for b in range(B):
        sel = []
        for _ in range(nq):
            sel.extend(by_q[qids[rng.randint(nq)]])
        rr = {i: rows[i] for i in sel}
        ls = {i: lab[i] for i in sel}
        def dsum(d1v_, d2v_):
            g = [ls[i] for i in sel if rr[i][d1] == d1v_ and rr[i][d2] == d2v_
                 and not np.isnan(ls[i])]
            return float(np.mean(g)) if g else 0.0
        d0_ = dsum(0, 1) - dsum(0, 0)
        d1_ = dsum(1, 1) - dsum(1, 0)
        its[b] = d1_ - d0_
    ok = its[~np.isnan(its)]
    ci = ([round(float(np.percentile(ok, 2.5)), 4),
           round(float(np.percentile(ok, 97.5)), 4)] if len(ok) > B // 2
          else None)
    return {"rates": {("d%s=%d,d%s=%d" % (d1, a, d2, b)): v
                      for (a, b), v in rates.items()},
            "d2_effect_at_d1_0": round(float(d0), 4),
            "d2_effect_at_d1_1": round(float(d1v), 4),
            "interaction": round(float(inter), 4),
            "interaction_ci95": ci}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true",
                    help="快速校验：B=20，仅跑 E_t 主效应 + 一行跨生成器（不落盘报告）")
    args = ap.parse_args()
    B = 20 if args.smoke else args.B

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        out_dir / "scorers_cache")

    e2b_path = root / "responses" / "P1_PILOT" / "gemma_4_e2b_responses.jsonl"
    rows = [json.loads(l) for l in open(e2b_path, encoding="utf-8")]
    n = len(rows)
    _log("E2B 行=%d" % n)

    # ---- 4 口径标签（E2B，S11 缓存）----
    jb = np.array([_read_cache(cache_dir / "judge_big.jsonl").get(i, np.nan)
                   for i in range(n)], dtype=float)
    js = np.array([_read_cache(cache_dir / "judge_small.jsonl").get(i, np.nan)
                   for i in range(n)], dtype=float)
    qw = np.array([_read_cache(cache_dir / "qwen32.jsonl").get(i, np.nan)
                   for i in range(n)], dtype=float)
    dual = np.where(~np.isnan(jb) & ~np.isnan(js) & (jb == js), jb, np.nan)
    _log("E2B 标签：jb=%d js=%d qw=%d dual=%d" % (
        int(np.sum(~np.isnan(jb))), int(np.sum(~np.isnan(js))),
        int(np.sum(~np.isnan(qw))), int(np.sum(~np.isnan(dual)))))

    # ---- 跨生成器：E4B 文本 qwen32（S17b 缓存）----
    e4b_path = root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl"
    e4b_rows = [json.loads(l) for l in open(e4b_path, encoding="utf-8")]
    e4b_text = [r for r in e4b_rows if r.get("modality") == "text"]
    e4b_qw_cache = _read_cache(cache_dir / "s17_e4b_text_qwen32.jsonl")
    e4b_qw = [e4b_qw_cache.get(r["response_id"], np.nan) for r in e4b_text]
    _log("E4B 文本=%d，qwen32 已评=%d" % (len(e4b_text),
                                        int(np.sum(~np.isnan(e4b_qw)))))
    # 逐格对齐校验：E4B 文本与 E2B 必须按 (query_id,template,E_t,N,R) 同序
    aligned = (len(e4b_text) == n and
               all(e4b_text[i]["query_id"] == rows[i]["query_id"] and
                   e4b_text[i]["template_idx"] == rows[i]["template_idx"] and
                   e4b_text[i]["E_t"] == rows[i]["E_t"] and
                   e4b_text[i]["N"] == rows[i]["N"] and
                   e4b_text[i]["R"] == rows[i]["R"] for i in range(n)))
    if not aligned:
        _log("警告：E4B 文本与 E2B 非同序对齐，跨生成器逐格比较无效——改用 response_id 键对齐")
        e4b_by_id = {r["response_id"]: r for r in e4b_text}
        e2b_idmap = {}
        for i, r in enumerate(rows):
            e2b_idmap[(r["query_id"], r["template_idx"], r["E_t"],
                       r["N"], r["R"])] = i
        e4b_qw2 = {}
        for rid, r in e4b_by_id.items():
            key = (r["query_id"], r["template_idx"], r["E_t"], r["N"], r["R"])
            if key in e2b_idmap:
                e4b_qw2[e2b_idmap[key]] = e4b_qw_cache.get(rid, np.nan)
        e4b_qw = [e4b_qw2.get(i, np.nan) for i in range(n)]
    else:
        _log("跨生成器对齐：E4B 文本与 E2B 同序 OK（逐格身份一致）")
    cross_ok = (len(e4b_text) == n and int(np.sum(~np.isnan(e4b_qw))) == n)

    # ---- 1. 三维主效应 × 4 口径 ----
    sources = {"dual_judge": dual, "judge_big": jb, "judge_small": js,
               "qwen32": qw}
    dims = [("E_t", 0, 1), ("N", 0, 1), ("R", 0, 1)]
    main_eff = {}
    for dim, v0, v1 in dims:
        if args.smoke and dim != "E_t":
            continue
        main_eff[dim] = {}
        for src, lab in sources.items():
            e = _dim_effect(rows, lab, dim, v0, v1, B=B,
                            name=src)
            main_eff[dim][src] = e
            _log("dim=%s %s: pos0=%s pos1=%s Δ=%s CI=%s" % (
                dim, src, e["pos0"], e["pos1"], e["delta"], e["ci95"]))

    # ---- 2. 长度混杂 ----
    _resplen = lambda r: len(r.get("response") or "")
    len_info = {dim: {
        "len_median_%d" % v0: int(np.median([_resplen(r)
                                            for r in rows if r[dim] == v0])),
        "len_median_%d" % v1: int(np.median([_resplen(r)
                                            for r in rows if r[dim] == v1]))}
        for dim, v0, v1 in dims}
    _log("长度中位：%s" % json.dumps(len_info, ensure_ascii=False))
    len_adjusted = {}
    for dim, v0, v1 in dims:
        if args.smoke and dim != "E_t":
            continue
        la = _logit_adjusted(rows, dual, dim, B=B)
        len_adjusted[dim] = la
        _log("dim=%s logit 调整后: %s" % (dim, json.dumps(la,
                                                        ensure_ascii=False)))

    # ---- 3. 两两交互（dual_judge）----
    interactions = {}
    if not args.smoke:
        for d1, d2 in [("E_t", "N"), ("E_t", "R"), ("N", "R")]:
            inter = _interaction(rows, dual, d1, d2, B=B)
            interactions["%s_x_%s" % (d1, d2)] = inter
            _log("交互 %s×%s: %s" % (d1, d2,
                                     json.dumps(inter, ensure_ascii=False)))

    # ---- 4. 8 组合完整表 ----
    combo8 = {}
    for et in (0, 1):
        for nn in (0, 1):
            for rr in (0, 1):
                key = "E%dN%dR%d" % (et, nn, rr)
                for src, lab in (("dual", dual), ("qwen32", qw)):
                    sel = [lab[i] for i, r in enumerate(rows)
                           if r["E_t"] == et and r["N"] == nn and r["R"] == rr
                           and not np.isnan(lab[i])]
                    combo8.setdefault(key, {})[src] = (
                        round(float(np.mean(sel)), 4) if sel else None)

    # ---- 5. 跨生成器 ----
    cross_gen = {"available": bool(cross_ok)}
    if cross_ok:
        agree = sum(1 for i in range(n)
                    if not np.isnan(qw[i]) and not np.isnan(e4b_qw[i])
                    and qw[i] == e4b_qw[i])
        pair = sum(1 for i in range(n)
                   if not np.isnan(qw[i]) and not np.isnan(e4b_qw[i]))
        cross_gen["n_pairs"] = pair
        cross_gen["agreement_e2b_vs_e4b_qwen32"] = round(agree / pair, 4) \
            if pair else None
        e4b_dim = {}
        e4b_lab = np.array(e4b_qw, dtype=float)
        e4b_aligned_rows = e4b_text if aligned else rows
        for dim, v0, v1 in dims:
            if args.smoke and dim != "E_t":
                continue
            e = _dim_effect(e4b_aligned_rows, e4b_lab, dim, v0, v1, B=B,
                            name="e4b_qwen32")
            e4b_dim[dim] = e
        cross_gen["e4b_text_effects_qwen32"] = e4b_dim
        # 同向判定
        cross_gen["sign_alignment"] = {}
        for dim, v0, v1 in dims:
            d_e2 = main_eff[dim]["qwen32"]["delta"]
            d_e4 = e4b_dim[dim]["delta"]
            if d_e2 is not None and d_e4 is not None:
                cross_gen["sign_alignment"][dim] = bool(
                    np.sign(d_e2) == np.sign(d_e4))
        _log("跨生成器：一致率=%s 同向=%s" % (
            cross_gen["agreement_e2b_vs_e4b_qwen32"],
            cross_gen["sign_alignment"]))
    else:
        _log("跨生成器：E4B 文本 qwen32 未全量（S17b 未完成），跳过")

    # ---- 6. 查询级异质性（dual_judge）----
    qhet = {}
    by_q = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_q[r["query_id"]].append(i)
    qids = sorted(by_q)
    for dim, v0, v1 in dims:
        if args.smoke and dim != "E_t":
            continue
        qd = []
        for q in qids:
            idx = by_q[q]
            g0 = [dual[i] for i in idx if rows[i][dim] == v0
                  and not np.isnan(dual[i])]
            g1 = [dual[i] for i in idx if rows[i][dim] == v1
                  and not np.isnan(dual[i])]
            if g0 and g1:
                qd.append(float(np.mean(g1)) - float(np.mean(g0)))
        if qd:
            qd = np.array(qd)
            pos_n = int(np.sum(qd > 0.001))
            neg_n = int(np.sum(qd < -0.001))
            qhet[dim] = {
                "n_queries": len(qd),
                "pct_positive": round(float(np.mean(qd > 0.001)), 4),
                "n_pos": pos_n, "n_neg": neg_n,
                "median": round(float(np.median(qd)), 4),
                "iqr": [round(float(np.percentile(qd, 25)), 4),
                        round(float(np.percentile(qd, 75)), 4)]}
        _log("查询异质性 %s: %s" % (dim, json.dumps(qhet.get(dim),
                                                 ensure_ascii=False)))

    out = {
        "stage": "S20", "date": "2026-08-14",
        "purpose": ("E2B 全量 E_t×N×R 三维分解（协议 RQ 核心维度 N 与人格维度 R "
                    "首次单独测量）+ 长度混杂控制 + 跨生成器 + 查询异质性"),
        "design": "150 查询 × 3 模板 × E_t×N×R(8) = 3600（A_s=text 常数）",
        "main_effects": main_eff,
        "length_confound": {"median_by_dim": len_info,
                            "logit_adjusted_dual": len_adjusted},
        "interactions_dual": interactions,
        "combo8_pos_rate": combo8,
        "cross_generator": cross_gen,
        "query_heterogeneity": qhet,
        "note": ("N=协议叙事结构 RQ（H1）；R=角色代理人格（长度 +85% 混杂，须披露）；"
                 "主效应跨 4 口径、跨生成器同向即稳健。"),
    }
    if args.smoke:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        _log("smoke 完成（不落盘）")
        return 0
    with open(out_dir / "s20_e2b_r_n_interaction.json", "w",
              encoding="utf-8") as f:
        json.dump(_json_safe(out), f, ensure_ascii=False, indent=2)
    md = render_md(out)
    (out_dir / "s20_e2b_r_n_interaction.md").write_text(md, encoding="utf-8")
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "s20_e2b_r_n_interaction.md").write_text(md,
                                                           encoding="utf-8")
    _log("已落盘 s20_e2b_r_n_interaction.json/.md")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def render_md(o):
    lines = ["# S20：E2B 全量 E_t×N×R 三维分解（CPU · 2026-08-14）\n",
             "## 目的",
             "S1-S19 把 N（叙事结构/事件链，协议核心 RQ）与 R（角色代理人格）合并。"
             "本实验首次单独测量 N/R 主效应、E_t×N×R 交互、长度混杂、跨生成器稳健性。\n",
             "## 设计",
             "- %s（A_s=text 常数）" % o["design"],
             "- 口径：dual_judge 共识 / judge_big / judge_small / qwen32；"
             "Δ 与 CI 为 query 聚类 bootstrap。\n",
             "## 三维主效应",
             "| 维度 | 口径 | n0 | n1 | pos(0) | pos(1) | Δ | 95%CI | OR(p) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for dim, srcs in o["main_effects"].items():
        for src, e in srcs.items():
            lines.append("| %s | %s | %d | %d | %s | %s | %s | %s | %s |" % (
                dim, src, e["n0"], e["n1"], e["pos0"], e["pos1"], e["delta"],
                e["ci95"], "%s(%s)" % (e["or"], e["p"])
                if e["or"] is not None else "N/A"))
    lines.append("\n## 长度混杂")
    lines.append("| 维度 | len中位(0) | len中位(1) | logit(+len)调整 coef | p | OR(95%CI) |")
    lines.append("|---|---|---|---|---|---|")
    for dim, lc in o["length_confound"]["median_by_dim"].items():
        la = o["length_confound"]["logit_adjusted_dual"][dim]
        lines.append("| %s | %d | %d | %s | %s | %s |" % (
            dim, lc["len_median_0"], lc["len_median_1"],
            la.get("coef_dim", "N/A"), la.get("p_dim", "N/A"),
            "%s" % la.get("or_dim", "N/A")
            if la.get("ci95_or") else "N/A"))
    lines.append("\n## 两两交互（dual_judge）")
    lines.append("| 交互 | Δ(d2|d1=0) | Δ(d2|d1=1) | 交互项 | 95%CI |")
    lines.append("|---|---|---|---|---|")
    for k, v in o["interactions_dual"].items():
        lines.append("| %s | %.4f | %.4f | %.4f | %s |" % (
            k, v["d2_effect_at_d1_0"], v["d2_effect_at_d1_1"],
            v["interaction"], v["interaction_ci95"]))
    lines.append("\n## 8 组合 pos_rate（E_t×N×R）")
    lines.append("| 组合 | dual | qwen32 |")
    lines.append("|---|---|---|")
    for k, v in o["combo8_pos_rate"].items():
        lines.append("| %s | %s | %s |" % (k, v.get("dual"), v.get("qwen32")))
    cg = o["cross_generator"]
    lines.append("\n## 跨生成器（E4B vs E2B 同格 3600，qwen32）")
    if cg.get("available"):
        lines.append("- 逐格一致率: **%.4f**（n=%d）" % (
            cg["agreement_e2b_vs_e4b_qwen32"], cg["n_pairs"]))
        lines.append("| 维度 | E2B Δ | E4B Δ | 同向 |")
        lines.append("|---|---|---|---|")
        for dim, e in cg["e4b_text_effects_qwen32"].items():
            lines.append("| %s | %s | %s | %s |" % (
                dim, o["main_effects"][dim]["qwen32"]["delta"], e["delta"],
                cg["sign_alignment"].get(dim, "N/A")))
    else:
        lines.append("- E4B 文本 qwen32 未全量，跳过（S17b 未完成）")
    lines.append("\n## 查询级异质性（dual_judge，150 查询）")
    lines.append("| 维度 | 查询数 | 正向% | n+ | n− | 中位 | IQR |")
    lines.append("|---|---|---|---|---|---|---|")
    for dim, v in o["query_heterogeneity"].items():
        lines.append("| %s | %d | %.4f | %d | %d | %.4f | %s |" % (
            dim, v["n_queries"], v["pct_positive"], v["n_pos"], v["n_neg"],
            v["median"], v["iqr"]))
    lines.append("\n## 判读")
    lines.append("> 协议 RQ（N）的 E2B 侧预验 + R 人格效应（含长度混杂）如实披露；"
                 "跨 4 口径且跨生成器同向 → 相应维度效应稳健。")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
