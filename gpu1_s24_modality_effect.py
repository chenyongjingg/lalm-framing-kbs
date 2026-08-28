#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S24：E4B 模态效应（text vs neutral_audio vs styled_audio）＋模态内 framing 效应（CPU · 2026-08-14）。

动机：论文核心卖点为「音频/多模态通道放大 framing 攻击」（prompt.md L56 多模态贡献）。
但截至 S23，所有 framing 因子效应（N/E_t/R）都只在文本模态内测量过（E2B 纯文本、
E4B 文本 3600）。音频模态（7200 单元，占 P1-PILOT 2/3）从未进入效应分析，且
audio 与 text 构成完全配对设计（每 (query,combo,template) 有 text + neutral_audio +
styled_audio 三行）。本实验第一次回答：
  1. 模态主效应：audio（分 neutral/styled）的有害率是否显著高于 text？（多模态放大声明）
  2. 模态内 framing 效应：N（协议 RQ）/E_t/R 在 audio 内是否与 text 同向、量级如何？
     —— G1 闸门 C2 要求决策口径 dual_judge N 主效应 ≥10pp，而文本侧仅 ~6pp；
       音频是否放大 N 效应直接决定全量闸门可行性。
  3. 长度混杂（R3 纪律）：audio 响应显著更短（S10：中位 275 vs 文本 722），
     必须披露 len 按模态分布并给出按 len 三分位分层敏感性。
  4. 模板伪影：t2（E_t=1/t2 触发 plot_stall，S12b）按模板分层 + 干净 t0+t1 敏感性。

口径（R2 预注册 + S9/S11 已验证）：主口径 dual_judge 共识 + qwen32 强锚；
judge_big/judge_small 仅敏感性。

纪律：
  - 纯 CPU、零生成。只读 responses + scorers_cache 既有缓存，只写 s24_* 产物。
  - audio 为当前快照（4872/7200，E4B 仍在生成），如实标注 snapshot，
    E4B 完成后重跑本脚本获得权威值（幂等，rid 键控）。
  - 不写账本/done，不修改任何主流水线文件。

用法：python gpu1_s24_modality_effect.py [--B 2000] [--smoke]
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s24 %s] %s" % (Path(__file__).stem, m), flush=True)


def _json_safe(o):
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


def _read_rid_cache(p):
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                out[rec["rid"]] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
    return out


def _fisher(a, b, c, d):
    """2x2 [[a,b],[c,d]] Fisher exact OR+p（scipy 兜底手算）。"""
    try:
        from scipy.stats import fisher_exact
        odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        return round(float(odds), 4) if odds is not None else None, round(float(p), 6)
    except Exception:  # noqa: BLE001
        return None, None


def _modality_bootstrap(rows, labels, a0, a1, seed=20260815, B=2000):
    """query 聚类 bootstrap：Δ = pos(A_s=a1) − pos(A_s=a0)，a 为 A_s 水平。"""
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
        g0 = [labels[i] for i in sel if rows[i].get("A_s") == a0
              and not np.isnan(labels[i])]
        g1 = [labels[i] for i in sel if rows[i].get("A_s") == a1
              and not np.isnan(labels[i])]
        if not g0 or not g1:
            deltas[b] = np.nan
            continue
        deltas[b] = np.mean(g1) - np.mean(g0)
    ok = deltas[~np.isnan(deltas)]
    if len(ok) < B // 2:
        return None
    return [round(float(np.percentile(ok, 2.5)), 4),
            round(float(np.percentile(ok, 97.5)), 4)]


def _modality_effect(rows, labels, a0, a1, B=2000, name=""):
    e0 = [labels[i] for i, r in enumerate(rows)
          if r.get("A_s") == a0 and not np.isnan(labels[i])]
    e1 = [labels[i] for i, r in enumerate(rows)
          if r.get("A_s") == a1 and not np.isnan(labels[i])]
    p0 = float(np.mean(e0)) if e0 else None
    p1 = float(np.mean(e1)) if e1 else None
    delta = (p1 - p0) if (p0 is not None and p1 is not None) else None
    ci = _modality_bootstrap(rows, labels, a0, a1, B=B)
    # Fisher 2x2：[[有害1,无害1],[有害0,无害0]]
    h1 = int(sum(e1)) if e1 else 0
    h0 = int(sum(e0)) if e0 else 0
    odds, p = _fisher(h1, len(e1) - h1, h0, len(e0) - h0)
    return {"n0": len(e0), "n1": len(e1),
            "pos0": round(p0, 4) if p0 is not None else None,
            "pos1": round(p1, 4) if p1 is not None else None,
            "delta": round(delta, 4) if delta is not None else None,
            "ci95": ci, "or_fisher": odds, "p_fisher": p, "name": name}


def _dim_effect_m(rows, labels, dim, v0, v1, B=2000, name=""):
    """复用 S20c._dim_effect：行级过滤后的单模态内效应。"""
    from gpu1_s20c_e4b_dual_crossgen import _dim_effect
    return _dim_effect(rows, labels, dim, v0, v1, B=B, name=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--B", type=int, default=2000)
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

    rows = [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl",
        encoding="utf-8")]
    for r in rows:
        r["_len"] = len(r.get("response") or "")
    n_text = sum(1 for r in rows if r.get("A_s") == "text")
    n_neu = sum(1 for r in rows if r.get("A_s") == "neutral_audio")
    n_sty = sum(1 for r in rows if r.get("A_s") == "styled_audio")
    _log("E4B 总=%d text=%d neutral_audio=%d styled_audio=%d (audio 快照)"
         % (len(rows), n_text, n_neu, n_sty))

    # ---- 标签（rid 键控）----
    rid = [r["response_id"] for r in rows]
    jb_c = _read_rid_cache(cache_dir / "s17_e4b_audio_judge_big.jsonl")
    js_c = _read_rid_cache(cache_dir / "s17_e4b_audio_judge_small.jsonl")
    qw_c = _read_rid_cache(cache_dir / "s17_e4b_audio_qwen32.jsonl")
    # text 侧
    for tag, fn in [("jb_t", "s17_e4b_text_judge_big.jsonl"),
                    ("js_t", "s17_e4b_text_judge_small.jsonl"),
                    ("qw_t", "s17_e4b_text_qwen32.jsonl")]:
        d = _read_rid_cache(cache_dir / fn)
        _log("%s 缓存 %d 条" % (tag, len(d)))

    def _lbl(cache, tag):
        return np.array([cache.get(x, np.nan) for x in rid], dtype=float)

    # audio 缓存缺 text rid 会自动 NaN；text 缓存缺 audio rid 自动 NaN。
    # 合并：audio 行用 audio 缓存、text 行用 text 缓存。
    jb_a = _lbl(jb_c, "jb_a")
    js_a = _lbl(js_c, "js_a")
    qw_a = _lbl(qw_c, "qw_a")
    jb_t = _lbl(_read_rid_cache(cache_dir / "s17_e4b_text_judge_big.jsonl"), "jb_t")
    js_t = _lbl(_read_rid_cache(cache_dir / "s17_e4b_text_judge_small.jsonl"), "js_t")
    qw_t = _lbl(_read_rid_cache(cache_dir / "s17_e4b_text_qwen32.jsonl"), "qw_t")

    n_audio = len(rows)
    jb = np.where([r.get("A_s") == "audio" or str(r.get("A_s", "")).endswith("audio")
                   for r in rows], jb_a, jb_t)
    js = np.where([str(r.get("A_s", "")).endswith("audio") for r in rows],
                  js_a, js_t)
    qw = np.where([str(r.get("A_s", "")).endswith("audio") for r in rows],
                  qw_a, qw_t)
    dual = np.where(~np.isnan(jb) & ~np.isnan(js) & (jb == js), jb, np.nan)
    for tag, arr in [("jb", jb), ("js", js), ("qw", qw), ("dual", dual)]:
        _log("label %s 非空=%d/%d" % (tag, int(np.sum(~np.isnan(arr))), len(rows)))

    srcs = ["dual", "qwen32", "judge_big", "judge_small"]
    lbl = {"dual": dual, "qwen32": qw, "judge_big": jb, "judge_small": js}
    a_levels = ["text", "neutral_audio", "styled_audio"]

    out = {
        "stage": "S24",
        "date": "2026-08-14",
        "purpose": "E4B 模态效应（text/neutral/styled）＋模态内 N/E_t/R 效应；G1 C2 闸门可行性",
        "snapshot": {
            "note": "audio 为 E4B 生成中快照（4872/7200），E4B 完成后重跑获得权威值",
            "text": n_text, "neutral_audio": n_neu, "styled_audio": n_sty,
        },
    }

    # ============ 1. 模态主效应 ============
    _log("---- 1. 模态主效应 ----")
    mod_eff = {}
    pairs = [("text", "neutral_audio"), ("text", "styled_audio"),
             ("neutral_audio", "styled_audio")]
    for s in srcs:
        mod_eff[s] = {}
        for a0, a1 in pairs:
            if args.smoke and a1 != "styled_audio":
                continue
            me = _modality_effect(rows, lbl[s], a0, a1, B=B, name="%s_%s_%s" % (s, a0, a1))
            mod_eff[s]["%s_vs_%s" % (a0, a1)] = me
            _log("mod %s %s_vs_%s: pos0=%s pos1=%s Δ=%s CI=%s" % (
                s, a0, a1, me["pos0"], me["pos1"], me["delta"], me["ci95"]))
    out["modality_main_effect"] = mod_eff

    # ============ 2. 模态内 framing 效应（N/E_t/R × 每模态） ============
    _log("---- 2. 模态内 framing 效应 ----")
    dims = [("N", 0, 1), ("E_t", 0, 1), ("R", 0, 1)]
    within = {}
    for s in ["dual", "qwen32"]:
        within[s] = {}
        for lvl in a_levels:
            idx = [i for i, r in enumerate(rows) if r.get("A_s") == lvl]
            sub_r = [rows[i] for i in idx]
            sub_l = lbl[s][idx]
            within[s][lvl] = {}
            for dim, v0, v1 in dims:
                if args.smoke and dim != "N":
                    continue
                d = _dim_effect_m(sub_r, sub_l, dim, v0, v1, B=B,
                                  name="%s_%s" % (s, lvl))
                within[s][lvl][dim] = d
                _log("within %s %s %s: Δ=%s CI=%s (n0=%s n1=%s)" % (
                    s, lvl, dim, d["delta"], d["ci95"], d["n0"], d["n1"]))
    out["within_modality_framing"] = within

    # ============ 3. 长度混杂（R3 纪律） ============
    _log("---- 3. 长度分布 ----")
    lenstat = {}
    for lvl in a_levels:
        ls = [r["_len"] for r in rows if r.get("A_s") == lvl]
        if ls:
            lenstat[lvl] = {
                "n": len(ls), "median": int(np.median(ls)),
                "mean": round(float(np.mean(ls)), 1),
                "q25": int(np.percentile(ls, 25)), "q75": int(np.percentile(ls, 75)),
            }
        else:
            lenstat[lvl] = None
    _log("len by modality: %s" % json.dumps(lenstat, ensure_ascii=False))
    out["length_by_modality"] = lenstat

    # 按 len 三分位分层的模态效应（dual + qwen32）——长度混杂敏感性
    _log("---- 3b. 按 len 三分位分层的 text vs styled 模态效应 ----")
    lens = sorted(r["_len"] for r in rows if r.get("A_s") == "text")
    # 统一用全样本 len 三分位阈值（避免每组样本量分裂）
    all_lens = sorted(r["_len"] for r in rows)
    q33, q67 = int(np.percentile(all_lens, 33)), int(np.percentile(all_lens, 67))
    len_strat = {}
    for s in ["dual", "qwen32"]:
        len_strat[s] = {}
        # 三层显式构造（全样本 len 三分位阈值）
        for name, (lo, hi) in {"low": (None, q33), "mid": (q33, q67),
                               "high": (q67, None)}.items():
            sel = [i for i, r in enumerate(rows)
                   if (lo is None or r["_len"] > lo) and (hi is None or r["_len"] <= hi)]
            if not sel:
                len_strat[s][name] = None
                continue
            sub_r = [rows[i] for i in sel]
            sub_l = lbl[s][sel]
            me = _modality_effect(sub_r, sub_l, "text", "styled_audio", B=B,
                                  name="%s_len_%s" % (s, name))
            len_strat[s][name] = me
            _log("lenstrata %s %s (%s,%s]: Δ=%s CI=%s n=%d" % (
                s, name, lo, hi, me["delta"], me["ci95"], len(sel)))
    out["modality_by_len_tercile"] = len_strat

    # ============ 4. 按模板分层（t2 plot_stall 伪影敏感性） ============
    _log("---- 4. 按模板分层的 text vs styled ----")
    tpl = {}
    for s in ["dual", "qwen32"]:
        tpl[s] = {}
        for t in [0, 1, 2]:
            sel = [i for i, r in enumerate(rows) if r.get("template_idx") == t]
            sub_r = [rows[i] for i in sel]
            sub_l = lbl[s][sel]
            me = _modality_effect(sub_r, sub_l, "text", "styled_audio", B=B,
                                  name="%s_t%d" % (s, t))
            tpl[s]["t%d" % t] = me
        # 干净 t0+t1
        sel = [i for i, r in enumerate(rows) if r.get("template_idx") in (0, 1)]
        tpl[s]["t01_clean"] = _modality_effect(
            [rows[i] for i in sel], lbl[s][sel], "text", "styled_audio",
            B=B, name="%s_t01" % s)
        _log("tpl %s t01: Δ=%s" % (s, tpl[s]["t01_clean"]["delta"]))
    out["modality_by_template"] = tpl

    # ============ 5. 覆盖率 / 争议率 ============
    _log("---- 5. 覆盖率 ----")
    cov = {}
    for lvl in a_levels:
        idx = [i for i, r in enumerate(rows) if r.get("A_s") == lvl]
        n = len(idx)
        n_dual = int(np.sum(~np.isnan(dual[idx])))
        n_qw = int(np.sum(~np.isnan(qw[idx])))
        dispute = int(np.sum(~np.isnan(jb[idx]) & ~np.isnan(js[idx])
                             & (jb[idx] != js[idx])))
        both = int(np.sum(~np.isnan(jb[idx]) & ~np.isnan(js[idx])))
        cov[lvl] = {"n": n, "dual_n": n_dual,
                    "dual_coverage": round(n_dual / n, 4) if n else None,
                    "qwen32_n": n_qw,
                    "dispute": dispute,
                    "dispute_rate": round(dispute / both, 4) if both else None}
        _log("cov %s: %s" % (lvl, json.dumps(cov[lvl], ensure_ascii=False)))
    out["coverage"] = cov

    # 快照标记：N_main 效应是否达到 G1 C2 10pp 判据（dual，各模态）
    g1 = {}
    for lvl in a_levels:
        d = within["dual"][lvl]["N"]
        g1[lvl] = {"N_delta": d["delta"], "N_ci95": d["ci95"],
                   "ge_10pp": (d["delta"] is not None and abs(d["delta"]) >= 0.10)}
    out["g1_c2_projection"] = g1
    _log("G1 C2 投影: %s" % json.dumps(_json_safe(g1), ensure_ascii=False))

    if args.smoke:
        print(json.dumps(_json_safe(out), ensure_ascii=False, indent=2))
        _log("smoke 完成（不落盘）")
        return 0

    with open(out_dir / "s24_modality_effect.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(out), f, ensure_ascii=False, indent=2)
    _render_md(out, report_dir / "s24_modality_effect.md")
    _log("已落盘 s24_modality_effect.json + report/s24_modality_effect.md")
    return 0


def _render_md(o, path):
    L = ["# S24：E4B 模态效应（text / neutral_audio / styled_audio）\n",
         "\n**日期**：2026-08-14 ｜ **类型**：CPU 全缓存分析 ｜ **状态**：完成 ｜ "
         "**快照**：audio=%d/%d（E4B 生成中，完成后重跑）\n" % (
             o["snapshot"]["neutral_audio"] + o["snapshot"]["styled_audio"],
             o["snapshot"]["neutral_audio"] + o["snapshot"]["styled_audio"])]
    # 占位：snapshot 目标 7200 由脚本外注明
    L.append("\n> audio 为当前快照（4866→4872 量级），**非权威值**；E4B 完成后重跑本脚本。\n")

    L.append("\n## 1. 模态主效应（Δ = pos(au) − pos(text)）\n")
    L.append("| 口径 | 对比 | n0 | n1 | pos(text) | pos(audio) | Δ | 95%CI | Fisher OR(p) |\n")
    L.append("|---|---|---|---|---|---|---|---|---|\n")
    for s, dv in o["modality_main_effect"].items():
        for cmpname, me in dv.items():
            if "text_vs_neutral_audio" in cmpname:
                continue  # 主表只放 text vs styled；neutral 单独下段
            a0, a1 = cmpname.split("_vs_")
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" % (
                s, cmpname, me["n0"], me["n1"], me["pos0"], me["pos1"],
                me["delta"], me["ci95"],
                ("%s(%s)" % (me["or_fisher"], me["p_fisher"]))
                if me["or_fisher"] is not None else "-"))
    L.append("\n### 全部三水平两两（含 neutral vs styled）\n")
    L.append("| 口径 | 对比 | pos0 | pos1 | Δ | 95%CI |\n")
    L.append("|---|---|---|---|---|---|\n")
    for s, dv in o["modality_main_effect"].items():
        for cmpname, me in dv.items():
            L.append("| %s | %s | %s | %s | %s | %s |\n" % (
                s, cmpname, me["pos0"], me["pos1"], me["delta"], me["ci95"]))

    L.append("\n## 2. 模态内 framing 效应（dual_judge + qwen32）\n")
    L.append("| 口径 | 模态 | 维度 | n0 | n1 | pos0 | pos1 | Δ | 95%CI |\n")
    L.append("|---|---|---|---|---|---|---|---|---|\n")
    for s, mdl in o["within_modality_framing"].items():
        for lvl, dmap in mdl.items():
            for dim, d in dmap.items():
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" % (
                    s, lvl, dim, d["n0"], d["n1"], d["pos0"], d["pos1"],
                    d["delta"], d["ci95"]))

    L.append("\n## 3. 长度分布（R3 混杂披露）\n")
    L.append("| 模态 | n | 中位 | 均值 | q25 | q75 |\n")
    L.append("|---|---|---|---|---|---|\n")
    for lvl, d in o["length_by_modality"].items():
        if d:
            L.append("| %s | %s | %s | %s | %s | %s |\n" % (
                lvl, d["n"], d["median"], d["mean"], d["q25"], d["q75"]))
    L.append("\n### 按 len 三分位分层的 text vs styled\n")
    L.append("| 口径 | len层 | n | pos(text) | pos(styled) | Δ | 95%CI |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for s, dmap in o["modality_by_len_tercile"].items():
        for nm, me in dmap.items():
            if me is None:
                continue
            L.append("| %s | %s | %s | %s | %s | %s | %s |\n" % (
                s, nm, me["n0"] + me["n1"], me["pos0"], me["pos1"],
                me["delta"], me["ci95"]))

    L.append("\n## 4. 按模板分层（t2 plot_stall 伪影敏感性）\n")
    L.append("| 口径 | 模板 | n | pos(text) | pos(styled) | Δ | 95%CI |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for s, dmap in o["modality_by_template"].items():
        for nm, me in dmap.items():
            L.append("| %s | %s | %s | %s | %s | %s | %s |\n" % (
                s, nm, me["n0"] + me["n1"], me["pos0"], me["pos1"],
                me["delta"], me["ci95"]))

    L.append("\n## 5. G1 C2 闸门投影（N 主效应 per 模态，dual 口径，≥10pp 判据）\n")
    L.append("| 模态 | N Δ | 95%CI | ≥10pp |\n")
    L.append("|---|---|---|---|\n")
    for lvl, d in o["g1_c2_projection"].items():
        L.append("| %s | %s | %s | %s |\n" % (
            lvl, d["N_delta"], d["N_ci95"], d["ge_10pp"]))

    L.append("\n## 判读\n")
    L.append("> 模态效应与模态内 framing 效应的符号、量级、显著性、长度混杂与模板伪影影响，"
             "以及 G1 C2 是否随音频纳入而达标，均如实披露。\n")
    path.write_text("".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
