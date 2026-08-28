#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S19：E2B 全量 4 评分器 × 8 聚合口径下 E_t 主效应敏感性矩阵（2026-08-14）。

动机：KBS 审稿人核心追问——「叙事框架（E_t=1）改变有害率」这一主效应是否
稳健于评分/聚合选择？S11 已在 E2B 全量 3600 用 dual_judge 共识核验跨族收敛，
但主效应本身只见单一口径。本实验在 E2B 侧对 4 个标注源（judge_big /
judge_small / qwen32 强锚 / cross_check 弱锚）做 8 种聚合口径全覆盖：

  Part A（GPU1）：补打分 cross_check（Qwen2.5-3B，第 4 标注源，增量缓存
    scorers_cache/cross_check_e2b.jsonl，崩溃可续）。
  Part B（CPU）：8 口径 = dual_judge 共识 / 单腿 ×4 / DS-3 / DS-4 /
    多数投票（3-of-4，争议行弃票）下计算 pos_rate(E_t)、Δ=pos(E1)−pos(E0)、
    query 聚类 bootstrap 95%CI（R76 按 query 抽样加权纪律）、Fisher OR+p。

判据：8 口径 Δ 符号一致 → 「主效应跨测量稳健」；否则如实披露分歧。
零人工标注；只写 results/gpu1_pipeline/s19_* + report/；不碰 E4B/E2B 账本。

用法：CUDA_VISIBLE_DEVICES=1 python gpu1_s19_e2b_sweep.py [--skip-cc]
"""
import argparse
import collections
import gc
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s19 %s] %s" % (Path(__file__).stem, m), flush=True)


def _read_cache(p):
    out = {}
    if not p.exists():
        return out
    for line in p.open(encoding="utf-8"):
        try:
            rec = json.loads(line)
            out[rec["i"]] = rec["label"]
        except Exception:  # noqa: BLE001
            continue
    return out


def _cache_full(p, need):
    if not p.exists():
        return False
    try:
        return len(_read_cache(p)) >= need
    except Exception:  # noqa: BLE001
        return False


def _score_cc(rows, cache_path):
    """cross_check 顺序打分，增量缓存（跳过 label=null 重评）。"""
    import gpu1_s9_cross_family as s9
    done = _read_cache(cache_path)
    if done:
        _log("cross_check 缓存恢复 %d 条" % len(done))
    cfg = yaml.safe_load(open("pipeline_config.yaml", encoding="utf-8"))
    s9.register_scorers(cfg)
    sc = s9.SCORER_FACTORIES["cross_check"]()
    _log("cross_check 加载完成")
    n_ok = 0
    for i, r in enumerate(rows):
        if i in done:
            continue
        try:
            res = sc.score_one(r["prompt"], r["response"] or "")
            label = res.get("label")
        except Exception as e:  # noqa: BLE001
            _log("[cc] idx=%d 失败: %s" % (i, str(e)[:120]))
            label = None
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"i": i, "label": label},
                               ensure_ascii=False) + "\n")
        if label is not None:
            n_ok += 1
        if (i + 1) % 400 == 0:
            _log("cross_check %d/%d" % (i + 1, len(rows)))
    try:
        sc.close()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    _log("cross_check 完成，非空 %d/%d" % (n_ok, len(rows)))


def _bootstrap_delta(rows, labels, seed=20260814, B=2000):
    """query 聚类 bootstrap：重抽 query_id 有放回，重算 Δ=pos(E1)-pos(E0)。"""
    by_q = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_q[r["query_id"]].append(i)
    qids = sorted(by_q)
    rng = np.random.RandomState(seed)
    n = len(qids)
    deltas = np.empty(B)
    for b in range(B):
        sel = []
        for _ in range(n):
            sel.extend(by_q[qids[rng.randint(n)]])
        e0 = [labels[i] for i in sel if rows[i]["E_t"] == 0]
        e1 = [labels[i] for i in sel if rows[i]["E_t"] == 1]
        e0 = [x for x in e0 if not np.isnan(x)]
        e1 = [x for x in e1 if not np.isnan(x)]
        if not e0 or not e1:
            deltas[b] = np.nan
            continue
        deltas[b] = np.mean(e1) - np.mean(e0)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--skip-cc", action="store_true")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    suffix = "_smoke" if args.smoke else ""
    n_rows_full = 60 if args.smoke else 3600

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        out_dir / "scorers_cache")

    e2b_path = root / "responses" / "P1_PILOT" / "gemma_4_e2b_responses.jsonl"
    rows = [json.loads(l) for l in open(e2b_path, encoding="utf-8")]
    if args.smoke:
        rows = rows[:n_rows_full]
        _log("SMOKE 模式：只取前 %d 行（不覆盖正式产物）" % len(rows))
    _log("E2B 行=%d" % len(rows))

    # ---- Part A：cross_check 第 4 标注源（GPU1）----
    cc_path = cache_dir / ("cross_check_e2b_smoke.jsonl" if args.smoke
                           else "cross_check_e2b.jsonl")
    if not args.skip_cc and not _cache_full(cc_path, len(rows)):
        _log("Part A：cross_check 打分（%d 条）" % len(rows))
        _score_cc(rows, cc_path)
    elif args.skip_cc:
        _log("Part A 跳过（--skip-cc）")
    else:
        _log("Part A 缓存完备，复用 cross_check_e2b.jsonl")

    # ---- 加载 4 源标签 ----
    caches = {sn: _read_cache(cache_dir / (sn + ".jsonl"))
              for sn in ("judge_big", "judge_small", "qwen32")}
    cc = _read_cache(cc_path)
    caches["cross_check"] = cc
    for sn, c in caches.items():
        _log("缓存 %s=%d 行" % (sn, len(c)))
    n = len(rows)
    B = np.array([caches["judge_big"].get(i, np.nan) for i in range(n)],
                 dtype=float)
    S = np.array([caches["judge_small"].get(i, np.nan) for i in range(n)],
                 dtype=float)
    Q = np.array([caches["qwen32"].get(i, np.nan) for i in range(n)],
                 dtype=float)
    C = np.array([caches["cross_check"].get(i, np.nan) for i in range(n)],
                 dtype=float)

    # ---- 8 种口径的标签数组 ----
    def _lab(x):
        return np.where(np.isnan(x), np.nan, x)

    dual = np.where(~np.isnan(B) & ~np.isnan(S) & (B == S), B, np.nan)
    maj4 = np.where(
        np.nansum(np.vstack([B, S, Q, C]), axis=0) >= 3, 1,
        np.where(np.nansum(np.vstack([B, S, Q, C]), axis=0) <= 1, 0, np.nan))
    from scorer_utils import dawid_skene
    mat3 = np.vstack([B, S, Q]).T
    mat4 = np.vstack([B, S, Q, C]).T
    ds3 = dawid_skene(mat3, n_iter=100)["item_label"]
    ds4 = dawid_skene(mat4, n_iter=100)["item_label"]

    protocols = {
        "dual_judge_consensus": dual,
        "judge_big": B,
        "judge_small": S,
        "qwen32": Q,
        "cross_check": C,
        "DS3_b_s_qw": ds3.astype(float),
        "DS4_b_s_qw_cc": ds4.astype(float),
        "majority_3of4": maj4,
    }

    # ---- 每口径主效应表 ----
    rows_idx = range(n)
    tab = {}
    for name, lab in protocols.items():
        e0 = [lab[i] for i in rows_idx if rows[i]["E_t"] == 0
              and not np.isnan(lab[i])]
        e1 = [lab[i] for i in rows_idx if rows[i]["E_t"] == 1
              and not np.isnan(lab[i])]
        p0 = float(np.mean(e0)) if e0 else None
        p1 = float(np.mean(e1)) if e1 else None
        delta = (p1 - p0) if (p0 is not None and p1 is not None) else None
        ci = _bootstrap_delta(rows, lab, seed=20260814, B=args.B)
        if e0 and e1:
            a = sum(1 for i in rows_idx if rows[i]["E_t"] == 1 and lab[i] == 1)
            c_ = sum(1 for i in rows_idx if rows[i]["E_t"] == 0 and lab[i] == 1)
            b_ = sum(1 for i in rows_idx if rows[i]["E_t"] == 1 and lab[i] == 0)
            d = sum(1 for i in rows_idx if rows[i]["E_t"] == 0 and lab[i] == 0)
            or_, pv = _fisher(a, b_, c_, d)
        else:
            or_, pv = None, None
        tab[name] = {
            "n_e0": len(e0), "n_e1": len(e1),
            "pos_rate_e0": round(p0, 4) if p0 is not None else None,
            "pos_rate_e1": round(p1, 4) if p1 is not None else None,
            "delta": round(delta, 4) if delta is not None else None,
            "delta_ci95": ci,
            "fisher_or": or_, "fisher_p": pv,
        }
        def _fmt(x):
            return ("%.4f" % x) if x is not None else "N/A"
        _log("%s: E0=%s E1=%s Δ=%s CI=%s OR=%s p=%s" % (
            name, _fmt(tab[name]["pos_rate_e0"]),
            _fmt(tab[name]["pos_rate_e1"]), _fmt(tab[name]["delta"]),
            ci, or_, pv))

    # ---- 符号一致性判定 ----
    deltas = [tab[k]["delta"] for k in tab if tab[k]["delta"] is not None]
    sign_consistent = bool(deltas) and len(set(np.sign(deltas))) == 1
    ci_excl0 = sum(1 for k in tab
                   if tab[k]["delta_ci95"] is not None
                   and tab[k]["delta_ci95"][0] > 0)
    n_ci = sum(1 for k in tab if tab[k]["delta_ci95"] is not None)

    # ---- 4 评分器两两一致率/κ（全量 3600）----
    from scorer_utils import cohens_kappa
    pair_agree = {}
    names = ["judge_big", "judge_small", "qwen32", "cross_check"]
    srcs = {"judge_big": B, "judge_small": S, "qwen32": Q, "cross_check": C}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            pairs = [(srcs[a][k], srcs[b][k]) for k in rows_idx
                     if not np.isnan(srcs[a][k]) and not np.isnan(srcs[b][k])]
            if len(pairs) < 10:
                continue
            agree = sum(1 for x, y in pairs if x == y) / len(pairs)
            try:
                kres = cohens_kappa([x for x, _ in pairs],
                                    [y for _, y in pairs])
                kappa, ci = kres["kappa"], kres["ci95"]
            except Exception:  # noqa: BLE001
                kappa, ci = None, None
            pair_agree["%s↔%s" % (a, b)] = {
                "n": len(pairs), "agreement": round(agree, 4),
                "kappa": round(kappa, 4) if kappa is not None else None,
                "kappa_ci95": [round(v, 4) for v in ci] if ci is not None
                else None}

    # ---- 争议仲裁（b≠s 时 qw/cc 站队）----
    disc = [(B[i], S[i], Q[i], C[i]) for i in rows_idx
            if not np.isnan(B[i]) and not np.isnan(S[i]) and B[i] != S[i]]
    qw_big = sum(1 for b, s, q, c in disc if not np.isnan(q) and q == b)
    qw_small = sum(1 for b, s, q, c in disc if not np.isnan(q) and q == s)
    cc_big = sum(1 for b, s, q, c in disc if not np.isnan(c) and c == b)
    cc_small = sum(1 for b, s, q, c in disc if not np.isnan(c) and c == s)
    agree2 = sum(1 for b, s, q, c in disc
                 if not np.isnan(q) and not np.isnan(c) and q == c)

    # ---- 按 template 分层（dual_judge + DS4）----
    t_strat = {}
    for name in ("dual_judge_consensus", "DS4_b_s_qw_cc", "majority_3of4"):
        lab = protocols[name]
        t2 = {}
        for t in (0, 1, 2):
            e0 = [lab[i] for i in rows_idx if rows[i]["E_t"] == 0
                  and rows[i]["template_idx"] == t and not np.isnan(lab[i])]
            e1 = [lab[i] for i in rows_idx if rows[i]["E_t"] == 1
                  and rows[i]["template_idx"] == t and not np.isnan(lab[i])]
            if e0 and e1:
                t2["t%d" % t] = {
                    "n_e0": len(e0), "n_e1": len(e1),
                    "pos_e0": round(float(np.mean(e0)), 4),
                    "pos_e1": round(float(np.mean(e1)), 4),
                    "delta": round(float(np.mean(e1)) - float(np.mean(e0)), 4)}
        t_strat[name] = t2

    # ---- DS 收敛与评分器质量 ----
    ds3_info = {"converged": bool(dawid_skene(mat3, n_iter=100)["converged"]),
                "sens": [round(float(v), 4) for v in
                         dawid_skene(mat3, n_iter=100)["sensitivity"]],
                "spec": [round(float(v), 4) for v in
                         dawid_skene(mat3, n_iter=100)["specificity"]]}
    ds4_res = dawid_skene(mat4, n_iter=100)
    ds4_info = {"converged": bool(ds4_res["converged"]),
                "sens": [round(float(v), 4) for v in ds4_res["sensitivity"]],
                "spec": [round(float(v), 4) for v in ds4_res["specificity"]]}

    out = {
        "stage": "S19", "date": "2026-08-14",
        "purpose": ("E2B 全量 4 标注源 × 8 聚合口径下 E_t 主效应敏感性矩阵；"
                    "检验叙事框架效应是否跨测量选择稳健"),
        "n_rows": n,
        "label_sources": {
            "judge_big": int(np.sum(~np.isnan(B))),
            "judge_small": int(np.sum(~np.isnan(S))),
            "qwen32": int(np.sum(~np.isnan(Q))),
            "cross_check": int(np.sum(~np.isnan(C))),
        },
        "protocols": tab,
        "sign_consistent_across_protocols": bool(sign_consistent),
        "n_protocols_ci_excl0": ci_excl0, "n_protocols_ci": n_ci,
        "pairwise_agreement": pair_agree,
        "dispute_arbitration": {
            "n_disputed_b_neq_s": len(disc),
            "qwen32_agrees_big": qw_big, "qwen32_agrees_small": qw_small,
            "cross_check_agrees_big": cc_big,
            "cross_check_agrees_small": cc_small,
            "qw_and_cc_agree_each_other": agree2,
        },
        "template_stratified": t_strat,
        "ds3": ds3_info, "ds4": ds4_info,
        "note": ("若 sign_consistent=True 且多数口径 CI 不含 0：叙事框架主效应"
                 "跨评分/聚合稳健。cross_check 为弱锚（3B 宽松），其单腿口径"
                 "预期与强评分器分歧，如实披露。"),
    }
    with open(out_dir / ("s19_e2b_sweep%s.json" % suffix), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    md = render_md(out)
    (out_dir / ("s19_e2b_sweep%s.md" % suffix)).write_text(md,
                                                           encoding="utf-8")
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / ("s19_e2b_sweep%s.md" % suffix)).write_text(md,
                                                              encoding="utf-8")
    _log("已落盘 s19_e2b_sweep%s.json/.md" % suffix)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def render_md(o):
    lines = [
        "# S19：E2B 全量评分口径敏感性矩阵（GPU1 · 2026-08-14）\n",
        "## 目的",
        "检验「叙事框架（E_t=1）改变有害率」主效应是否稳健于评分/聚合选择："
        "4 个标注源 × 8 种聚合口径全覆盖，query 聚类 bootstrap 95%%CI。\n",
        "## 数据",
        "- 行：E2B 全量 %d；标注源覆盖 judge_big=%d / judge_small=%d / "
        "qwen32=%d / cross_check=%d" % (
            o["n_rows"], o["label_sources"]["judge_big"],
            o["label_sources"]["judge_small"], o["label_sources"]["qwen32"],
            o["label_sources"]["cross_check"]),
        "## 8 口径主效应表",
        "| 口径 | n(E0) | n(E1) | pos(E0) | pos(E1) | Δ | Δ 95%CI | Fisher OR(p) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, v in o["protocols"].items():
        lines.append("| %s | %d | %d | %s | %s | %s | %s | %s |" % (
            k, v["n_e0"], v["n_e1"],
            v["pos_rate_e0"], v["pos_rate_e1"],
            v["delta"], v["delta_ci95"],
            "%s(%s)" % (v["fisher_or"], v["fisher_p"])
            if v["fisher_or"] is not None else "N/A"))
    lines.append("\n## 符号一致性判定")
    lines.append("- 8 口径 Δ 符号一致：**%s**（%d/%d 口径 CI 不含 0）" % (
        o["sign_consistent_across_protocols"], o["n_protocols_ci_excl0"],
        o["n_protocols_ci"]))
    lines.append("\n## 4 评分器两两一致率 / κ")
    lines.append("| 对 | n | 一致率 | κ(95%CI) |")
    lines.append("|---|---|---|---|")
    for k, v in o["pairwise_agreement"].items():
        lines.append("| %s | %d | %.4f | %s |" % (
            k, v["n"], v["agreement"], v["kappa"]))
    da = o["dispute_arbitration"]
    lines.append("\n## 争议仲裁（b≠s，n=%d）" % da["n_disputed_b_neq_s"])
    lines.append("- qwen32 站队 judge_big: %d / judge_small: %d" % (
        da["qwen32_agrees_big"], da["qwen32_agrees_small"]))
    lines.append("- cross_check 站队 judge_big: %d / judge_small: %d" % (
        da["cross_check_agrees_big"], da["cross_check_agrees_small"]))
    lines.append("\n## 按 template 分层（关键口径）")
    for k, v in o["template_stratified"].items():
        lines.append("\n**%s**" % k)
        lines.append("| t | n(E0) | n(E1) | pos(E0) | pos(E1) | Δ |")
        lines.append("|---|---|---|---|---|---|")
        for t, tv in v.items():
            lines.append("| %s | %d | %d | %.4f | %.4f | %.4f |" % (
                t, tv["n_e0"], tv["n_e1"], tv["pos_e0"], tv["pos_e1"],
                tv["delta"]))
    lines.append("\n## DS 质量")
    for k in ("ds3", "ds4"):
        d = o[k]
        lines.append("- %s: converged=%s sens=%s spec=%s" % (
            k, d["converged"], d["sens"], d["spec"]))
    lines.append("\n## 判读")
    lines.append("> 若符号一致且多数口径 CI 排除 0：叙事框架主效应跨测量选择稳健"
                 "（审稿人最可能追问的稳健性问题闭合）。cross_check 为弱锚单腿，"
                 "预期分歧已如实披露。")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
