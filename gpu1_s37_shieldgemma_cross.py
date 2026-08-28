#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S37：ShieldGemma-9B 独立评分家族——交叉一致矩阵 + N/E_t 效应复现。

ShieldGemma-9B（4bit bnb，ShieldGemmaScorer，gpu1_common）回答「对话是否不安全」
（yes=1 有害 / no=0 安全），与论文 Gemma-4 双评分器（judge_big/judge_small）是
不同训练目标与世代——**异构训练成立、跨家族不成立**（同 Google 家族）。S3 已在
JALMBench 602 条上验证其有效性（guard-bench）。

覆盖：
  - E2B 主链 3600 格（index 制）：与 judge_big/judge_small/qwen32/cross_check/
    strongreject/harmbench/forced(S35) 共 7 家交叉一致；
  - S28 异族音频 1200 格（rid 制）：与 judge_big/judge_small/strongreject/
    harmbench/forced(S35) 交叉一致。

产出：
  1) 8 家族交叉一致矩阵（一致率/κ/Spearman）；
  2) 用 ShieldGemma 独立评分器复现论文 N/E_t 主效应（query 聚类 bootstrap），
     与 dual_judge 权威口径对照——若方向/显著性跨评分器家族成立，是 KBS
     测量稳健性的强证据；
  3) 分布坍缩检测（解析格有害率 <0.05 或 >0.95 → 如实披露）；
  4) 独立新文件 s37_shieldgemma_*_labels.jsonl，未改写任何生产缓存。

纪律：零人工标注、零账本、不写 .complete/.done；CUDA_VISIBLE_DEVICES=1 调用方
注入；GPU1 串行检测（被占退出）；只读 responses/* + scorers_cache/* + S35 标签。

用法：CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s37_shieldgemma_cross.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu1_s28_hetero_audio as s28  # noqa: E402 _bootstrap_pair


def _gpu1_busy(min_free_mib=8192):
    """GPU1（index 1）剩余显存是否不足。主链在 GPU1 有 ~4.8GB 常驻，
    不阻塞 4bit 评分器；真正冲突是 GPU1 free < 8GB。"""
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines()
    for ln in out:
        if not ln.strip():
            continue
        idx, free = [x.strip() for x in ln.split(",")]
        if idx == "1":
            return int(free) < min_free_mib
    return True


def _load_indexed(p):
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            d = json.loads(line)
            out[d["i"]] = d.get("label")
    return out


def _load_rid(p, strip_prefix=True):
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            d = json.loads(line)
            r = d["rid"]
            if strip_prefix and r.startswith("s28_"):
                r = r[4:]
            out[r] = d.get("label")
    return out


def _pairwise(lab1, lab2, ids):
    a = np.array([float(lab1[i]) for i in ids
                  if lab1.get(i) is not None and lab2.get(i) is not None])
    b = np.array([float(lab2[i]) for i in ids
                  if lab1.get(i) is not None and lab2.get(i) is not None])
    n = len(a)
    if n < 10:
        return None
    agree = float((a == b).mean())
    p0 = agree
    p1 = float(a.mean()) * float((1 - b).mean()) + \
        (1 - float(a.mean())) * float(b.mean())
    kappa = (p0 - p1) / (1 - p1) if (1 - p1) > 1e-9 else 1.0
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    da, db = ar - ar.mean(), br - br.mean()
    rho = float((da * db).sum() /
                np.sqrt((da ** 2).sum() * (db ** 2).sum()))
    return {"n": n, "agreement": round(agree, 4), "kappa": round(kappa, 4),
            "spearman": round(rho, 4)}


def _fmt_eff(eff):
    if not eff:
        return "N/A"
    return "%.4f [%s,%s] %s" % (eff["effect"], eff["ci95"][0], eff["ci95"][1],
                                "✓" if eff["excl_zero"] else "✗")


def main():
    cfg = yaml.safe_load(open("pipeline_config.yaml", encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    cache_dir = out_dir / "scorers_cache"
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    def _log(m):
        print("[s37] %s" % m, flush=True)

    if _gpu1_busy():
        _log("GPU1 剩余显存不足，退出（串行纪律）")
        return 1
    _log("GPU1 空闲，开始")

    from gpu1_common import ShieldGemmaScorer
    _log("加载 ShieldGemma-9B（4bit）评分器 ...")
    sc = ShieldGemmaScorer()
    _log("评分器就绪")

    B = cfg.get("seeds", {}).get("bootstrap", 2000)
    seed = cfg.get("seeds", {}).get("bootstrap", 42)
    results = {"stage": "S37", "date": "2026-08-16",
               "purpose": ("ShieldGemma-9B 独立评分家族：交叉一致矩阵 + "
                           "N/E_t 效应复现（测量稳健性）"),
               "method": ("ShieldGemmaScorer（4bit bnb, yes=1/no=0）评分 E2B 主链 "
                          "3600 + S28 1200；效应=query 聚类 bootstrap B=%d" % B),
               "disclosure": ("ShieldGemma 与 Gemma-4 评分器同 Google 家族——异构"
                               "训练成立、跨家族不成立。label=回答「对话是否不安全」"
                               "（yes→1 有害/no→0 安全），与论文有害合规口径方向一致但"
                               "绝对基率可能不同。")}

    # ============ A. E2B 主链（index 制，3600 格） ============
    _log("=== A. E2B 主链 ===")
    e2b_rows = [json.loads(l) for l in
                (root / "responses" / "P1_PILOT" /
                 "gemma_4_e2b_responses.jsonl").open(encoding="utf-8")]
    n = len(e2b_rows)
    sh = {}
    for i, r in enumerate(e2b_rows):
        res = sc.score_one(r.get("prompt") or "", str(r.get("response") or ""))
        sh[i] = res.get("label")
        if (i + 1) % 500 == 0:
            _log("ShieldGemma E2B 主链 %d/%d" % (i + 1, n))

    # 各家族标签（E2B 主链）
    jb = _load_indexed(cache_dir / "judge_big.jsonl")
    js = _load_indexed(cache_dir / "judge_small.jsonl")
    qw = _load_indexed(cache_dir / "qwen32.jsonl")
    xc = _load_indexed(cache_dir / "cross_check_e2b.jsonl")
    forced = _load_indexed(out_dir / "s35_forced_verdict_e2b_main_labels.jsonl")
    ssr_rid = _load_rid(cache_dir / "s17_e2b_text_strongreject.jsonl")
    hb_rid = _load_rid(cache_dir / "s17_e2b_text_harmbench.jsonl")
    ssr = {i: ssr_rid.get(e2b_rows[i]["response_id"]) for i in range(n)}
    hb = {i: hb_rid.get(e2b_rows[i]["response_id"]) for i in range(n)}

    sh_parse = sum(1 for i in range(n) if sh[i] is not None)
    sh_none = n - sh_parse
    harm_rate = sum(1 for i in range(n) if sh[i] == 1) / max(1, sh_parse)
    _log("ShieldGemma E2B 主链完成：解析 %d（None %d），有害率 %.4f" % (
        sh_parse, sh_none, harm_rate))

    families = {
        "judge_big": jb, "judge_small": js, "qwen32": qw,
        "cross_check": xc, "strongreject": ssr, "harmbench": hb,
        "shieldgemma": sh, "forced": forced}
    matrix = {}
    keys = list(families)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            name = "%s_vs_%s" % (keys[a], keys[b])
            matrix[name] = _pairwise(families[keys[a]], families[keys[b]],
                                     list(range(n)))

    def _dual(i):
        b, s = jb.get(i), js.get(i)
        if b is not None and s is not None and b == s:
            return float(b)
        return None

    def _eff2(fn, sel_a, sel_b, off):
        a = [(e2b_rows[i]["query_id"], fn(i)) for i in range(n)
             if sel_a(i) and fn(i) is not None]
        b = [(e2b_rows[i]["query_id"], fn(i)) for i in range(n)
             if sel_b(i) and fn(i) is not None]
        return s28._bootstrap_pair(a, b, B, seed + off)

    def _rate(fn, sel):
        xs = [fn(i) for i in range(n) if sel(i) and fn(i) is not None]
        if not xs:
            return None
        h = sum(1 for x in xs if x == 1)
        return {"n": len(xs), "harm": h, "rate": round(h / len(xs), 4)}

    E0 = lambda i: e2b_rows[i]["E_t"] == 0  # noqa: E731
    E1 = lambda i: e2b_rows[i]["E_t"] == 1  # noqa: E731
    N0 = lambda i: e2b_rows[i]["N"] == 0  # noqa: E731
    N1 = lambda i: e2b_rows[i]["N"] == 1  # noqa: E731

    results["main"] = {
        "n_cells": n,
        "shieldgemma_parsed": sh_parse,
        "shieldgemma_none": sh_none,
        "shieldgemma_harmful_rate": round(harm_rate, 4),
        "distribution_collapse": bool(harm_rate < 0.05 or harm_rate > 0.95),
        "matrix": matrix,
        "shieldgemma_rates_by_condition": {
            "E0_N0": _rate(lambda i: sh[i], lambda i: E0(i) and N0(i)),
            "E0_N1": _rate(lambda i: sh[i], lambda i: E0(i) and N1(i)),
            "E1_N0": _rate(lambda i: sh[i], lambda i: E1(i) and N0(i)),
            "E1_N1": _rate(lambda i: sh[i], lambda i: E1(i) and N1(i)),
        },
        "effects": {
            "N_effect_Et0": {
                "shieldgemma": _eff2(lambda i: sh[i],
                                     lambda i: E0(i) and N0(i),
                                     lambda i: E0(i) and N1(i), 1),
                "dual_judge": _eff2(_dual, lambda i: E0(i) and N0(i),
                                    lambda i: E0(i) and N1(i), 2)},
            "Et_effect_N0": {
                "shieldgemma": _eff2(lambda i: sh[i],
                                     lambda i: E0(i) and N0(i),
                                     lambda i: E1(i) and N0(i), 3),
                "dual_judge": _eff2(_dual, lambda i: E0(i) and N0(i),
                                    lambda i: E1(i) and N0(i), 4)},
            "Et_effect_N1": {
                "shieldgemma": _eff2(lambda i: sh[i],
                                     lambda i: E0(i) and N1(i),
                                     lambda i: E1(i) and N1(i), 5),
                "dual_judge": _eff2(_dual, lambda i: E0(i) and N1(i),
                                    lambda i: E1(i) and N1(i), 6)},
            "Et_effect_pooled": {
                "shieldgemma": _eff2(lambda i: sh[i], E0, E1, 7),
                "dual_judge": _eff2(_dual, E0, E1, 8)},
        },
    }
    _log("E2B 主链效应：ShieldGemma N=%.4f / Et(N0)=%.4f / dual N=%.4f / "
         "Et(N0)=%.4f" % (
             results["main"]["effects"]["N_effect_Et0"]["shieldgemma"]["effect"],
             results["main"]["effects"]["Et_effect_N0"]["shieldgemma"]["effect"],
             results["main"]["effects"]["N_effect_Et0"]["dual_judge"]["effect"],
             results["main"]["effects"]["Et_effect_N0"]["dual_judge"]["effect"]))

    # ============ B. S28 异族音频（rid 制，1200 格） ============
    _log("=== B. S28 ===")
    s28_rows = [json.loads(l) for l in
                (out_dir / "responses" / "s28_hetero_audio.jsonl").open(
                    encoding="utf-8")]
    n28 = len(s28_rows)
    sh28 = {}
    for j, r in enumerate(s28_rows):
        res = sc.score_one(r.get("prompt") or "", str(r.get("response") or ""))
        sh28[r["response_id"]] = res.get("label")
        if (j + 1) % 300 == 0:
            _log("ShieldGemma S28 %d/%d" % (j + 1, n28))
    sh28_parse = sum(1 for v in sh28.values() if v is not None)
    h28 = sum(1 for v in sh28.values() if v == 1)
    _log("ShieldGemma S28 完成：解析 %d（None %d），有害率 %.4f" % (
        sh28_parse, n28 - sh28_parse, h28 / max(1, sh28_parse)))

    sjs = _load_rid(cache_dir / "s28_hetero_audio_judge_small.jsonl")
    sjb = _load_rid(cache_dir / "s28_hetero_audio_judge_big.jsonl")
    ssr28 = _load_rid(cache_dir / "s28_hetero_audio_strongreject.jsonl")
    shb28 = _load_rid(cache_dir / "s28_hetero_audio_harmbench.jsonl")
    forced28 = _load_rid(out_dir / "s35_forced_verdict_s28_labels.jsonl")
    rids28 = [r["response_id"] for r in s28_rows]
    fam28 = {"judge_small": sjs, "judge_big": sjb, "strongreject": ssr28,
             "harmbench": shb28, "forced": forced28, "shieldgemma": sh28}
    mat28 = {}
    k28 = list(fam28)
    for a in range(len(k28)):
        for b in range(a + 1, len(k28)):
            mat28["%s_vs_%s" % (k28[a], k28[b])] = _pairwise(
                fam28[k28[a]], fam28[k28[b]], rids28)
    results["s28"] = {
        "n_cells": n28,
        "shieldgemma_parsed": sh28_parse,
        "shieldgemma_none": n28 - sh28_parse,
        "shieldgemma_harmful_rate": round(h28 / max(1, sh28_parse), 4),
        "distribution_collapse": bool(
            (h28 / max(1, sh28_parse)) < 0.05 or
            (h28 / max(1, sh28_parse)) > 0.95),
        "matrix": mat28,
    }

    sc.close()
    _log("评分器已释放")

    # ---- 落盘 ----
    (out_dir / "s37_shieldgemma_cross.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "s37_shieldgemma_e2b_main_labels.jsonl").open(
            "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"i": i, "label": sh[i]},
                               ensure_ascii=False) + "\n")
    with (out_dir / "s37_shieldgemma_s28_labels.jsonl").open(
            "w", encoding="utf-8") as f:
        for r in s28_rows:
            rid = r["response_id"]
            f.write(json.dumps({"rid": "s28_" + rid, "label": sh28[rid]},
                               ensure_ascii=False) + "\n")

    # ---- 报告 ----
    lines = ["# S37：ShieldGemma-9B 独立评分家族（交叉一致 + 效应复现）",
             "",
             "- 日期：2026-08-16；方法：ShieldGemmaScorer（4bit bnb，yes=1/no=0）",
             "- 披露：同 Google 家族——异构训练成立、跨家族不成立；绝对基率可能与",
             "  Gemma-4 评分器不同（见条件效应而非绝对率）。",
             "- 结果 json：`results/gpu1_pipeline/s37_shieldgemma_cross.json`\n"]

    def _row(d):
        if not d:
            return "N/A"
        return "%d | %.3f | %.3f | %.3f" % (d["n"], d["agreement"],
                                            d["kappa"], d["spearman"])

    m = results["main"]
    lines += ["## A. E2B 主链（3600 格）\n",
              "ShieldGemma：解析 %d（None %d），有害率 %.4f%s\n" % (
                  m["shieldgemma_parsed"], m["shieldgemma_none"],
                  m["shieldgemma_harmful_rate"],
                  " ← 分布坍缩，谨慎解读" if m["distribution_collapse"] else ""),
              "### A.1 交叉一致矩阵（一致率 | κ | Spearman）\n",
              "| 对比 | n | 一致率 | κ | Spearman |",
              "|---|---|---|---|---|"]
    for fam in ("judge_big", "judge_small", "qwen32", "cross_check",
                "strongreject", "harmbench", "forced"):
        lines.append("| shieldgemma vs %s | %s |" % (
            fam, _row(m["matrix"].get("shieldgemma_vs_%s" % fam) or
                      m["matrix"].get("%s_vs_shieldgemma" % fam))))
    lines.append("")
    lines.append("### A.2 效应复现（query 聚类 bootstrap；✓=95%CI 排除 0）\n")
    lines.append("| 效应 | ShieldGemma | dual_judge（论文权威） |")
    lines.append("|---|---|---|")
    for eff_k, lab in (("N_effect_Et0", "N 效应（E_t=0, N0 vs N1）"),
                       ("Et_effect_N0", "E_t 效应（N=0, E0 vs E1）"),
                       ("Et_effect_N1", "E_t 效应（N=1）"),
                       ("Et_effect_pooled", "E_t 效应（pooled）")):
        e = m["effects"][eff_k]
        lines.append("| %s | %s | %s |" % (
            lab, _fmt_eff(e["shieldgemma"]), _fmt_eff(e["dual_judge"])))
    lines.append("")
    lines.append("### A.3 ShieldGemma 有害率按条件\n")
    lines.append("| 条件 | n | 有害率 |")
    lines.append("|---|---|---|")
    for ck, d in m["shieldgemma_rates_by_condition"].items():
        lines.append("| %s | %d | %.4f |" % (ck, d["n"], d["rate"]))

    s28r = results["s28"]
    lines += ["", "## B. S28 异族音频（1200 格）\n",
              "ShieldGemma：解析 %d（None %d），有害率 %.4f%s\n" % (
                  s28r["shieldgemma_parsed"], s28r["shieldgemma_none"],
                  s28r["shieldgemma_harmful_rate"],
                  " ← 分布坍缩，谨慎解读" if s28r["distribution_collapse"] else ""),
              "| 对比 | n | 一致率 | κ | Spearman |",
              "|---|---|---|---|---|"]
    for fam in ("judge_small", "judge_big", "strongreject", "harmbench",
                "forced"):
        lines.append("| shieldgemma vs %s | %s |" % (
            fam, _row(s28r["matrix"].get("shieldgemma_vs_%s" % fam) or
                      s28r["matrix"].get("%s_vs_shieldgemma" % fam))))
    lines += ["", "## 解读",
              "",
              ("- 若 ShieldGemma 的 N/E_t 效应与 dual_judge 同方向且显著：效应跨"
               "评分器家族复现，测量稳健性强证据（KBS 加分）。"),
              ("- 若不一致：如实披露为评分器敏感性（论文结论以权威 dual_judge 口径"
               "为准），并给出基率差异解释。"),
              "",
              "- 标签文件：`results/gpu1_pipeline/s37_shieldgemma_e2b_main_labels.jsonl`",
              "  + `s37_shieldgemma_s28_labels.jsonl`（独立新文件，未改写生产缓存）"]
    (report_dir / "s37_shieldgemma_cross.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    _log("完成 → s37_shieldgemma_cross.json + s37_shieldgemma_*_labels.jsonl + "
         "report/s37_shieldgemma_cross.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
