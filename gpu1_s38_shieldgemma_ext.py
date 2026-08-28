#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S38：ShieldGemma-9B 扩展评分——S17/S33 全量 + 跨生成器效应复现。

S37 发现（E2B 主链）：ShieldGemma 与 judge_big 整体一致率高（κ=0.777），但
**N 条件效应符号翻转**（N1-N0=-0.036✓，与 Gemma 评委家族 +0.048~+0.054 相反），
E_t@N=1 复现（+0.025✓）而 E_t@N=0/pooled 不显著。本脚本把同一独立评分器扩展到
流水线其余 3 个 scope，检验该分歧是 E2B 特有还是跨生成器泛化：

  - S17 E4B 音频 7200 格 + S17 E4B 文本 3600 格（第二生成器）
  - S33 异族音频 344 格

产出：
  1) 每个 scope：ShieldGemma vs 各家族（judge_big/judge_small/qwen32/
     strongreject/harmbench）交叉一致；
  2) 每个 scope：ShieldGemma vs judge_big 的 N/Et 条件效应（query 聚类
     bootstrap），刻画 N 翻转是否跨生成器成立；
  3) 独立新文件 s38_shieldgemma_labels.jsonl，未改写生产缓存。

纪律：零人工标注、零账本、不写 .complete/.done；CUDA_VISIBLE_DEVICES=1；
GPU1 串行检测；只读。

用法：CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s38_shieldgemma_ext.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu1_s28_hetero_audio as s28  # noqa: E402 _bootstrap_pair


def _gpu1_busy(min_free_mib=8192):
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


def _load_rid(p):
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            d = json.loads(line)
            r = d["rid"]
            out[r[4:] if r.startswith("s28_") else r] = d.get("label")
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
        print("[s38] %s" % m, flush=True)

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
    results = {"stage": "S38", "date": "2026-08-16",
               "purpose": ("ShieldGemma 扩展到 S17/S33：N 翻转是否跨生成器泛化 "
                           "+ 跨家族一致"),
               "method": ("ShieldGemmaScorer（4bit, yes=1/no=0）评分 E4B 音频 "
                          "7200 + E4B 文本 3600 + S33 344；效应=query 聚类 "
                          "bootstrap B=%d" % B),
               "disclosure": ("ShieldGemma 同 Google 家族（异构训练成立、跨家族不"
                               "成立）；label=「对话是否不安全」≠有害合规，绝对基率"
                               "可能不同。")}

    # ---- 响应加载 ----
    e4b_all = [json.loads(l) for l in
               (root / "responses" / "P1_PILOT" /
                "gemma_4_e4b_responses.jsonl").open(encoding="utf-8")]
    e4b_audio = [r for r in e4b_all if r.get("modality") == "audio"]
    e4b_text = [r for r in e4b_all if r.get("modality") == "text"]
    s33_all = [json.loads(l) for l in
               (out_dir / "responses" / "s33_hetero_audio.jsonl").open(
                   encoding="utf-8")]
    _log("E4B 音频=%d 文本=%d；S33=%d" % (len(e4b_audio), len(e4b_text),
                                        len(s33_all)))

    scopes = [
        {"key": "s17_e4b_audio", "rows": e4b_audio,
         "rid_fn": lambda r: r["response_id"]},
        {"key": "s17_e4b_text", "rows": e4b_text,
         "rid_fn": lambda r: r["response_id"]},
        {"key": "s33_hetero_audio", "rows": s33_all,
         "rid_fn": lambda r: "s33_" + r["response_id"]},
    ]

    all_label_lines = []
    for sp in scopes:
        key, rows, rid_fn = sp["key"], sp["rows"], sp["rid_fn"]
        n = len(rows)
        _log("=== %s（%d 格）===" % (key, n))
        rids = [rid_fn(r) for r in rows]
        by_rid = {rid_fn(r): r for r in rows}

        # ShieldGemma 全量评分
        sh = {}
        for k, r in enumerate(rows):
            res = sc.score_one(r.get("prompt") or "",
                               str(r.get("response") or ""))
            sh[rid_fn(r)] = res.get("label")
            if (k + 1) % 1200 == 0:
                _log("  %s ShieldGemma %d/%d" % (key, k + 1, n))
        parse = sum(1 for r in rids if sh[r] is not None)
        harm = sum(1 for r in rids if sh[r] == 1)

        # 家族缓存
        jb = _load_rid(cache_dir / (key + "_judge_big.jsonl"))
        js = _load_rid(cache_dir / (key + "_judge_small.jsonl"))
        others = {}
        for suf in ("qwen32", "strongreject", "harmbench"):
            p = cache_dir / (key + "_" + suf + ".jsonl")
            if p.exists():
                others[suf] = _load_rid(p)

        # 交叉一致
        fams = {"judge_big": jb, "judge_small": js}
        fams.update(others)
        fams["shieldgemma"] = sh
        matrix = {}
        kk = list(fams)
        for a in range(len(kk)):
            for b in range(a + 1, len(kk)):
                matrix["%s_vs_%s" % (kk[a], kk[b])] = _pairwise(
                    fams[kk[a]], fams[kk[b]], rids)

        # 效应复现（ShieldGemma vs judge_big）
        def _eff2(fn, sa, sb, off):
            a = [(by_rid[r]["query_id"], fn(r)) for r in rids
                 if sa(r) and fn(r) is not None]
            b = [(by_rid[r]["query_id"], fn(r)) for r in rids
                 if sb(r) and fn(r) is not None]
            return s28._bootstrap_pair(a, b, B, seed + off)

        E0 = lambda r: by_rid[r]["E_t"] == 0  # noqa: E731
        E1 = lambda r: by_rid[r]["E_t"] == 1  # noqa: E731
        N0 = lambda r: by_rid[r]["N"] == 0  # noqa: E731
        N1 = lambda r: by_rid[r]["N"] == 1  # noqa: E731

        results[key] = {
            "n_cells": n,
            "shieldgemma_parsed": parse,
            "shieldgemma_none": n - parse,
            "shieldgemma_harmful_rate": round(harm / max(1, parse), 4),
            "distribution_collapse": bool(
                (harm / max(1, parse)) < 0.05 or (harm / max(1, parse)) > 0.95),
            "matrix": matrix,
            "effects": {
                "N_effect_Et0": {
                    "shieldgemma": _eff2(lambda r: sh[r],
                                         lambda r: E0(r) and N0(r),
                                         lambda r: E0(r) and N1(r), 1),
                    "judge_big": _eff2(lambda r: jb.get(r),
                                       lambda r: E0(r) and N0(r),
                                       lambda r: E0(r) and N1(r), 2)},
                "Et_effect_N0": {
                    "shieldgemma": _eff2(lambda r: sh[r],
                                         lambda r: E0(r) and N0(r),
                                         lambda r: E1(r) and N0(r), 3),
                    "judge_big": _eff2(lambda r: jb.get(r),
                                       lambda r: E0(r) and N0(r),
                                       lambda r: E1(r) and N0(r), 4)},
                "Et_effect_N1": {
                    "shieldgemma": _eff2(lambda r: sh[r],
                                         lambda r: E0(r) and N1(r),
                                         lambda r: E1(r) and N1(r), 5),
                    "judge_big": _eff2(lambda r: jb.get(r),
                                       lambda r: E0(r) and N1(r),
                                       lambda r: E1(r) and N1(r), 6)},
            },
        }
        for r in rows:
            rid = rid_fn(r)
            all_label_lines.append({"scope": key, "rid": rid,
                                    "label": sh[rid]})
        e = results[key]["effects"]
        _log("  %s 完成：SG 有害率=%.4f；N(Et0) SG=%+.4f%s JB=%+.4f%s；"
             "Et(N1) SG=%+.4f%s JB=%+.4f%s" % (
                 key, results[key]["shieldgemma_harmful_rate"],
                 e["N_effect_Et0"]["shieldgemma"]["effect"],
                 "✓" if e["N_effect_Et0"]["shieldgemma"]["excl_zero"] else "✗",
                 e["N_effect_Et0"]["judge_big"]["effect"],
                 "✓" if e["N_effect_Et0"]["judge_big"]["excl_zero"] else "✗",
                 e["Et_effect_N1"]["shieldgemma"]["effect"],
                 "✓" if e["Et_effect_N1"]["shieldgemma"]["excl_zero"] else "✗",
                 e["Et_effect_N1"]["judge_big"]["effect"],
                 "✓" if e["Et_effect_N1"]["judge_big"]["excl_zero"] else "✗"))

    sc.close()
    _log("评分器已释放")

    (out_dir / "s38_shieldgemma_ext.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "s38_shieldgemma_labels.jsonl").open(
            "w", encoding="utf-8") as f:
        for x in all_label_lines:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # ---- 报告 ----
    lines = ["# S38：ShieldGemma 扩展（S17/S33）——N 翻转跨生成器检验",
             "",
             "- 日期：2026-08-16；方法：ShieldGemmaScorer（4bit, yes=1/no=0）",
             "- 结果 json：`results/gpu1_pipeline/s38_shieldgemma_ext.json`\n"]

    def _row(d):
        if not d:
            return "N/A"
        return "%d | %.3f | %.3f | %.3f" % (d["n"], d["agreement"],
                                            d["kappa"], d["spearman"])

    for key in ("s17_e4b_audio", "s17_e4b_text", "s33_hetero_audio"):
        r = results[key]
        lines += ["## %s（%d 格）\n" % (key, r["n_cells"]),
                  "ShieldGemma：解析 %d（None %d），有害率 %.4f%s\n" % (
                      r["shieldgemma_parsed"], r["shieldgemma_none"],
                      r["shieldgemma_harmful_rate"],
                      " ← 分布坍缩" if r["distribution_collapse"] else ""),
                  "| 对比 | n | 一致率 | κ | Spearman |",
                  "|---|---|---|---|---|"]
        for fam in ("judge_big", "judge_small", "qwen32", "strongreject",
                    "harmbench"):
            d = r["matrix"].get("shieldgemma_vs_%s" % fam) or \
                r["matrix"].get("%s_vs_shieldgemma" % fam)
            if d:
                lines.append("| SG vs %s | %s |" % (fam, _row(d)))
        lines.append("")
        lines.append("| 效应（N1-N0 / E1-E0） | ShieldGemma | judge_big |")
        lines.append("|---|---|---|")
        e = r["effects"]
        for eff_k, lab in (("N_effect_Et0", "N(E_t=0)"),
                           ("Et_effect_N0", "E_t(N=0)"),
                           ("Et_effect_N1", "E_t(N=1)")):
            lines.append("| %s | %s | %s |" % (
                lab, _fmt_eff(e[eff_k]["shieldgemma"]),
                _fmt_eff(e[eff_k]["judge_big"])))
        lines.append("")
    lines += ["## 解读",
              "",
              "- 若 N 翻转在 E4B 上也成立 → 分歧是跨生成器的（Scorer×Generator 无关），",
              "  归因于 ShieldGemma 的 moderation 构念而非某生成器伪影。",
              "- 若 E4B 上不再翻转 → 分歧可能部分来自 E2B 特有内容模式，需在论文中",
              "  限定该敏感性的适用范围。",
              "",
              "- 标签文件：`results/gpu1_pipeline/s38_shieldgemma_labels.jsonl`",
              "  （独立新文件，未改写生产缓存）"]
    (report_dir / "s38_shieldgemma_ext.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    _log("完成 → s38_shieldgemma_ext.json + s38_shieldgemma_labels.jsonl + "
         "report/s38_shieldgemma_ext.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
