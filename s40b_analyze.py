#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S40b 良性对照分析——framing 特异性检验（纯 CPU）。

输入：
  results/benign_control/benign_scores.jsonl（S40 评分产物，last-wins per response_id）
  results/p1_full_scored.parquet（P1-FULL 恶意池对照）
输出：
  results/benign_control/benign_control_analysis.json
  report/benign_control.md

核心问题：
  1) 良性查询在各 framing 条件下的 dual_judge 有害率（期望 ≈0）→ 假阳性地板。
  2) storytelling/unrestricted vs baseline 的良性 Δ（期望 ≈0）→ 与恶意池
     （baseline 2.09% → storytelling 40.1%）对比，证明 framing 特异性放大恶意查询。
  3) 逐条翻检：良性查询在 framing 下是否出现 0→1 翻转（期望极少）。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load_scores(path):
    """last-wins per response_id。"""
    recs = {}
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if l:
                r = json.loads(l)
                recs[r["response_id"]] = r
    return list(recs.values())


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def asr_stats(df):
    """对一组行计算 dual_judge 有害率 + Wilson CI。"""
    dual = df["dual_judge_label"].dropna()
    n = len(df)
    n_valid = int(dual.notna().sum())
    if n_valid == 0:
        return {"n": n, "n_valid": 0, "asr": None, "ci": None}
    p, lo, hi = wilson(int(dual.astype(int).sum()), n_valid)
    return {"n": n, "n_valid": n_valid, "asr": round(p * 100, 3),
            "ci": [round(lo * 100, 3), round(hi * 100, 3)]}


def cluster_boot_delta(base, story, n_boot=2000, seed=42):
    """query 配对簇 bootstrap 的 Δ(story-base) 95% CI（query 级均值对齐）。"""
    rng = np.random.default_rng(seed)
    qids = np.array(sorted(set(base.index) & set(story.index)))
    b = base[qids].to_numpy()
    s = story[qids].to_numpy()
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(qids), size=len(qids))
        deltas[i] = s[idx].mean() - b[idx].mean()
    return round(deltas.mean() * 100, 3), \
        [round(np.percentile(deltas, 2.5) * 100, 3),
         round(np.percentile(deltas, 97.5) * 100, 3)]


def main():
    scores = load_scores(ROOT / "results" / "benign_control" / "benign_scores.jsonl")
    if not scores:
        print("[致命] 无评分产物，先跑 S40")
        return 3
    df = pd.DataFrame(scores)
    # 合并 dual_judge 共识（一致才给标签）
    b = df["judge_big_label"].astype("float")
    m = df["judge_mistral_label"].astype("float")
    df["dual_judge_label"] = np.where(b.notna() & m.notna() & (b == m), b, np.nan)
    df["dual_agree"] = b.notna() & m.notna() & (b == m)
    df["dispute"] = b.notna() & m.notna() & (b != m)

    n_total = len(df)
    n_dual = int(df["dual_agree"].sum())
    n_dispute = int(df["dispute"].sum())
    print(f"响应 {n_total}  双判一致 {n_dual}  争议 {n_dispute}")

    out = {
        "stage": "S40b", "n_responses": n_total, "n_dual_agree": n_dual,
        "n_dispute": n_dispute,
        "per_condition": {}, "per_condition_lang": {},
        "per_condition_template": {}, "flips": None,
    }

    # 1) 按条件
    for cond in ["baseline", "storytelling", "unrestricted"]:
        sub = df[df["condition"] == cond]
        out["per_condition"][cond] = asr_stats(sub)

    # 2) 按条件 × 语言
    for cond in ["baseline", "storytelling", "unrestricted"]:
        out["per_condition_lang"][cond] = {}
        for lang in ["zh", "en"]:
            sub = df[(df["condition"] == cond) & (df["lang"] == lang)]
            out["per_condition_lang"][cond][lang] = asr_stats(sub)

    # 3) 按条件 × 模板
    for cond in ["baseline", "storytelling", "unrestricted"]:
        out["per_condition_template"][cond] = {}
        for ti in sorted(df["template_idx"].dropna().unique()):
            sub = df[(df["condition"] == cond) & (df["template_idx"] == ti)]
            out["per_condition_template"][cond][int(ti)] = asr_stats(sub)

    # 4) Δ：storytelling/unrestricted vs baseline（query 配对）
    base = df[df["condition"] == "baseline"].groupby("query_id")[
        "dual_judge_label"].mean()
    out["delta_vs_baseline"] = {}
    for cond in ["storytelling", "unrestricted"]:
        grp = df[df["condition"] == cond].groupby("query_id")[
            "dual_judge_label"].mean()
        if len(base.dropna()) and len(grp.dropna()):
            d, ci = cluster_boot_delta(base.dropna(), grp.dropna())
            out["delta_vs_baseline"][cond] = {"delta_pp": d, "ci": ci}
        else:
            out["delta_vs_baseline"][cond] = {"delta_pp": None, "ci": None}

    # 5) 逐 query 翻转（baseline 无 → story/unrest 有害）
    piv = df.pivot_table(index="query_id", columns="condition",
                         values="dual_judge_label", aggfunc="first")
    flips = {}
    for cond in ["storytelling", "unrestricted"]:
        if cond in piv.columns and "baseline" in piv.columns:
            flips[cond] = {
                "0_to_1": int(((piv[cond] == 1) & (piv["baseline"] == 0)).sum()),
                "1_to_0": int(((piv[cond] == 0) & (piv["baseline"] == 1)).sum()),
                "n_with_both": int(piv[[cond, "baseline"]].dropna().shape[0]),
            }
    out["flips"] = flips

    # 6) 恶意池对照（P1-FULL 主 scope）
    p = ROOT / "results" / "p1_full_scored.parquet"
    contrast = {}
    if p.exists():
        pf = pd.read_parquet(p)
        pf = pf[(pf["lang"].isin(["zh", "en"])) &
                (pf["model"].isin(["gemma_4_e4b", "gemma_4_e2b"]))]
        contrast["malicious"] = {}
        for cond in ["baseline", "storytelling", "unrestricted"]:
            sub = pf[pf["condition"] == cond]
            contrast["malicious"][cond] = asr_stats(sub)
    out["contrast_malicious"] = contrast.get("malicious")

    (ROOT / "results" / "benign_control").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "benign_control" / "benign_control_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7) 报告
    md = ["# S40 良性查询对照：framing 特异性检验", "",
          f"- 响应数：{n_total}（zh/en 对半，category 分层）",
          f"- 双判一致（dual_judge 权威口径）：{n_dual}，争议：{n_dispute}",
          f"- 生成：Gemma-4-E2B bf16 贪心；评分：DualJudge（E4B+E2B），与 P1-FULL 协议一致", ""]
    md.append("## 各条件良性有害率（dual_judge，Wilson 95% CI）")
    md.append("| condition | n | dual 有效 | ASR% | 95% CI |")
    md.append("|---|---|---|---|---|")
    for cond in ["baseline", "storytelling", "unrestricted"]:
        s = out["per_condition"][cond]
        ci = f"{s['ci'][0]}–{s['ci'][1]}" if s["ci"] else "—"
        md.append(f"| {cond} | {s['n']} | {s['n_valid']} | {s['asr']} | {ci} |")
    md.append("")
    md.append("## Δ vs baseline（query 配对簇 bootstrap 95% CI）")
    for cond in ["storytelling", "unrestricted"]:
        d = out["delta_vs_baseline"][cond]
        md.append(f"- **{cond}**: Δ={d['delta_pp']}pp, CI={d['ci']}")
    md.append("")
    md.append("## 恶意池对照（P1-FULL 主 scope：text zh/en e4b/e2b）")
    md.append("| condition | ASR% |")
    md.append("|---|---|")
    for cond in ["baseline", "storytelling", "unrestricted"]:
        s = contrast["malicious"][cond]
        md.append(f"| {cond} | {s['asr']} |")
    md.append("")
    md.append("## 逐 query 翻转（framing 下良性是否被判有害）")
    for cond, f_ in flips.items():
        md.append(f"- {cond}: 0→1 = {f_['0_to_1']}, 1→0 = {f_['1_to_0']}, "
                  f"双条件均有效 n={f_['n_with_both']}")
    md.append("")
    md.append("## 判读")
    md.append("- 若各条件良性有害率 ≈0 且 framing Δ 排除显著为正 → 支持"
              "「framing 特异性放大恶意查询」，良性假阳性地板极低（红旗4 闭合）。")
    md.append("- 若 storytelling 良性有害率显著 >0 → 如实披露 framing 对良性亦有"
              "一定放大，需量化并纳入论文敏感性讨论。")

    (ROOT / "report").mkdir(parents=True, exist_ok=True)
    (ROOT / "report" / "benign_control.md").write_text("\n".join(md), encoding="utf-8")
    print("分析完成 → report/benign_control.md")
    print("\n".join(md[:20]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
