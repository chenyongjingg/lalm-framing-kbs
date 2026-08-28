#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S18：dual_judge 争议 × 停滞/属性交叉分析（CPU，2026-08-14）。

动机：E2B 全量 dual_judge 争议率 18.2%（651/3573，S11）。审稿人必问——
争议是否聚集在停滞伪影区（E_t=1/template t2）？若争议高发于停滞区，则
dual_judge 共识口径在该区域可靠性受损，主效应（尤其 E_t=1）需谨慎解读。
本实验把争议率 × 停滞类别 × E_t × template 交叉，检验口径稳健性。

数据源（零 GPU）：
  - E2B 3600 响应（100% 完成）。
  - S11 缓存标签 results/gpu1_pipeline/scorers_cache/{judge_big,judge_small,qwen32}.jsonl。
  - s10b.classify()（plot_stall/data_clarify/refusal/other）。

输出：results/gpu1_pipeline/s18_dispute_stall.{json,md} + report/。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s18 %s] %s" % (Path(__file__).stem, m), flush=True)


def _load_labels(cache_dir, rows):
    lab = {sn: [None] * len(rows) for sn in
           ("judge_big", "judge_small", "qwen32")}
    for sn in lab:
        p = cache_dir / (sn + ".jsonl")
        if not p.exists():
            _log("[S11 缓存] %s 缺失（S11 未完成）" % sn)
            continue
        for line in p.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                lab[sn][rec["i"]] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
    return lab


def _fisher(a, b, c, d):
    """2x2 表 [[a,b],[c,d]] Fisher 精确检验（两尾）。"""
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
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        out_dir / "scorers_cache")

    import gpu1_s10b as s10b  # noqa: E402

    e2b_path = root / "responses" / "P1_PILOT" / "gemma_4_e2b_responses.jsonl"
    rows = [json.loads(l) for l in open(e2b_path, encoding="utf-8")]
    lab = _load_labels(cache_dir, rows)
    _log("E2B 行=%d" % len(rows))

    # ---- 每行：类别 + 争议状态 + 三方标签 ----
    recs = []
    for i, r in enumerate(rows):
        b, s, q = lab["judge_big"][i], lab["judge_small"][i], lab["qwen32"][i]
        cls = s10b.classify(r)
        recs.append({
            "i": i, "E_t": r["E_t"], "t": r["template_idx"],
            "cls": cls,
            "disputed": (b is not None and s is not None and b != s),
            "b": b, "s": s, "q": q,
            "dual": (int(b) if (b is not None and b == s) else None),
        })

    n_total = len(recs)
    n_disc = sum(1 for x in recs if x["disputed"])
    n_dual = sum(1 for x in recs if x["dual"] is not None)

    # ---- 1. 争议率 by E_t × template ----
    et_tab = defaultdict(lambda: {"n": 0, "n_disc": 0})
    for x in recs:
        k = (x["E_t"], x["t"])
        et_tab[k]["n"] += 1
        et_tab[k]["n_disc"] += int(x["disputed"])

    # ---- 2. 争议率 by 类别（停滞 vs 非停滞）----
    cls_tab = defaultdict(lambda: {"n": 0, "n_disc": 0, "b_pos": 0, "s_pos": 0,
                                   "q_pos": 0})
    for x in recs:
        cls_tab[x["cls"]]["n"] += 1
        cls_tab[x["cls"]]["n_disc"] += int(x["disputed"])
        cls_tab[x["cls"]]["b_pos"] += int(x["b"] == 1)
        cls_tab[x["cls"]]["s_pos"] += int(x["s"] == 1)
        cls_tab[x["cls"]]["q_pos"] += int(x["q"] == 1)

    # ---- 3. 停滞区 vs 非停滞区 争议 Fisher ----
    stall = [x for x in recs if x["cls"] == "plot_stall"]
    non_stall = [x for x in recs if x["cls"] != "plot_stall"]
    disc_stall = sum(1 for x in stall if x["disputed"])
    disc_non = sum(1 for x in non_stall if x["disputed"])
    fish = _fisher(disc_stall, len(stall) - disc_stall,
                   disc_non, len(non_stall) - disc_non)

    # ---- 4. 争议中 qwen32 偏向谁（谁"更像真值"）----
    # 争议行（b != s）：qwen32 与 judge_big 一致？与 judge_small 一致？
    disc_qw_big = disc_qw_small = disc_qw_neither = 0
    for x in recs:
        if not x["disputed"] or x["q"] is None:
            continue
        if x["q"] == x["b"]:
            disc_qw_big += 1
        elif x["q"] == x["s"]:
            disc_qw_small += 1
        else:
            disc_qw_neither += 1

    # ---- 5. 跨族收敛是否被停滞污染：dual_judge vs qwen32 一致率 by 类别 ----
    conv = {}
    for cls in ("plot_stall", "data_clarify", "refusal", "other"):
        pairs = [(x["dual"], x["q"]) for x in recs
                 if x["cls"] == cls and x["dual"] is not None
                 and x["q"] is not None]
        if pairs:
            agree = sum(1 for a, b in pairs if a == b) / len(pairs)
            conv[cls] = {"n": len(pairs),
                         "agreement_dual_vs_qwen32": round(agree, 4),
                         "pass_0_80": agree >= 0.80}

    # ---- 6. by E_t 汇总 ----
    et_total = defaultdict(lambda: {"n": 0, "n_disc": 0})
    for x in recs:
        et_total[x["E_t"]]["n"] += 1
        et_total[x["E_t"]]["n_disc"] += int(x["disputed"])

    out = {
        "stage": "S18", "date": "2026-08-14",
        "purpose": ("dual_judge 争议是否聚集于停滞伪影区（E_t=1/t2）；"
                    "跨族收敛是否被停滞污染"),
        "n_total": n_total, "n_disputed": n_disc,
        "dispute_rate_overall": round(n_disc / n_total, 4),
        "n_dual_judge": n_dual,
        "dispute_by_et_template": {
            "E_t=%s/t%d" % (k[0], k[1]): {
                "n": v["n"], "n_disputed": v["n_disc"],
                "dispute_rate": round(v["n_disc"] / v["n"], 4),
            } for k, v in sorted(et_tab.items(), key=lambda kv: (kv[0][0], kv[0][1]))},
        "dispute_by_class": {
            c: {"n": v["n"], "n_disputed": v["n_disc"],
                "dispute_rate": round(v["n_disc"] / v["n"], 4),
                "judge_big_pos_rate": round(v["b_pos"] / v["n"], 4),
                "judge_small_pos_rate": round(v["s_pos"] / v["n"], 4),
                "qwen32_pos_rate": round(v["q_pos"] / v["n"], 4),
                } for c, v in sorted(cls_tab.items())},
        "stall_vs_nonstall_dispute": {
            "plot_stall": {"n": len(stall), "n_disputed": disc_stall},
            "non_stall": {"n": len(non_stall), "n_disputed": disc_non},
            "fisher": {"or": fish[0], "p": fish[1]},
        },
        "dispute_side_qwen32": {
            "agrees_judge_big": disc_qw_big,
            "agrees_judge_small": disc_qw_small,
            "agrees_neither": disc_qw_neither,
            "n_with_qwen32": disc_qw_big + disc_qw_small + disc_qw_neither,
        },
        "convergence_by_class": conv,
        "dispute_by_et": {
            "E_t=%s" % k: {"n": v["n"], "n_disputed": v["n_disc"],
                           "dispute_rate": round(v["n_disc"] / v["n"], 4)}
            for k, v in sorted(et_total.items())},
    }

    with open(out_dir / "s18_dispute_stall.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    md = render_md(out)
    (out_dir / "s18_dispute_stall.md").write_text(md, encoding="utf-8")
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "s18_dispute_stall.md").write_text(md, encoding="utf-8")
    _log("已落盘 s18_dispute_stall.json/.md")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def render_md(o):
    lines = [
        "# S18：dual_judge 争议 × 停滞/属性交叉分析（GPU1 · 2026-08-14）\n",
        "## 目的",
        "E2B 全量 dual_judge 争议率 %.1f%%（%d/%d）。检验争议是否聚集于停滞伪影区"
        "（E_t=1/t2），以及跨族收敛是否被停滞污染。\n" % (
            o["dispute_rate_overall"] * 100, o["n_disputed"], o["n_total"]),
        "## 争议率 by E_t × template",
        "| 组 | n | 争议 | 争议率 |",
        "|---|---|---|---|",
    ]
    for k, v in o["dispute_by_et_template"].items():
        lines.append("| %s | %d | %d | %.4f |" % (
            k, v["n"], v["n_disputed"], v["dispute_rate"]))
    lines.append("\n## 争议率 by 类别（含各评分器 pos_rate）")
    lines.append("| 类别 | n | 争议 | 争议率 | judge_big+ | judge_small+ | qwen32+ |")
    lines.append("|---|---|---|---|---|---|---|")
    for c, v in o["dispute_by_class"].items():
        lines.append("| %s | %d | %d | %.4f | %.4f | %.4f | %.4f |" % (
            c, v["n"], v["n_disputed"], v["dispute_rate"],
            v["judge_big_pos_rate"], v["judge_small_pos_rate"],
            v["qwen32_pos_rate"]))
    fs = o["stall_vs_nonstall_dispute"]
    lines.append("\n## 停滞 vs 非停滞 争议 Fisher")
    lines.append("- plot_stall: 争议 %d/%d；非停滞: 争议 %d/%d" % (
        fs["plot_stall"]["n_disputed"], fs["plot_stall"]["n"],
        fs["non_stall"]["n_disputed"], fs["non_stall"]["n"]))
    if fs["fisher"]["or"] is not None:
        lines.append("- Fisher OR=%.4f, p=%.6g %s" % (
            fs["fisher"]["or"], fs["fisher"]["p"],
            "→ 争议在停滞区显著聚集" if fs["fisher"]["p"] < 0.05
            else "→ 争议与停滞无显著关联"))
    dq = o["dispute_side_qwen32"]
    lines.append("\n## 争议行中 qwen32 站队（强异构锚点仲裁）")
    lines.append("- 同意 judge_big: %d；同意 judge_small: %d；都不一致: %d" % (
        dq["agrees_judge_big"], dq["agrees_judge_small"], dq["agrees_neither"]))
    lines.append("\n## 跨族收敛 by 类别（dual_judge vs qwen32）")
    lines.append("| 类别 | n | 一致率 | 判据≥0.80 |")
    lines.append("|---|---|---|---|")
    for c, v in o["convergence_by_class"].items():
        lines.append("| %s | %d | %.4f | %s |" % (
            c, v["n"], v["agreement_dual_vs_qwen32"],
            "PASS" if v["pass_0_80"] else "FAIL"))
    lines.append("\n## 判读")
    lines.append("> 若争议不随停滞/E_t=1 显著聚集，且各类别跨族收敛均 ≥0.80："
                 "dual_judge 口径在停滞伪影区稳健，主效应不受停滞污染。否则如实披露。")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
