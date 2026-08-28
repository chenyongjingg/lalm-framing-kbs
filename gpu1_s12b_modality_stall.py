#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S12b：模态 × E_t × template 停滞权威交叉表（CPU，2026-08-14）。

动机：S10 只报告音频停滞（E_t=1 10.8% vs E_t=0 0.64%，OR≈19）。文本模态
（E_t=1 21.78%、OR=71.31、集中在 template t2=56.3%）是操纵提示词歧义缺陷的
更强证据，且是模板伪影。本脚本给出可复现的权威交叉表，作论文补充材料。

注意：E4B 文本 3600 已 100% 完成（权威）；音频仍在生成（本文记录当前快照
行数与生成状态，E4B 完成后可重跑获得音频权威表）。只读 E4B jsonl，零写入
主账本；输出 results/gpu1_pipeline/s12b_modality_stall.{json,md} + report/。

用法：python gpu1_s12b_modality_stall.py
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu1_s10b as s10b  # noqa: E402  (classify)

CLASSES = ("plot_stall", "data_clarify", "refusal", "other")


def _log(m):
    print("[s12b] %s" % m, flush=True)


def build(rows):
    """rows -> {modality: {E_t: {template: {cls: n}, total}}}"""
    tab = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))))
    for r in rows:
        mod = "text" if r["A_s"] == "text" else "audio"
        tab[mod][r["E_t"]][r["template_idx"]][s10b.classify(r)] += 1
    return tab


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
    n_text = sum(1 for r in rows if r["A_s"] == "text")
    n_audio = len(rows) - n_text
    _log("E4B 总行=%d（text=%d, audio=%d）" % (len(rows), n_text, n_audio))

    tab = build(rows)

    def summ(mod, et, cls):
        return sum(tab[mod][et][t][cls] for t in (0, 1, 2))

    overview = {"stage": "S12b", "date": "2026-08-14",
                "data": "gemma_4_e4b_responses.jsonl",
                "rows_total": len(rows), "rows_text": n_text,
                "rows_audio_snapshot": n_audio,
                "audio_authoritative": False,
                "note_audio": ("音频仍在生成（文本已 100% 完成）。本表音频列为 "
                               "当前快照，E4B 完成后重跑本脚本获得权威值。"),
                "by_modality": {}}
    for mod in ("text", "audio"):
        m = {"n_rows": n_text if mod == "text" else n_audio,
             "by_et": {}}
        for et in (0, 1):
            total = sum(tab[mod][et][t][c] for t in (0, 1, 2)
                        for c in CLASSES)
            stall = summ(mod, et, "plot_stall")
            e = {"n": total, "plot_stall": stall,
                 "plot_stall_rate": round(stall / total, 4) if total else None,
                 "by_template": {}}
            for t in (0, 1, 2):
                tt = sum(tab[mod][et][t][c] for c in CLASSES)
                e["by_template"]["t%d" % t] = {
                    "n": tt,
                    "counts": {c: tab[mod][et][t][c] for c in CLASSES},
                    "plot_stall_rate": round(
                        tab[mod][et][t]["plot_stall"] / tt, 4) if tt else None}
            m["by_et"]["E_t=%d" % et] = e
        # OR：plot_stall E_t1 vs E_t0
        n0, n1 = summ(mod, 0, "plot_stall"), summ(mod, 1, "plot_stall")
        t0, t1 = summ(mod, 0, "plot_stall") + 0, \
            summ(mod, 1, "plot_stall") + 0
        tot0 = sum(tab[mod][0][t][c] for t in (0, 1, 2) for c in CLASSES)
        tot1 = sum(tab[mod][1][t][c] for t in (0, 1, 2) for c in CLASSES)
        or_val = None
        if tot0 > n0 and tot1 > n1 and n0 > 0 and n1 > 0:
            or_val = round((n1 / (tot1 - n1)) / (n0 / (tot0 - n0)), 2)
        m["odds_ratio_Et1_vs_Et0"] = or_val
        overview["by_modality"][mod] = m
        _log("%s: E_t0 stall=%.4f, E_t1 stall=%.4f, OR=%s" % (
            mod, n0 / tot0, n1 / tot1, or_val))

    with open(out_dir / "s12b_modality_stall.json", "w",
              encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    md = render_md(overview)
    (out_dir / "s12b_modality_stall.md").write_text(md, encoding="utf-8")
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "s12b_modality_stall.md").write_text(md, encoding="utf-8")
    _log("已落盘 s12b_modality_stall.json/.md")

    print(json.dumps({"stage": "S12b",
                      "text": {k: {"plot_stall_rate": v["plot_stall_rate"],
                                   "OR": overview["by_modality"]["text"]
                                   ["odds_ratio_Et1_vs_Et0"]}
                               for k, v in
                               overview["by_modality"]["text"]["by_et"].items()},
                      "audio": {k: {"plot_stall_rate": v["plot_stall_rate"],
                                    "OR": overview["by_modality"]["audio"]
                                    ["odds_ratio_Et1_vs_Et0"]}
                                for k, v in
                                overview["by_modality"]["audio"]["by_et"].items()}},
                     ensure_ascii=False, indent=2))
    return 0


def render_md(o):
    lines = [
        "# S12b：模态 × E_t × template 停滞权威交叉表（GPU1 · 2026-08-14）\n",
        "## 数据",
        "- 源：%s（总行 %d：text %d，audio 快照 %d）" % (
            o["data"], o["rows_total"], o["rows_text"], o["rows_audio_snapshot"]),
        "- 文本已 100%% 完成（权威）；音频仍在生成（%s）" % o["note_audio"],
        "> 停滞分类器：gpu1_s10b.classify（plot_stall / data_clarify / refusal / other），"
        "与 S10 同源。\n",
    ]
    for mod in ("text", "audio"):
        m = o["by_modality"][mod]
        lines.append("## %s 模态（n=%d）" % (mod, m["n_rows"]))
        lines.append("| E_t | template | plot_stall | data_clarify | refusal | other | n |")
        lines.append("|---|---|---|---|---|---|---|")
        for et in ("E_t=0", "E_t=1"):
            e = m["by_et"][et]
            for t in ("t0", "t1", "t2"):
                b = e["by_template"][t]
                lines.append("| %s | %s | **%d (%.4f)** | %d | %d | %d | %d |" % (
                    et, t, b["counts"]["plot_stall"], b["plot_stall_rate"],
                    b["counts"]["data_clarify"], b["counts"]["refusal"],
                    b["counts"]["other"], b["n"]))
            lines.append("| **%s 合计** | | **%d (%.4f)** | | | | %d |" % (
                et, e["plot_stall"], e["plot_stall_rate"], e["n"]))
        lines.append("- **OR(E_t=1 vs E_t=0, plot_stall) = %s**" % m["odds_ratio_Et1_vs_Et0"])
        lines.append("")
    lines += [
        "## 判读",
        "- 文本模态：E_t=1 停滞 21.78%%（392/1800）vs E_t=0 0.39%%（7/1800），"
        "OR=71.31 —— 操纵提示词歧义缺陷在文本上比音频（OR≈19）更强。",
        "- **模板异质性**：停滞集中于 t2（'叙述一段情节后回答'，56.3%%），"
        "t0=6.2%%、t1=2.8%% → 该缺陷是【模板伪影】（特定 prompt 措辞诱导模型索取"
        "情节），而非模型行为差异。这强化缺陷归因的因果性。",
        "- 与 S10b（音频澄清）、S12（文本澄清）结合：澄清提示词消除停滞，"
        "缺陷可修复；原始 ASR 存在向下偏差，论文需披露并按偏差量级处理。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
