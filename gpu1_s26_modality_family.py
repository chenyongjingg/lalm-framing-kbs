#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S26：模态效应 × 攻击族交互（CPU · 2026-08-14）。

动机：S24 确立 audio>>text 模态主效应（dual Δ≈+0.14，qwen32 Δ≈+0.21）。
S22 确立 N 效应跨攻击族泛化且仇恨言论最强。审稿人必问："音频放大是否跨攻击族
一致，还是仅由某一类攻击驱动？" 本实验把模态效应按攻击族分层（与 S22 同族映射），
回答泛化边界。

方法：
  - E4B responses（text+audio 快照）经 query_id → category 映射（queries_v2.jsonl，
    复用 S22._load_qfam，与 stage_p2_msrf C2 修复同逻辑）。
  - 每族：pos(text)/pos(styled_audio)、Δ(styled−text)、query 聚类 bootstrap 95%CI、
    Fisher OR+p（口径 dual_judge + qwen32）。
  - 每族：N 效应在 text 内 vs styled_audio 内（dual）——检验"叙事结构放大是否在
    音频侧更强且跨族成立"。
  - 覆盖缺口如实披露（P1-PILOT 150 查询覆盖 5/6 族，无网络攻击/良性请求）。

纪律：纯 CPU、零生成、只读缓存只写 s26_*；audio 为快照（E4B 生成中），
E4B 完成后重跑获得权威值。

用法：python gpu1_s26_modality_family.py [--B 1000] [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s26 %s] %s" % (Path(__file__).stem, m), flush=True)


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

    from gpu1_s22_attack_family import _load_qfam
    qfam = _load_qfam(root)
    _log("query→category 映射 %d 条" % len(qfam))

    rows = [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl",
        encoding="utf-8")]
    rows = [r for r in rows if r.get("query_id") in qfam]
    all_rows = [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl",
        encoding="utf-8")]
    _log("E4B 响应命中族映射 %d/%d" % (len(rows), len(all_rows)))
    miss = sum(1 for r in all_rows if r.get("query_id") not in qfam)
    _log("未命中（良性/无族）%d" % miss)

    # ---- 标签（rid 键控，S24 同款）----
    from gpu1_s24_modality_effect import _read_rid_cache
    rid = [r["response_id"] for r in rows]
    def _lbl(fn):
        d = _read_rid_cache(cache_dir / fn)
        return np.array([d.get(x, np.nan) for x in rid], dtype=float)
    jb = np.where([str(r.get("A_s", "")).endswith("audio") for r in rows],
                  _lbl("s17_e4b_audio_judge_big.jsonl"),
                  _lbl("s17_e4b_text_judge_big.jsonl"))
    js = np.where([str(r.get("A_s", "")).endswith("audio") for r in rows],
                  _lbl("s17_e4b_audio_judge_small.jsonl"),
                  _lbl("s17_e4b_text_judge_small.jsonl"))
    qw = np.where([str(r.get("A_s", "")).endswith("audio") for r in rows],
                  _lbl("s17_e4b_audio_qwen32.jsonl"),
                  _lbl("s17_e4b_text_qwen32.jsonl"))
    dual = np.where(~np.isnan(jb) & ~np.isnan(js) & (jb == js), jb, np.nan)
    for tag, arr in [("jb", jb), ("js", js), ("qw", qw), ("dual", dual)]:
        _log("label %s 非空=%d/%d" % (tag, int(np.sum(~np.isnan(arr))), len(rows)))

    from gpu1_s24_modality_effect import _modality_effect
    fams = sorted({qfam[r["query_id"]] for r in rows})

    out = {
        "stage": "S26",
        "date": "2026-08-14",
        "purpose": "模态效应（text vs styled_audio）× 攻击族分层；音频放大泛化边界",
        "snapshot": "audio 为 E4B 生成中快照，E4B 完成后重跑",
        "families_covered": fams,
        "families_missing": [x for x in
                             ("网络攻击", "良性请求") if x not in fams],
    }

    # ---- 1. 每族模态效应 ----
    _log("---- 1. 每族 模态效应（styled−text） ----")
    fam_mod = {}
    for s in ["dual", "qwen32"]:
        fam_mod[s] = {}
        for fam in fams:
            idx = [i for i, r in enumerate(rows) if qfam[r["query_id"]] == fam]
            sub_r = [rows[i] for i in idx]
            sub_l = {"dual": dual, "qwen32": qw}[s][idx]
            me = _modality_effect(sub_r, sub_l, "text", "styled_audio",
                                  B=B, name="%s_%s" % (s, fam))
            fam_mod[s][fam] = me
            _log("fam %s %s: pos(text)=%s pos(styled)=%s Δ=%s CI=%s" % (
                s, fam, me["pos0"], me["pos1"], me["delta"], me["ci95"]))
    out["family_modality_effect"] = fam_mod

    # ---- 2. 每族 N 效应：text 内 vs styled 内（dual）----
    _log("---- 2. 每族 N 效应（text vs styled，dual） ----")
    from gpu1_s24_modality_effect import _dim_effect_m
    fam_n = {}
    for fam in fams:
        fam_n[fam] = {}
        for lvl in ["text", "styled_audio"]:
            idx = [i for i, r in enumerate(rows)
                   if qfam[r["query_id"]] == fam and r.get("A_s") == lvl]
            sub_r = [rows[i] for i in idx]
            sub_l = dual[idx]
            d = _dim_effect_m(sub_r, sub_l, "N", 0, 1, B=B,
                              name="N_%s_%s" % (fam, lvl))
            fam_n[fam][lvl] = d
            _log("famN %s %s: Δ=%s CI=%s" % (fam, lvl, d["delta"], d["ci95"]))
    out["family_N_by_modality"] = fam_n

    if args.smoke:
        print(json.dumps(_json_safe(out), ensure_ascii=False, indent=2))
        _log("smoke 完成（不落盘）")
        return 0

    with open(out_dir / "s26_modality_family.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(out), f, ensure_ascii=False, indent=2)
    _render_md(out, report_dir / "s26_modality_family.md")
    _log("已落盘 s26_modality_family.json + report/s26_modality_family.md")
    return 0


def _render_md(o, path):
    L = ["# S26：模态效应 × 攻击族（GPU1 · 2026-08-14）\n",
         "\n> audio 为 E4B 生成中快照，**非权威值**；E4B 完成后重跑。覆盖 %d 族，未覆盖 %s。\n"
         % (len(o["families_covered"]), "、".join(o["families_missing"])),
         "\n## 1. 每族模态效应（styled_audio − text）\n",
         "| 口径 | 攻击族 | n | pos(text) | pos(styled) | Δ | 95%CI | Fisher OR(p) |\n",
         "|---|---|---|---|---|---|---|---|\n"]
    for s, fm in o["family_modality_effect"].items():
        for fam, me in fm.items():
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |\n" % (
                s, fam, me["n0"] + me["n1"], me["pos0"], me["pos1"],
                me["delta"], me["ci95"],
                ("%s(%s)" % (me["or_fisher"], me["p_fisher"]))
                if me["or_fisher"] is not None else "-"))
    L.append("\n## 2. 每族 N 效应（text 内 vs styled 内，dual 口径）\n")
    L.append("| 攻击族 | 模态 | n0 | n1 | pos0 | pos1 | Δ | 95%CI |\n")
    L.append("|---|---|---|---|---|---|---|---|\n")
    for fam, md in o["family_N_by_modality"].items():
        for lvl, d in md.items():
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |\n" % (
                fam, lvl, d["n0"], d["n1"], d["pos0"], d["pos1"],
                d["delta"], d["ci95"]))
    L.append("\n## 判读\n")
    L.append("> 音频放大与 N 效应的跨族方向一致性、族间量级差异（如仇恨言论是否在音频侧更甚）、"
             "以及覆盖边界均如实披露。\n")
    path.write_text("".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
