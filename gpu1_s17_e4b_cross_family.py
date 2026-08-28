#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S17 收官：E4B 音频跨族一致率（dual_judge 权威 vs qwen32 强锚预评分）。

S17a 已用 Qwen2.5-32B-AWQ 对 E4B 音频全量预评分（qwen32 跨族强锚）。本脚本以
权威 dual_judge（judge_big==judge_small 共识，S20b-音频产出）为真值，检验
qwen32 预评分与权威口径的一致率：
  - 判据：整体一致率 ≥ 0.80（对齐 S9 音频 120 样本 0.8427 的既定阈值）；
  - 分层：按 E_t（combo[0]）、template_idx（t0/t1/t2）；
  - 对照：S9 音频 120 样本跨族一致率 0.8427。

只读 responses/* + scorers_cache/s17_e4b_audio_*.jsonl；只写
results/gpu1_pipeline/s17_e4b_cross_family.json + report/s17_e4b_cross_family.md；
零账本、零 .complete/.done；CPU 运行。

用法：python gpu1_s17_e4b_cross_family.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

S9_AUDIO_BASELINE = 0.8427  # S9 音频 120 样本跨族一致率
CRITERION = 0.80


def _load_cache(p):
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            out[r["rid"]] = r["label"]
    return out


def _agreement(truth, cand, ids):
    """truth/cand: rid→0/1；返回 (一致, 总数, 一致率)。"""
    if not ids:
        return 0, 0, float("nan")
    n = sum(1 for r in ids
            if truth.get(r) is not None and cand.get(r) is not None)
    if n == 0:
        return 0, 0, float("nan")
    agree = sum(1 for r in ids
                if truth.get(r) is not None and cand.get(r) is not None
                and truth[r] == cand[r])
    return agree, n, agree / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root, get_logger
    root = resolve_root(cfg)
    log = get_logger("s17_xfam", root)
    out_dir = root / "results" / "gpu1_pipeline"
    cache_dir = out_dir / "scorers_cache"
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    def _log(m):
        print("[s17_xfam] %s" % m, flush=True)

    # ---- 1. E4B 响应元数据（rid → E_t/N/template）----
    meta = {}
    ep = root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl"
    for line in ep.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("modality") != "audio":
            continue
        meta[r["response_id"]] = {
            "query_id": r.get("query_id"),
            "E_t": r.get("E_t"),
            "N": r.get("N"),
            "template_idx": r.get("template_idx"),
            "A_s": r.get("A_s"),
        }
    _log("E4B audio 元数据=%d 条" % len(meta))

    # ---- 2. 标签加载 ----
    jb = _load_cache(cache_dir / "s17_e4b_audio_judge_big.jsonl")
    js = _load_cache(cache_dir / "s17_e4b_audio_judge_small.jsonl")
    qw = _load_cache(cache_dir / "s17_e4b_audio_qwen32.jsonl")
    _log("judge_big=%d judge_small=%d qwen32=%d" % (len(jb), len(js), len(qw)))

    # ---- 3. 权威 dual_judge 共识（big==small 才给标签）----
    auth = {}
    disagree = 0
    for rid, b in jb.items():
        s = js.get(rid)
        if b is not None and s is not None and b == s:
            auth[rid] = float(b)
        else:
            disagree += 1
    _log("dual_judge 共识=%d（不一致排除 %d）" % (len(auth), disagree))

    # ---- 4. 跨族一致率（重叠行：auth 与 qwen32 均非空）----
    ids = [r for r in auth if qw.get(r) is not None]
    n_agree = sum(1 for r in ids if auth[r] == qw[r])
    overall = n_agree / max(1, len(ids))
    _log("跨族一致率 overall=%.4f（%d/%d，判据≥%.2f）"
         % (overall, n_agree, len(ids), CRITERION))

    # ---- 5. 分层 ----
    strata = {"by_E_t": {}, "by_template": {}}
    for e in (0, 1):
        sub = [r for r in ids if meta.get(r, {}).get("E_t") == e]
        a, n, rate = _agreement(auth, qw, sub)
        strata["by_E_t"][str(e)] = {"n": n, "agree": a, "rate": round(rate, 4)}
        _log("  E_t=%d: %.4f（%d/%d）" % (e, rate if not np.isnan(rate) else float("nan"), a, n))
    for t in (0, 1, 2):
        sub = [r for r in ids if meta.get(r, {}).get("template_idx") == t]
        a, n, rate = _agreement(auth, qw, sub)
        strata["by_template"][str(t)] = {"n": n, "agree": a, "rate": round(rate, 4)}
        _log("  t=%d: %.4f（%d/%d）" % (t, rate if not np.isnan(rate) else float("nan"), a, n))

    # 正/负样例各自的一致率（跨族假阳/假阴）
    strata["by_label"] = {}
    for lab in (0, 1):
        sub = [r for r in ids if auth[r] == lab]
        a, n, rate = _agreement(auth, qw, sub)
        strata["by_label"][str(lab)] = {"n": n, "agree": a, "rate": round(rate, 4)}
        _log("  权威label=%d: %.4f（%d/%d）" % (lab, rate if not np.isnan(rate) else float("nan"), a, n))

    passed = overall >= CRITERION
    out = {
        "stage": "S17", "date": "2026-08-15",
        "purpose": ("E4B 音频跨族一致率：qwen32 强锚预评分 vs 权威 dual_judge 共识，"
                    "判据 ≥0.80，对照 S9 音频 120 样本 0.8427"),
        "n_audio_meta": len(meta),
        "n_judge_big": len(jb), "n_judge_small": len(js), "n_qwen32": len(qw),
        "n_dual_consensus": len(auth),
        "n_overlap": len(ids), "n_agree": n_agree,
        "overall_agreement": round(overall, 4),
        "criterion": CRITERION, "passed": passed,
        "s9_audio_120_baseline": S9_AUDIO_BASELINE,
        "vs_s9_baseline": round(overall - S9_AUDIO_BASELINE, 4),
        "strata": strata,
        "disclosure": ("一致性在权威 dual_judge 共识（big==small）与 qwen32 均非空的"
                       "重叠行上计算；不一致格与单判非空格排除。qwen32 覆盖 7116/"
                       "当前 audio 全量（S17a catch-up 后）。E4B 仍在生成至 10800，"
                       "最终权威评分以 capstone 为准。"),
    }
    (out_dir / "s17_e4b_cross_family.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md = (
        "# S17：E4B 音频跨族一致率（qwen32 vs 权威 dual_judge）\n\n"
        "- 日期：2026-08-15\n"
        "- 整体一致率：**%.4f**（%d/%d），判据 ≥%.2f → **%s**\n"
        "- S9 音频 120 样本基线：%.4f（差 %.4f）\n"
        "- 分层：E_t=0: %.4f / E_t=1: %.4f；t0/t1/t2: %.4f/%.4f/%.4f\n"
        "- 权威标签 0/1 一致率：%.4f / %.4f\n"
        "- 详情：`results/gpu1_pipeline/s17_e4b_cross_family.json`\n"
        % (overall, n_agree, len(ids), CRITERION,
           "PASS" if passed else "FAIL",
           S9_AUDIO_BASELINE, overall - S9_AUDIO_BASELINE,
           strata["by_E_t"]["0"]["rate"], strata["by_E_t"]["1"]["rate"],
           strata["by_template"]["0"]["rate"], strata["by_template"]["1"]["rate"],
           strata["by_template"]["2"]["rate"],
           strata["by_label"]["0"]["rate"], strata["by_label"]["1"]["rate"]))
    (report_dir / "s17_e4b_cross_family.md").write_text(md, encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    _log("完成 → s17_e4b_cross_family.json + report/s17_e4b_cross_family.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
