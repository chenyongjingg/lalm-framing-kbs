#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S36：S35 强制解码协议补全——S17/S33 全量格 + 剩余 null 打标。

S35（2026-08-15）已在 E2B 主链 + S28 验证强制解码协议：
首 token argmax(logits('0'), logits('1')) vs 自由生成 一致率 99.52%/99.92%，
null 格是低置信格（|margin| 显著更小）。本脚本把同一协议扩展到流水线其余 3 个
scope，使 S34 审计表的全部 judge_small null 格具备强制标签：

  - S17 E4B 音频  7200 格（cache s17_e4b_audio_judge_small）
  - S17 E4B 文本  3600 格（cache s17_e4b_text_judge_small）
  - S33 异族音频   344 格（cache s33_hetero_audio_judge_small，rid 带 s33_ 前缀）

产出：
  1) 每个 scope 全量格：forced vs freegen 一致率 + E_t/N 分层 + |margin| 诊断
     （协议稳定性全量验证）；
  2) 剩余 45 条 null 的强制标签 + 独立评分器对照（judge_big/qwen32/
     strongreject/harmbench）+ null 格 |margin| vs 总体（低置信本质刻画）；
  3) 独立新文件 s36_forced_complete_labels.jsonl（全格，未改写任何生产缓存）。

纪律：零人工标注、零账本、不写 .complete/.done；CUDA_VISIBLE_DEVICES=1 调用方
注入；GPU1 串行检测（被占退出）；只读 responses/* + scorers_cache/*。

用法：CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s36_forced_complete.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu1_s9_cross_family as s9  # noqa: E402 SCORER_FACTORIES/register_scorers


def _gpu1_busy(min_free_mib=8192):
    """GPU1（index 1）剩余显存是否不足。主链在 GPU1 有 ~4.8GB 常驻，
    不阻塞小评分器；真正的冲突是 GPU1 free < 8GB。"""
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
    """rid 制缓存：{"rid": "s28_...", "label": ...} → {response_id: label}
    （归一化 s28_ 前缀；s33_ 前缀由调用方 rid_fn 注入）。"""
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            d = json.loads(line)
            r = d["rid"]
            out[r[4:] if r.startswith("s28_") else r] = d.get("label")
    return out


def _forced_verdict(sc, behavior, generation):
    """首 token 强制解码：argmax(logits(token"0"), logits(token"1"))。
    返回 (label, s0, s1)，s0/s1 为两个 token 的原始 score（logits 或 log-prob，
    argmax 不变）。协议与 S35 完全一致。"""
    prompt = sc._make_prompt(behavior, generation)
    text = sc._apply_template(prompt)
    inputs = sc.processor(text=text, return_tensors="pt", truncation=True,
                          max_length=4096)
    inputs = {k: v.to(sc.model.device) if hasattr(v, "to") else v
              for k, v in inputs.items()}
    out = sc.model.generate(
        **inputs, max_new_tokens=1, do_sample=False,
        pad_token_id=sc.processor.tokenizer.pad_token_id,
        output_scores=True, return_dict_in_generate=True)
    tok = sc.processor.tokenizer
    id0 = tok.convert_tokens_to_ids("0")
    id1 = tok.convert_tokens_to_ids("1")
    if id0 is None or id1 is None:
        raise RuntimeError("tokenizer 无单字 token '0'/'1'")
    s = None
    for attr in ("scores", "logits"):
        val = getattr(out, attr, None)
        if val is not None and len(val) > 0:
            s = val[0][0]  # [vocab]
            break
    if s is None:
        raise RuntimeError("generate 未返回 scores/logits")
    s0 = float(s[id0].item())
    s1 = float(s[id1].item())
    return int(s1 >= s0), s0, s1


def _pairwise(lab1, lab2, ids):
    """在 ids 上比较两个 binary label dict（None 跳过）→ 一致率/κ/Spearman。"""
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


def main():
    cfg = yaml.safe_load(open("pipeline_config.yaml", encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    cache_dir = out_dir / "scorers_cache"
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    def _log(m):
        print("[s36] %s" % m, flush=True)

    # ---- 0. GPU1 显存检测 ----
    if _gpu1_busy():
        _log("GPU1 剩余显存不足，退出（串行纪律）")
        return 1
    _log("GPU1 空闲，开始")

    # ---- 1. 加载 judge_small（E2B）评分器 ----
    s9.register_scorers(cfg)
    _log("加载 judge_small (Gemma-4-E2B) 评分器 ...")
    sc = s9.SCORER_FACTORIES["judge_small"]()
    _log("评分器就绪")

    results = {"stage": "S36", "date": "2026-08-16",
               "purpose": ("S35 强制解码协议扩展到 S17/S33 全量格：协议稳定性 "
                           "全量验证 + 剩余 judge_small null 补全"),
               "method": ("与 S35 相同：GemmaJudgeScorer 同一 prompt 模板；"
                          "generate(max_new_tokens=1, output_scores=True) → "
                          "argmax(logits('0'), logits('1'))")}

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

        js = _load_rid(cache_dir / (key + "_judge_small.jsonl"))
        jb = _load_rid(cache_dir / (key + "_judge_big.jsonl"))
        others = {}
        for suf in ("qwen32", "strongreject", "harmbench"):
            p = cache_dir / (key + "_" + suf + ".jsonl")
            if p.exists():
                others[suf] = _load_rid(p)
        _log("  judge_small 覆盖=%d null=%d" % (
            sum(1 for r in rids if js.get(r) is not None),
            sum(1 for r in rids if js.get(r) is None)))

        # ---- 全量格强制解码 ----
        forced, margin = {}, {}
        for k, r in enumerate(rows):
            lab, s0, s1 = _forced_verdict(sc, r.get("prompt") or "",
                                          str(r.get("response") or ""))
            rid = rid_fn(r)
            forced[rid] = lab
            margin[rid] = round(s1 - s0, 4)
            if (k + 1) % 1500 == 0:
                _log("  %s 强制解码 %d/%d" % (key, k + 1, n))

        scored = [r for r in rids if js.get(r) is not None]
        agree_raw = sum(1 for r in scored if forced[r] == js[r])
        res = {"n_cells": n,
               "freegen_covered": len(scored),
               "freegen_null": n - len(scored),
               "forced_all": n,
               "forced_vs_freegen": _pairwise(forced, js, rids),
               "forced_vs_freegen_agreement_raw":
                   round(agree_raw / max(1, len(scored)), 4),
               "forced_vs_judge_big": _pairwise(forced, jb, rids),
               "strata": {}}
        for e in (0, 1):
            sub = [r for r in scored if by_rid[r]["E_t"] == e]
            res["strata"]["E_t=%d" % e] = _pairwise(forced, js, sub)
        for nn in (0, 1):
            sub = [r for r in scored if by_rid[r]["N"] == nn]
            res["strata"]["N=%d" % nn] = _pairwise(forced, js, sub)

        cons = np.array([abs(margin[r]) for r in scored
                         if forced[r] == js[r]])
        disc = np.array([abs(margin[r]) for r in scored
                         if forced[r] != js[r]])
        res["abs_margin_agree"] = {
            "n": int(len(cons)), "mean": round(float(cons.mean()), 4),
            "median": round(float(np.median(cons)), 4)}
        res["abs_margin_disagree"] = {
            "n": int(len(disc)), "mean": round(float(disc.mean()), 4),
            "median": round(float(np.median(disc)), 4)}

        # ---- null 格画像 ----
        nulls = [r for r in rids if js.get(r) is None]
        np_ = {"n": len(nulls),
               "forced_label_dist": {
                   "0": sum(1 for r in nulls if forced[r] == 0),
                   "1": sum(1 for r in nulls if forced[r] == 1)},
               "N_dist": {
                   "0": sum(1 for r in nulls if by_rid[r]["N"] == 0),
                   "1": sum(1 for r in nulls if by_rid[r]["N"] == 1)}}
        for name, lab in (("judge_big", jb),) + tuple(others.items()):
            ag = sum(1 for r in nulls
                     if lab.get(r) is not None and lab[r] == forced[r])
            de = sum(1 for r in nulls if lab.get(r) is not None)
            np_["forced_vs_%s" % name] = {
                "agree": ag, "denom": de,
                "rate": round(ag / de, 4) if de else None}
        jb_harm = sum(1 for r in nulls if jb.get(r) == 1)
        np_["judge_big_harmful_rate_on_nulls"] = round(
            jb_harm / max(1, len(nulls)), 4)
        pop_m = np.array([abs(margin[r]) for r in rids])
        null_m = np.array([abs(margin[r]) for r in nulls])
        np_["null_margin_diag"] = {
            "pop_median": round(float(np.median(pop_m)), 4),
            "null_median": (round(float(np.median(null_m)), 4)
                            if len(null_m) else None),
            "null_below5": int((null_m < 5).sum()),
            "n": int(len(null_m))}
        res["null_profile"] = np_

        results[key] = res
        for r in rows:
            rid = rid_fn(r)
            all_label_lines.append({
                "scope": key, "rid": rid, "label": forced[rid],
                "margin": margin[rid], "is_null": js.get(rid) is None})
        _log("  %s 完成：forced vs freegen=%.4f；null=%d 已打标" % (
            key, res["forced_vs_freegen_agreement_raw"], len(nulls)))

    # ---- 汇总 ----
    sc.close()
    _log("评分器已释放")
    filled = sum(results[k]["null_profile"]["n"]
                 for k in ("s17_e4b_audio", "s17_e4b_text", "s33_hetero_audio"))
    results["summary"] = {
        "scopes": ["s17_e4b_audio", "s17_e4b_text", "s33_hetero_audio"],
        "nulls_filled_by_forced_in_this_stage": filled,
        "note": ("E2B 主链 27 + S28 11 已在 S35 打标；本阶段补齐 S17 音频/S17 "
                 "文本/S33 共 %d 条。null 总数以各 scope 实际缓存为准。" % filled)}
    results["disclosure"] = (
        "与 S35 相同：强制解码=首 token argmax，独立评分模式；一致率=协议稳定性"
        "判据。标签写独立新文件 s36_forced_complete_labels.jsonl，未改写任何生产"
        "缓存。rid 制缓存已归一（s28_/s33_ 前缀）。")

    (out_dir / "s36_forced_complete.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "s36_forced_complete_labels.jsonl").open(
            "w", encoding="utf-8") as f:
        for x in all_label_lines:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # ---- 报告 ----
    lines = ["# S36：强制解码协议补全（S17/S33 全量 + 剩余 null 打标）",
             "",
             "- 日期：2026-08-16；方法：同 S35（首 token argmax(logits('0'),logits('1'))）",
             "- 结果 json：`results/gpu1_pipeline/s36_forced_complete.json`",
             "- 标签文件：`results/gpu1_pipeline/s36_forced_complete_labels.jsonl`\n"]

    def _row(d):
        if not d:
            return "N/A"
        return "%d | %.3f | %.3f | %.3f" % (d["n"], d["agreement"],
                                            d["kappa"], d["spearman"])

    lines += ["## 1. 全量格 协议稳定性（forced vs freegen）\n",
              "| scope | n | null | 一致率 | κ | Spearman | 一致格|m|med | 分歧格|m|med |",
              "|---|---|---|---|---|---|---|---|"]
    for k in ("s17_e4b_audio", "s17_e4b_text", "s33_hetero_audio"):
        r = results[k]
        pw = r["forced_vs_freegen"]
        kappa = pw["kappa"] if pw else float("nan")
        rho = pw["spearman"] if pw else float("nan")
        lines.append("| %s | %d | %d | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
            k, r["n_cells"], r["freegen_null"],
            r["forced_vs_freegen_agreement_raw"], kappa, rho,
            r["abs_margin_agree"]["median"],
            r["abs_margin_disagree"]["median"]))
    lines.append("")
    for k in ("s17_e4b_audio", "s17_e4b_text", "s33_hetero_audio"):
        r = results[k]
        parts = ["分层 %s：" % k]
        for sk, d in r["strata"].items():
            parts.append("%s=%s" % (sk, _row(d)))
        lines.append("  ".join(parts))
    lines.append("")

    lines += ["## 2. null 格画像（强制打标 + 独立评分器对照）\n",
              "| scope | null n | forced 0/1 | judge_big 对null有害率 | 对null |m|med | 总体 |m|med | null<5 |",
              "|---|---|---|---|---|---|---|---|"]
    for k in ("s17_e4b_audio", "s17_e4b_text", "s33_hetero_audio"):
        np_ = results[k]["null_profile"]
        nm = np_["null_margin_diag"]
        lines.append("| %s | %d | %d/%d | %.3f | %s | %s | %d |" % (
            k, np_["n"], np_["forced_label_dist"]["0"],
            np_["forced_label_dist"]["1"],
            np_["judge_big_harmful_rate_on_nulls"],
            ("%.4f" % nm["null_median"]) if nm["null_median"] is not None else "N/A",
            ("%.4f" % nm["pop_median"]), nm["null_below5"]))
    lines.append("")
    for k in ("s17_e4b_audio", "s17_e4b_text", "s33_hetero_audio"):
        np_ = results[k]["null_profile"]
        for name, d in np_.items():
            if name.startswith("forced_vs_"):
                lines.append("null 格 %s forced vs %s：%d/%d（%.3f）" % (
                    k, name[len("forced_vs_"):], d["agree"], d["denom"],
                    d["rate"] if d["rate"] is not None else float("nan")))
    lines += ["", "## 3. 汇总",
              "",
              "- 本阶段补全 null：%d 条（S17 音频 + S17 文本 + S33）" % filled,
              "- 与 S35（E2B 主链 27 + S28 11）合并后，S34 审计表全部 null 均具强制标签",
              "- 标签文件：`results/gpu1_pipeline/s36_forced_complete_labels.jsonl`",
              "  （全格；未改写任何生产缓存）",
              "- 详情：`results/gpu1_pipeline/s36_forced_complete.json`"]
    (report_dir / "s36_forced_complete.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    _log("完成 → s36_forced_complete.json + s36_forced_complete_labels.jsonl + "
         "report/s36_forced_complete.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
