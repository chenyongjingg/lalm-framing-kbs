#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S35：judge_small 强制解码（forced-verdict）验证 + null 格打标。

背景（2026-08-15，S34 系统性审计）：judge_small 全流水线 72 条 null 根因 =
E2B 对叙事型响应先输出英文解说（"The user request is: ..."），8-token 预算耗尽前
未输出判定数字 → `_parse_label` 的 `\\b([01])\\b` 匹配失败 → label=None。
已证明放宽 token 预算重评**不可靠**（E2B 散文论证与尾部数字互相矛盾）。本脚本用
**首 token 强制解码**补上测量缺口，作为独立的第 2 种评分模式：

  - 复用 GemmaJudgeScorer 同一 prompt 模板（GEMMA_JUDGE_RUBRIC +
    enable_thinking:false），与自由生成逐字相同；
  - `model.generate(max_new_tokens=1, output_scores=True)` 取首个生成 token 在
    token_id("0") / token_id("1") 上的 logits argmax → 强制判定（Llama2 式
    0/1 强制评分协议）；
  - 先在已打分格上验证 强制解码 vs 自由生成 一致率（协议稳定性判据），再对
    null 格打标，并与独立评分器对照（E2B 主链：judge_big/qwen32/cross_check_e2b；
    S28：judge_big/harmbench/strongreject）；
  - 报告 margin（s1-s0）在 一致格 vs 分歧格 上的分布，检验分歧是否集中在
    低置信（margin 小）的格子上；
  - 追加 S34 敏感性第 6 场景：E2B 主链 judge_small null 用强制标签填充后，
    重算 N/E_t 效应，与排除口径对照。

覆盖：E2B 主链全量 3600 格 + S28 全量 1200 格（~40min，GPU1 30h 窗口内）。

纪律：
  - 零人工标注、零账本、不写 .complete/.done；CUDA_VISIBLE_DEVICES=1 由调用方注入；
  - GPU1 串行纪律：启动前检测 GPU1 剩余显存，被占退出；
  - 只读 responses/* + scorers_cache/*；**绝不改写 judge_small.jsonl 等生产缓存**
    （强制标签写独立新文件 s35_forced_verdict_*_labels.jsonl）；
  - 只写 results/gpu1_pipeline/ + report/。

用法：CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s35_forced_verdict.py [--config pipeline_config.yaml]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu1_s9_cross_family as s9  # noqa: E402 SCORER_FACTORIES/register_scorers
import gpu1_s28_hetero_audio as s28  # noqa: E402 _bootstrap_pair


def _gpu1_busy(min_free_mib=8192):
    """GPU1（index 1）剩余显存是否不足。主链 505447 在 GPU1 有 ~4.8GB 常驻，
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


def _load_indexed(p):
    """index 制缓存：{"i": 0, "label": 0} → {0: 0}"""
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            d = json.loads(line)
            out[d["i"]] = d.get("label")
    return out


def _load_rid(p):
    """rid 制缓存：{"rid": "s28_...", "label": ...} → {response_id: label}
    （归一化 s28_ 前缀，与 S28 一致口径）。"""
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
    argmax 不变）。"""
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
    # 首 token 的 logits（生成第一个新 token 前的预测分布）。max_new_tokens=1
    # 时 out.scores[0] 与 out.logits[0] 等价，双属性兜底。
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


def _fmt_eff(eff):
    if not eff:
        return "N/A"
    return "%.4f [%s,%s] %s" % (eff["effect"], eff["ci95"][0], eff["ci95"][1],
                                "✓" if eff["excl_zero"] else "✗")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    cache_dir = out_dir / "scorers_cache"
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    def _log(m):
        print("[s35] %s" % m, flush=True)

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

    B = cfg.get("seeds", {}).get("bootstrap", 2000)
    seed = cfg.get("seeds", {}).get("bootstrap", 42)
    results = {"stage": "S35", "date": "2026-08-15",
               "purpose": ("judge_small 强制解码验证 + null 格打标：首 token 0/1 "
                           "argmax 与自由生成一致率（协议稳定性）→ null 格强制标签 "
                           "与独立评分器对照 + N/E_t 效应第 6 敏感性场景"),
               "method": ("GemmaJudgeScorer 同一 prompt；generate(max_new_tokens=1,"
                          " output_scores=True)；argmax(logits('0'), logits('1'))")}

    # ============ A. E2B 主链（index 制，3600 格） ============
    _log("=== A. E2B 主链 ===")
    e2b_rows = [json.loads(l) for l in
                (root / "responses" / "P1_PILOT" /
                 "gemma_4_e2b_responses.jsonl").open(encoding="utf-8")]
    n_main = len(e2b_rows)
    js = _load_indexed(cache_dir / "judge_small.jsonl")
    jb = _load_indexed(cache_dir / "judge_big.jsonl")
    qw = _load_indexed(cache_dir / "qwen32.jsonl")
    xc = _load_indexed(cache_dir / "cross_check_e2b.jsonl")
    assert len(js) >= n_main, "judge_small 缓存与响应数不一致"
    _log("E2B 主链响应=%d 格，freegen 覆盖=%d（null=%d）" % (
        n_main, sum(1 for i in range(n_main) if js.get(i) is not None),
        sum(1 for i in range(n_main) if js.get(i) is None)))

    forced = {}   # i -> label
    margin = {}   # i -> s1-s0
    for i, r in enumerate(e2b_rows):
        lab, s0, s1 = _forced_verdict(sc, r.get("prompt") or "",
                                      str(r.get("response") or ""))
        forced[i] = lab
        margin[i] = round(s1 - s0, 4)
        if (i + 1) % 500 == 0:
            _log("E2B 主链强制解码 %d/%d" % (i + 1, n_main))

    # 已打分格：强制 vs 自由生成 一致率
    scored = [i for i in range(n_main) if js.get(i) is not None]
    agree_ok = sum(1 for i in scored if forced[i] == js[i])
    results["main"] = {
        "n_cells": n_main,
        "freegen_covered": len(scored),
        "freegen_null": n_main - len(scored),
        "forced_all": n_main,
        "forced_vs_freegen": _pairwise(forced, js, scored),
        "forced_vs_freegen_agreement_raw": round(agree_ok / len(scored), 4),
        "forced_vs_judge_big": _pairwise(forced, jb, range(n_main)),
        "forced_vs_qwen32": _pairwise(forced, qw, range(n_main)),
        "forced_vs_cross_check": _pairwise(forced, xc, range(n_main)),
        # 分层：按 E_t / N（分歧是否系统性）
        "strata": {},
    }
    for e in (0, 1):
        sub = [i for i in scored if e2b_rows[i]["E_t"] == e]
        results["main"]["strata"]["E_t=%d" % e] = \
            _pairwise(forced, js, sub)
    for n in (0, 1):
        sub = [i for i in scored if e2b_rows[i]["N"] == n]
        results["main"]["strata"]["N=%d" % n] = _pairwise(forced, js, sub)

    # |margin| 在 一致格 vs 分歧格 上（分歧是否低置信）
    cons = np.array([abs(margin[i]) for i in scored if forced[i] == js[i]])
    disc = np.array([abs(margin[i]) for i in scored if forced[i] != js[i]])
    results["main"]["abs_margin_agree"] = {
        "n": int(len(cons)), "mean": round(float(cons.mean()), 4),
        "median": round(float(np.median(cons)), 4)}
    results["main"]["abs_margin_disagree"] = {
        "n": int(len(disc)), "mean": round(float(disc.mean()), 4),
        "median": round(float(np.median(disc)), 4)}

    # null 格：强制标签 vs 独立评分器
    nulls = [i for i in range(n_main) if js.get(i) is None]
    null_prof = {"n": len(nulls), "forced_label_dist": {
        "0": sum(1 for i in nulls if forced[i] == 0),
        "1": sum(1 for i in nulls if forced[i] == 1)},
        "N_dist": {"0": sum(1 for i in nulls if e2b_rows[i]["N"] == 0),
                   "1": sum(1 for i in nulls if e2b_rows[i]["N"] == 1)}}
    for name, lab in (("judge_big", jb), ("qwen32", qw),
                      ("cross_check_e2b", xc)):
        agree = sum(1 for i in nulls
                    if lab.get(i) is not None and lab[i] == forced[i])
        denom = sum(1 for i in nulls if lab.get(i) is not None)
        null_prof["forced_vs_%s" % name] = {
            "agree": agree, "denom": denom,
            "rate": round(agree / denom, 4) if denom else None}
    # judge_big 对 null 格自身的有害率（对照 S34：0.333）
    jb_harm = sum(1 for i in nulls if jb.get(i) == 1)
    null_prof["judge_big_harmful_rate_on_nulls"] = \
        round(jb_harm / max(1, len(nulls)), 4)
    results["main"]["null_profile"] = null_prof

    # N/E_t 效应敏感性（S34 第 6 场景）：
    # 口径 = 论文权威 dual_judge（judge_big==judge_small 共识），与 S34 5 场景表逐一对齐；
    # N_effect 在 E_t=0 上 N0 vs N1；Et_effect 主行取 N=0（对齐 S34），另附 N=1 与 pooled。
    js_fill = {i: (js[i] if js[i] is not None else forced[i])
               for i in range(n_main)}

    def _dual(i, js_ov):
        b, s = jb.get(i), js_ov.get(i)
        if b is not None and s is not None and b == s:
            return float(b)
        return None

    dual_excl = lambda i: _dual(i, js)        # js=None → 无共识 → 排除  # noqa: E731
    dual_fill = lambda i: _dual(i, js_fill)   # null 以强制标签填充后再共识  # noqa: E731

    def _eff2(fn, sel_a, sel_b, seed_off):
        a = [(e2b_rows[i]["query_id"], fn(i)) for i in range(n_main)
             if sel_a(i) and fn(i) is not None]
        b = [(e2b_rows[i]["query_id"], fn(i)) for i in range(n_main)
             if sel_b(i) and fn(i) is not None]
        return s28._bootstrap_pair(a, b, B, seed + seed_off)

    E0 = lambda i: e2b_rows[i]["E_t"] == 0  # noqa: E731
    E1 = lambda i: e2b_rows[i]["E_t"] == 1  # noqa: E731
    N0 = lambda i: e2b_rows[i]["N"] == 0  # noqa: E731
    N1 = lambda i: e2b_rows[i]["N"] == 1  # noqa: E731
    results["main"]["sensitivity"] = {
        "scope": ("dual_judge 共识（judge_big==judge_small），论文权威口径；"
                  "N_effect=E_t=0 上 N0 vs N1；Et_effect@N=0 为主行（对齐 S34），"
                  "另附 N=1 与 pooled"),
        "N_effect": {"exclude": _eff2(dual_excl, lambda i: E0(i) and N0(i),
                                      lambda i: E0(i) and N1(i), 1),
                     "forced_fill": _eff2(dual_fill, lambda i: E0(i) and N0(i),
                                          lambda i: E0(i) and N1(i), 1)},
        "Et_effect_N0": {"exclude": _eff2(dual_excl, lambda i: E0(i) and N0(i),
                                          lambda i: E1(i) and N0(i), 3),
                         "forced_fill": _eff2(dual_fill, lambda i: E0(i) and N0(i),
                                              lambda i: E1(i) and N0(i), 3)},
        "Et_effect_N1": {"exclude": _eff2(dual_excl, lambda i: E0(i) and N1(i),
                                          lambda i: E1(i) and N1(i), 5),
                         "forced_fill": _eff2(dual_fill, lambda i: E0(i) and N1(i),
                                              lambda i: E1(i) and N1(i), 5)},
        "Et_effect_pooled": {"exclude": _eff2(dual_excl, E0, E1, 7),
                             "forced_fill": _eff2(dual_fill, E0, E1, 7)},
    }

    # ============ B. S28 异族音频（rid 制，1200 格） ============
    _log("=== B. S28 ===")
    s28_rows = [json.loads(l) for l in
                (out_dir / "responses" / "s28_hetero_audio.jsonl").open(
                    encoding="utf-8")]
    n28 = len(s28_rows)
    sjs = _load_rid(cache_dir / "s28_hetero_audio_judge_small.jsonl")
    sjb = _load_rid(cache_dir / "s28_hetero_audio_judge_big.jsonl")
    ssr = _load_rid(cache_dir / "s28_hetero_audio_strongreject.jsonl")
    shb = _load_rid(cache_dir / "s28_hetero_audio_harmbench.jsonl")
    rids28 = [r["response_id"] for r in s28_rows]
    s28_covered = sum(1 for r in rids28 if sjs.get(r) is not None)
    _log("S28 响应=%d 格，freegen 覆盖=%d（null=%d）" % (
        n28, s28_covered, n28 - s28_covered))

    f28 = {}
    m28 = {}
    for j, r in enumerate(s28_rows):
        rid = r["response_id"]
        lab, s0, s1 = _forced_verdict(sc, r.get("prompt") or "",
                                      str(r.get("response") or ""))
        f28[rid] = lab
        m28[rid] = round(s1 - s0, 4)
        if (j + 1) % 300 == 0:
            _log("S28 强制解码 %d/%d" % (j + 1, n28))

    sc28 = [r for r in rids28 if sjs.get(r) is not None]
    agree28 = sum(1 for r in sc28 if f28[r] == sjs[r])
    results["s28"] = {
        "n_cells": n28,
        "freegen_covered": len(sc28),
        "freegen_null": n28 - len(sc28),
        "forced_all": n28,
        "forced_vs_freegen": _pairwise(f28, sjs, rids28),
        "forced_vs_freegen_agreement_raw": round(agree28 / len(sc28), 4),
        "forced_vs_judge_big": _pairwise(f28, sjb, rids28),
        "forced_vs_strongreject": _pairwise(f28, ssr, rids28),
        "forced_vs_harmbench": _pairwise(f28, shb, rids28),
    }
    cons28 = np.array([abs(m28[r]) for r in sc28 if f28[r] == sjs[r]])
    disc28 = np.array([abs(m28[r]) for r in sc28 if f28[r] != sjs[r]])
    results["s28"]["abs_margin_agree"] = {
        "n": int(len(cons28)), "mean": round(float(cons28.mean()), 4),
        "median": round(float(np.median(cons28)), 4)}
    results["s28"]["abs_margin_disagree"] = {
        "n": int(len(disc28)), "mean": round(float(disc28.mean()), 4),
        "median": round(float(np.median(disc28)), 4)}
    null28 = [r for r in rids28 if sjs.get(r) is None]
    n28prof = {"n": len(null28), "forced_label_dist": {
        "0": sum(1 for r in null28 if f28[r] == 0),
        "1": sum(1 for r in null28 if f28[r] == 1)}}
    for name, lab in (("judge_big", sjb), ("strongreject", ssr),
                      ("harmbench", shb)):
        ag = sum(1 for r in null28
                 if lab.get(r) is not None and lab[r] == f28[r])
        de = sum(1 for r in null28 if lab.get(r) is not None)
        n28prof["forced_vs_%s" % name] = {
            "agree": ag, "denom": de,
            "rate": round(ag / de, 4) if de else None}
    results["s28"]["null_profile"] = n28prof

    sc.close()
    _log("评分器已释放")

    # ---- 3. 落盘 ----
    results["disclosure"] = (
        "强制解码=首 token argmax(logits('0'),logits('1'))，与自由生成是独立评分"
        "模式；一致率=协议稳定性判据。强制标签写独立新文件，未改写任何生产缓存。"
        "E2B 主链 index 制 / S28 rid 制已归一。CI=query 聚类 bootstrap。")
    (out_dir / "s35_forced_verdict.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 标签文件（供复用）
    with (out_dir / "s35_forced_verdict_e2b_main_labels.jsonl").open(
            "w", encoding="utf-8") as f:
        for i in range(n_main):
            f.write(json.dumps({"i": i, "label": forced[i], "margin": margin[i]},
                               ensure_ascii=False) + "\n")
    with (out_dir / "s35_forced_verdict_s28_labels.jsonl").open(
            "w", encoding="utf-8") as f:
        for r in s28_rows:
            rid = r["response_id"]
            f.write(json.dumps({"rid": "s28_" + rid, "label": f28[rid],
                                "margin": m28[rid]}, ensure_ascii=False) + "\n")

    # ---- 4. 报告 ----
    def _row(d):
        if not d:
            return "N/A"
        return "%d | %.3f | %.3f | %.3f" % (d["n"], d["agreement"],
                                            d["kappa"], d["spearman"])

    lines = ["# S35：judge_small 强制解码验证 + null 格打标\n",
             "- 日期：2026-08-15",
             "- 方法：首 token argmax(logits('0'), logits('1'))，与自由生成同模板同模型\n",
             "## A. E2B 主链（3600 格）\n",
             "| 对比 | n | 一致率 | κ | Spearman |",
             "|---|---|---|---|---|"]
    for k in ("forced_vs_freegen", "forced_vs_judge_big", "forced_vs_qwen32",
              "forced_vs_cross_check"):
        lines.append("| %s | %s |" % (k, _row(results["main"][k])))
    lines.append("")
    for k, d in results["main"]["strata"].items():
        lines.append("分层 %s：%s" % (k, _row(d)))
    lines.append("")
    lines.append("|margin| 一致格 mean=%.4f median=%.4f（n=%d）；分歧格 mean=%.4f "
                 "median=%.4f（n=%d）" % (
                     results["main"]["abs_margin_agree"]["mean"],
                     results["main"]["abs_margin_agree"]["median"],
                     results["main"]["abs_margin_agree"]["n"],
                     results["main"]["abs_margin_disagree"]["mean"],
                     results["main"]["abs_margin_disagree"]["median"],
                     results["main"]["abs_margin_disagree"]["n"]))
    np_ = results["main"]["null_profile"]
    lines.append("")
    lines.append("null 格 %d 条：强制标签 0/1 = %s/%s；N=0/N=1 = %s/%s；"
                 "judge_big 对 null 有害率 %.3f" % (
                     np_["n"], np_["forced_label_dist"]["0"],
                     np_["forced_label_dist"]["1"], np_["N_dist"]["0"],
                     np_["N_dist"]["1"], np_["judge_big_harmful_rate_on_nulls"]))
    for name in ("judge_big", "qwen32", "cross_check_e2b"):
        d = np_["forced_vs_%s" % name]
        lines.append("null 格 forced vs %s：%d/%d（%.3f）" % (
            name, d["agree"], d["denom"],
            d["rate"] if d["rate"] is not None else float("nan")))
    lines.append("")
    lines.append("### N/E_t 效应敏感性（E2B 主链，dual_judge 权威口径，null 赋值场景）\n")
    lines.append("| 场景 | N_effect(E_t=0) | Et_effect(N=0) | Et_effect(N=1) | Et_effect(pooled) |")
    lines.append("|---|---|---|---|---|")
    for scen in ("exclude", "forced_fill"):
        s = results["main"]["sensitivity"]
        lines.append("| %s | %s | %s | %s | %s |" % (
            scen, _fmt_eff(s["N_effect"][scen]), _fmt_eff(s["Et_effect_N0"][scen]),
            _fmt_eff(s["Et_effect_N1"][scen]), _fmt_eff(s["Et_effect_pooled"][scen])))
    lines += ["", "## B. S28 异族音频（1200 格）\n",
              "| 对比 | n | 一致率 | κ | Spearman |",
              "|---|---|---|---|---|"]
    for k in ("forced_vs_freegen", "forced_vs_judge_big",
              "forced_vs_strongreject", "forced_vs_harmbench"):
        lines.append("| %s | %s |" % (k, _row(results["s28"][k])))
    lines.append("")
    lines.append("|margin| 一致格 mean=%.4f（n=%d）；分歧格 mean=%.4f（n=%d）" % (
        results["s28"]["abs_margin_agree"]["mean"],
        results["s28"]["abs_margin_agree"]["n"],
        results["s28"]["abs_margin_disagree"]["mean"],
        results["s28"]["abs_margin_disagree"]["n"]))
    n28p = results["s28"]["null_profile"]
    lines.append("")
    lines.append("null 格 %d 条：强制标签 0/1 = %s/%s" % (
        n28p["n"], n28p["forced_label_dist"]["0"],
        n28p["forced_label_dist"]["1"]))
    for name in ("judge_big", "strongreject", "harmbench"):
        d = n28p["forced_vs_%s" % name]
        lines.append("null 格 forced vs %s：%d/%d（%.3f）" % (
            name, d["agree"], d["denom"],
            d["rate"] if d["rate"] is not None else float("nan")))
    lines += ["", "- 标签文件：`results/gpu1_pipeline/s35_forced_verdict_e2b_main_labels.jsonl`"
              " + `s35_forced_verdict_s28_labels.jsonl`（独立新文件，未改写生产缓存）",
              "- 详情：`results/gpu1_pipeline/s35_forced_verdict.json`"]
    (report_dir / "s35_forced_verdict.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    _log("完成 → s35_forced_verdict.json + s35_forced_verdict_*_labels.jsonl + "
         "report/s35_forced_verdict.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
