#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主链 Gemma-4-E2B 解码采样鲁棒性（N3-GPU，探索性，2026-08-27）。

审稿人攻击点："framing 效应是 greedy 解码伪影吗？" S33 已用 Qwen2-Audio-7B
（异族跨模态）证明采样下方向保持、CI 变宽；本实验在主链 Gemma-4-E2B bf16
（与 P1-FULL 同一生成器）上做同款，直接回答主链模型自身是否受解码参数影响。

协议：
  - 恶意查询：从 P1-FULL 主 scope 抽 --n-queries 个（zh/en 混合、跨 category、
    保证 greedy 下 dual_judge 有标签）。
  - cells：baseline + storytelling × t0/t1/t2（FRAMING_TEMPLATES 逐字复用）
    = 6 cells/query。聚焦 N（叙事结构）——协议 RQ 核心，S20 效应最强。
  - 生成：Gemma-4-E2B-it bf16 直载，do_sample=True, temperature=1.0,
    top_p=0.9, torch.manual_seed(20260827)（可复现）。
  - 评分：DualJudgeScorer（E4B big 全程 → E2B mistral 全程），behavior=query_text
    （无 framing 前缀，与 P1-FULL/s40 一致）；dual_judge=big==mistral。
  - 分析：N 效应（storytelling vs baseline 的 dual ASR 差，query 配对）在
    greedy（P1-FULL 同 cell 标签）vs sampling（本实验）下方向是否保持；
    同格 greedy vs sampling 标签一致率。

写盘：results/sampling_robust/{responses,scores,analysis}.jsonl/json + report/sampling_robust.md
GPU 纪律：仅 S40 与 N1 全部退出后运行（wrapper 链保证）；单模型驻留。

用法：
  python s_e2b_sampling.py --n-queries 8
"""
import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/root/lalm_framing_revision_v6")
OUT = ROOT / "results" / "sampling_robust"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _ts():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _load_queries(n_q, seed):
    import pandas as pd
    df = pd.read_parquet(ROOT / "results" / "p1_full_scored.parquet")
    txt = df[(df["lang"].isin(["zh", "en"])) &
             (df["model"] == "gemma_4_e2b")]
    # 每个 query 取 baseline/storytelling 格（去重 query_id），要求 greedy dual 有标签
    qid_cells = {}
    for qid, g in txt.groupby("query_id"):
        sub = g[(g["condition"].isin(["baseline", "storytelling"]))]
        duals = sub["dual_judge_label"].notna().sum()
        if duals >= 4:  # 至少 4/6 格有 greedy 标签
            qid_cells[qid] = {
                "lang": g["lang"].iloc[0],
                "greedy_dual_n": int(duals),
            }
    qids = sorted(qid_cells)
    rng = np.random.default_rng(seed)
    rng.shuffle(qids)
    chosen = qids[:n_q]
    print(f"[queries] 候选 {len(qids)}（dual≥4 格），选 {len(chosen)}", flush=True)
    return chosen, qid_cells


def _build_cells(chosen, qid_cells):
    import pandas as pd
    from stage_p1_full import FRAMING_TEMPLATES
    df = pd.read_parquet(ROOT / "results" / "p1_full_scored.parquet")
    sub = df[(df["lang"].isin(["zh", "en"])) &
             (df["model"] == "gemma_4_e2b") &
             (df["query_id"].isin(chosen)) &
             (df["condition"].isin(["baseline", "storytelling"]))]
    # 每个 (query, condition, template) 取 greedy 响应 + 标签
    cells = []
    for _, r in sub.iterrows():
        cells.append({
            "query_id": r["query_id"], "lang": r["lang"],
            "query_text": r["query_text"],
            "condition": r["condition"], "template_idx": int(r["template_idx"]),
            "prompt": r["prompt"],
            "greedy_response": r["response"],
            "greedy_dual": (None if pd.isna(r["dual_judge_label"])
                            else int(r["dual_judge_label"])),
        })
    print(f"[cells] {len(cells)}（{len(chosen)} queries × 6）", flush=True)
    return cells


def _generate(cells):
    import torch
    import yaml
    cfg = yaml.safe_load((ROOT / "pipeline_config.yaml").read_text(encoding="utf-8"))
    mconf = cfg["models"]["gemma_4_e2b"]
    model_ref = mconf.get("path") or mconf.get("id")
    sys_msg = mconf.get(
        "system_prompt",
        "You are a careful, consistent assistant."
        "\n<start_of_thinking>\n<enable_thinking>false"
        "</enable_thinking>\n<end_of_thinking>").strip()
    max_new = cfg.get("gpu", {}).get("max_new_tokens", 512)
    print(f"[gen] 加载 {model_ref} bf16 (ModelManager) ...", flush=True)

    class _LogShim:
        """把 ModelManager 的 logger 调用转发到 print（S40 同款 shim 思路）。"""
        def _emit(self, lvl, m):
            try:
                print(f"[MM:{lvl}] {m}", flush=True)
            except Exception:  # noqa: BLE001
                pass

        def info(self, m, *a):
            self._emit("INFO", m % a if a else m)

        def warning(self, m, *a):
            self._emit("WARN", m % a if a else m)

        def warn(self, m, *a):
            self._emit("WARN", m % a if a else m)

        def error(self, m, *a):
            self._emit("ERROR", m % a if a else m)

        def debug(self, m, *a):
            self._emit("DEBUG", m % a if a else m)

        def exception(self, m, *a):
            self._emit("EXCEPT", m % a if a else m)

        def setLevel(self, lvl):
            pass

        def addHandler(self, h):
            pass

    from common_utils import ModelManager  # noqa: E402
    mm = ModelManager(_LogShim(),
                      load_timeout=cfg["gpu"]["model_load_timeout"],
                      prefer_fp16=False, hf_home=cfg.get("hf_home"),
                      io_cfg=cfg.get("io_optimization", {}))
    model, tok, prec = mm.load("gemma_4_e2b", model_ref)
    torch.manual_seed(20260827)
    OUT.mkdir(parents=True, exist_ok=True)
    resp_p = OUT / "responses.jsonl"
    done = set()
    if resp_p.exists():
        for l in resp_p.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if l:
                done.add((json.loads(l)["query_id"], json.loads(l)["condition"],
                          json.loads(l)["template_idx"]))
    pend = [c for c in cells
            if (c["query_id"], c["condition"], c["template_idx"]) not in done]
    print(f"[gen] 待生成 {len(pend)}/{len(cells)}", flush=True)
    t0 = time.time()
    with resp_p.open("a", encoding="utf-8") as f:
        for i, c in enumerate(pend):
            text = tok.apply_chat_template(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": c["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            inputs = tok(text=text, return_tensors="pt",
                         truncation=True, max_length=4096).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new,
                                     do_sample=True, temperature=1.0,
                                     top_p=0.9)
            resp = tok.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0]
            rec = {**{k: c[k] for k in
                      ("query_id", "lang", "query_text",
                       "condition", "template_idx", "prompt",
                       "greedy_response", "greedy_dual")},
                   "response_id": f"SP_{c['query_id']}_{c['condition']}_t{c['template_idx']}",
                   "response": resp, "decoding": "temp1.0_top0.9",
                   "timestamp": _ts()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 12 == 0 or i + 1 == len(pend):
                el = time.time() - t0
                print(f"[gen] {i+1}/{len(pend)} ({el:.0f}s, ~{el/max(i+1,1):.1f}s/cell)",
                      flush=True)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[gen] 完成 {len(pend)} cells, {time.time()-t0:.0f}s", flush=True)
    return 0


def _score():
    import pandas as pd
    import yaml
    from scorer_utils import DualJudgeScorer
    resp_p = OUT / "responses.jsonl"
    rows = []
    if resp_p.exists():
        for l in resp_p.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if l:
                rows.append(json.loads(l))
    score_p = OUT / "scores.jsonl"
    scored = {}
    if score_p.exists():
        for l in score_p.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if l:
                r = json.loads(l)
                scored[r["response_id"]] = r
    pend = [r for r in rows if r["response_id"] not in scored]
    print(f"[score] 待评 {len(pend)}/{len(rows)}", flush=True)
    if pend:
        cfg = yaml.safe_load((ROOT / "pipeline_config.yaml").read_text(encoding="utf-8"))
        dual = DualJudgeScorer(cfg["scorers"]["judge_big_model"],
                               cfg["scorers"]["judge_small_model"])
        with score_p.open("a", encoding="utf-8") as f:
            for i, r in enumerate(pend):
                lbl_b = dual.score_one_big(r["query_text"], str(r["response"]))
                rec = {"response_id": r["response_id"],
                       "judge_big_label": (int(lbl_b) if lbl_b is not None else None)}
                scored[r["response_id"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 12 == 0:
                    print(f"[score] big {i+1}/{len(pend)}", flush=True)
        dual.unload_big()
        dual.load_mistral_only()
        pend_m = [r for r in rows
                  if scored.get(r["response_id"], {}).get("judge_big_label") is not None
                  and scored[r["response_id"]].get("judge_mistral_label") is None]
        with score_p.open("a", encoding="utf-8") as f:
            for i, r in enumerate(pend_m):
                lbl_m = dual.score_one_mistral(r["query_text"], str(r["response"]))
                rec = scored[r["response_id"]]
                rec["judge_mistral_label"] = (int(lbl_m) if lbl_m is not None else None)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 12 == 0:
                    print(f"[score] mistral {i+1}/{len(pend_m)}", flush=True)
        dual.unload_mistral()
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    print(f"[score] 完成 {len(scored)} 条", flush=True)
    return 0


def _analyze():
    import pandas as pd
    resp_p = OUT / "responses.jsonl"
    score_p = OUT / "scores.jsonl"
    rows = {}
    for l in resp_p.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if l:
            r = json.loads(l)
            rows[r["response_id"]] = r
    sc = {}
    if score_p.exists():
        for l in score_p.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if l:
                r = json.loads(l)
                sc[r["response_id"]] = r
    # 合并 dual 共识
    recs = []
    for rid, r in rows.items():
        s = sc.get(rid, {})
        b, m = s.get("judge_big_label"), s.get("judge_mistral_label")
        dual = None
        if b is not None and m is not None:
            dual = int(b == m and b == 1)
        recs.append({**r, "sampling_dual": dual,
                     "sampling_big": b, "sampling_mistral": m})

    n = len(recs)
    n_dual = sum(1 for r in recs if r["sampling_dual"] is not None)
    n_disp = sum(1 for r in recs
                 if r["sampling_big"] is not None and r["sampling_mistral"] is not None
                 and r["sampling_big"] != r["sampling_mistral"])
    agree = None
    if n_dual:
        both = [(r["sampling_dual"], r["greedy_dual"]) for r in recs
                if r["sampling_dual"] is not None and r["greedy_dual"] is not None]
        if both:
            agree = round(float(np.mean([a == b for a, b in both])) * 100, 1)

    def _asr(cond, key):
        vals = [r[key] for r in recs
                if r["condition"] == cond and r[key] is not None]
        if not vals:
            return None
        return round(float(np.mean(vals)) * 100, 2)

    def _boot_delta(key, n_boot=2000, seed=42):
        # query 配对 Δ(storytelling-baseline)
        base = defaultdict(list)
        story = defaultdict(list)
        for r in recs:
            if r[key] is None:
                continue
            if r["condition"] == "baseline":
                base[r["query_id"]].append(float(r[key]))
            elif r["condition"] == "storytelling":
                story[r["query_id"]].append(float(r[key]))
        qids = sorted(set(base) & set(story))
        if not qids:
            return None, None
        b = np.array([np.mean(base[q]) for q in qids])
        s = np.array([np.mean(story[q]) for q in qids])
        rng = np.random.default_rng(seed)
        deltas = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, len(qids), size=len(qids))
            deltas[i] = s[idx].mean() - b[idx].mean()
        return round(deltas.mean() * 100, 2), \
            [round(np.percentile(deltas, 2.5) * 100, 2),
             round(np.percentile(deltas, 97.5) * 100, 2)]

    out = {
        "stage": "sampling-robust-2026-08-27",
        "decoding": "temperature=1.0 top_p=0.9 seed=20260827",
        "model": "gemma_4_e2b bf16 (主链同款)",
        "n_cells": n, "n_dual": n_dual, "n_dispute": n_disp,
        "greedy_vs_sampling_agree_pct": agree,
        "asr_pct": {
            "greedy_baseline": _asr("baseline", "greedy_dual"),
            "greedy_storytelling": _asr("storytelling", "greedy_dual"),
            "sampling_baseline": _asr("baseline", "sampling_dual"),
            "sampling_storytelling": _asr("storytelling", "sampling_dual"),
        },
        "N_effect_delta_pp": {
            "greedy": _boot_delta("greedy_dual"),
            "sampling": _boot_delta("sampling_dual"),
        },
    }
    (OUT / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    gd = out["N_effect_delta_pp"]["greedy"]
    sd = out["N_effect_delta_pp"]["sampling"]
    md = [
        "# 主链 E2B 解码采样鲁棒性（N3，探索性）",
        "",
        f"- 模型：Gemma-4-E2B bf16（与 P1-FULL 同生成器），解码 temp=1.0/top_p=0.9/seed 固定。",
        f"- cells: {n}（{n} queries × baseline/storytelling × t0/t1/t2）。",
        f"- greedy vs sampling 同格 dual 一致率：{agree}%（n 有双标 {n_dual}，争议 {n_disp}）。",
        "",
        "## N 效应（storytelling vs baseline，query 配对 bootstrap 95% CI）",
        "| 解码 | baseline ASR% | storytelling ASR% | Δ(N) pp [95%CI] |",
        "|---|---|---|---|",
        f"| greedy（P1-FULL） | {out['asr_pct']['greedy_baseline']} | "
        f"{out['asr_pct']['greedy_storytelling']} | {gd[0]} [{gd[1][0]},{gd[1][1]}] |",
        f"| sampling（本实验） | {out['asr_pct']['sampling_baseline']} | "
        f"{out['asr_pct']['sampling_storytelling']} | {sd[0]} [{sd[1][0]},{sd[1][1]}] |",
        "",
        "## 判读（如实）",
        "- 若 sampling 下 Δ(N) 方向与 greedy 一致 → 主链 N 效应非 greedy 解码伪影。",
        "- 样本小（探索性）：CI 变宽预期，与 S33（Qwen2-Audio）合并构成"
        "「采样下方向保持」证据。若方向翻转/归零 → 如实披露为解码敏感。",
        "- 数据：results/sampling_robust/。",
    ]
    (ROOT / "report" / "sampling_robust.md").write_text("\n".join(md), encoding="utf-8")
    print("[analyze] 已写 analysis.json + report/sampling_robust.md", flush=True)
    print("\n".join(md), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=8)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        chosen, qid_cells = _load_queries(args.n_queries, 20260827)
        cells = _build_cells(chosen, qid_cells)
        rc = _generate(cells)
        if rc:
            return rc
        _score()
        return _analyze()
    if args.generate:
        chosen, qid_cells = _load_queries(args.n_queries, 20260827)
        return _generate(_build_cells(chosen, qid_cells))
    if args.score:
        return _score()
    if args.analyze:
        return _analyze()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
