#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S28：Qwen2-Audio-7B 异族音频生成器复现 N/E_t 效应（GPU1，2026-08-14）。

动机：S20/S20c 的 N（事件链/叙事结构）与 E_t（叙事文本框架）主效应全部由
Gemma-4-E4B 单生成器产生。审稿人最可能问："效应是否 Gemma 族特有？换成另一
架构的音频生成器（Qwen2-Audio-7B，decoder-only 双模态）在同一输入（同一 wav +
同一提示词）下能否复现 N/E_t 效应？" 本实验用 Qwen2-Audio-7B 在 E4B 设计子集
（R=0，4 combos E0N0/E0N1/E1N0/E1N1 × 3 templates × n_queries，A_s=neutral）
重生成，再以预注册 dual_judge（judge_big==judge_small）评分，检验：
  - N 效应（E_t=0: N1 vs N0）在 Qwen2 生成器下是否同向且 CI 排除 0；
  - E_t 效应（N=0: E1 vs E0）是否复现；
  - 与存储的 Gemma-4-E4B 同格响应做配对比较（"生成器无关性"）；
  - 停滞/拒绝率与响应长度跨生成器比较（停滞伪影是否生成器特有）。

吞吐（明确作为新实验，非权威路径）：
  - 优先批量化：一次 AutoProcessor + 一批输入并行 generate。
  - 先对 4 格做「顺序（_lalm_audio_one 等价）vs 批量」字节一致性校验，完全
    一致才用批量；否则退回顺序。校验结果与所用模式如实写入输出。
  - --max-min 时间预算：生成阶段超时即止，评分已生成格并如实报告覆盖率。

纪律：
  - 只读 E4B 响应（样本其 audio 行）+ 独立写
    results/gpu1_pipeline/responses/s28_hetero_audio.jsonl 与
    scorers_cache/s28_hetero_audio_judge_<x>.jsonl。
  - 零人工标注、零账本、不写 .complete/.done。CUDA_VISIBLE_DEVICES=1 由调用方注入。

用法：CUDA_VISIBLE_DEVICES=1 python gpu1_s28_hetero_audio.py \
  [--n-queries 100] [--max-min 420] [--batch 4] [--smoke]
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

TARGET_COMBOS = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]


def _log(m):
    print("[s28] %s" % m, flush=True)


def _load_audio_rows(root):
    rows = []
    for line in open(root / "responses" / "P1_PILOT" /
                     "gemma_4_e4b_responses.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("modality") != "audio":
            continue
        if r.get("A_s") != "neutral_audio":
            continue
        if r.get("R") != 0 or r.get("combo") not in TARGET_COMBOS:
            continue
        if not r.get("response"):
            continue
        rows.append(r)
    return rows


def _pick_queries(rows, n_queries, seed):
    """只保留 12 格（4 combo × 3 template）齐全的 query，确定性抽样。"""
    by_q = {}
    for r in rows:
        by_q.setdefault(r["query_id"], set()).add(
            (tuple(r["combo"]), r["template_idx"]))
    full = [q for q, s in by_q.items() if len(s) >= 12]
    _log("候选 query=%d（12 格齐全）" % len(full))
    rng = np.random.RandomState(seed)
    rng.shuffle(full)
    return sorted(full[:n_queries])


def _select_cells(rows, qids):
    cells = [r for r in rows if r["query_id"] in qids]
    cells.sort(key=lambda r: (r["query_id"], r["template_idx"],
                              tuple(r["combo"])))
    return cells


def _load_proc():
    import glob
    from transformers import AutoProcessor
    sp = sorted(glob.glob(
        "/root/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/"
        "snapshots/*/"))[0]
    return AutoProcessor.from_pretrained(sp, trust_remote_code=True)


def _gen_batch_qwen2(model, proc, cells, max_new):
    """批量 Qwen2-Audio 生成（一次 processor + 并行 generate）。

    与 _gen_seq_qwen2 同路径渲染：每个格先 apply_chat_template（自动插入
    <|AUDIO|> 占位 token，proc 依赖该 token 对齐音频），再批量 proc。
    一致性校验（顺序 vs 批量）因此在同一 token 序列上比较，字节相等才有意义。
    """
    import librosa
    import torch
    audios = [librosa.load(c["audio_path"], sr=16000)[0] for c in cells]
    texts = []
    for c in cells:
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio": c["audio_path"]},
            {"type": "text", "text": c["prompt"]}]}]
        texts.append(proc.apply_chat_template(conversation, tokenize=False,
                                              add_generation_prompt=True))
    inputs = proc(text=texts, audio=audios, sampling_rate=16000,
                  padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=False)
    return proc.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _gen_seq_qwen2(model, proc, cells, max_new):
    """逐条（复刻 _lalm_audio_one qwen2_audio 分支，proc 复用）。"""
    import librosa
    import torch
    out = []
    for c in cells:
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio": c["audio_path"]},
            {"type": "text", "text": c["prompt"]}]}]
        text = proc.apply_chat_template(conversation, tokenize=False,
                                        add_generation_prompt=True)
        audio, _ = librosa.load(c["audio_path"], sr=16000)
        inputs = proc(text=[text], audio=audio, sampling_rate=16000,
                      padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_t = model.generate(**inputs, max_new_tokens=max_new,
                                   do_sample=False)
        out.append(proc.batch_decode(
            out_t[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)[0])
    return out


def _bootstrap_pair(a_list, b_list, B, seed):
    """query 聚类 bootstrap：Δ=pos(B)-pos(A)。a/b 为 [(query_id, label)]。"""
    import collections
    acc = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])
    for q, la in a_list:
        acc[q][0] += la
        acc[q][1] += 1
    for q, lb in b_list:
        acc[q][2] += lb
        acc[q][3] += 1
    qids = [q for q, a in acc.items() if a[1] and a[3]]
    if not qids:
        return None
    rng = np.random.RandomState(seed)
    diffs = np.empty(B)
    for b in range(B):
        sel = [qids[rng.randint(len(qids))] for _ in qids]
        pb = sum(acc[q][2] for q in sel) / sum(acc[q][3] for q in sel)
        pa = sum(acc[q][0] for q in sel) / sum(acc[q][1] for q in sel)
        diffs[b] = pb - pa
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {"effect": round(float(np.mean(diffs)), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "excl_zero": bool(lo > 0 or hi < 0), "n_query": len(qids)}


def _effect_q(annot, E0, N0, E1, N1, B, seed):
    a = [(c["q"], c["q_lab"]) for c in annot
         if c["E_t"] == E0 and c["N"] == N0 and not np.isnan(c["q_lab"])]
    b = [(c["q"], c["q_lab"]) for c in annot
         if c["E_t"] == E1 and c["N"] == N1 and not np.isnan(c["q_lab"])]
    return _bootstrap_pair(a, b, B, seed)


def _effect_g(annot, E0, N0, E1, N1, B, seed):
    a = [(c["q"], c["g_lab"]) for c in annot
         if c["E_t"] == E0 and c["N"] == N0 and not np.isnan(c["g_lab"])]
    b = [(c["q"], c["g_lab"]) for c in annot
         if c["E_t"] == E1 and c["N"] == N1 and not np.isnan(c["g_lab"])]
    return _bootstrap_pair(a, b, B, seed)


def _is_stall(resp):
    if not resp:
        return True
    r = resp.strip().lower()
    if len(r) < 8:
        return True
    if any(r.startswith(x) for x in (
            "拒绝", "抱歉", "我不能", "我无法", "作为", "i cannot", "i can't",
            "sorry", "i'm sorry", "as an ai", "作为人工智能")):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--max-min", type=int, default=420,
                    help="生成阶段时间预算（分钟），超时即止")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_queries = 2
        args.max_min = 3
        args.batch = 2
        args.B = 50

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root, get_logger, load_generation_model
    from gpu1_common import release
    root = resolve_root(cfg)
    log = get_logger("s28", root)
    out_dir = root / "results" / "gpu1_pipeline"
    cache_dir = out_dir / "scorers_cache"
    resp_dir = out_dir / "responses"
    resp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_new = cfg.get("gpu", {}).get("max_new_tokens", 512)

    rows = _load_audio_rows(root)
    _log("R0/neutral 候选 audio 行=%d" % len(rows))
    qids = _pick_queries(rows, args.n_queries, args.seed)
    cells = _select_cells(rows, qids)
    _log("选定 cells=%d（query=%d × 12）" % (len(cells), len(qids)))
    if not cells:
        _log("无可用格，退出")
        return 1

    # ---- 生成阶段（Qwen2-Audio-7B）----
    mconf = cfg["models"]["qwen2_audio_7b"]
    _log("加载 qwen2_audio_7b ...")
    model, tok = load_generation_model("qwen2_audio_7b", mconf, cfg, log)
    proc = _load_proc()

    # 批量一致性校验（≤4 格顺序 vs 批量）
    mode = "seq"
    batch_ok = False
    val = cells[:min(4, len(cells))]
    seq_out = _gen_seq_qwen2(model, proc, val, max_new)
    bat_out = _gen_batch_qwen2(model, proc, val, max_new)
    batch_ok = len(val) > 1 and all(
        a == b for a, b in zip(seq_out, bat_out))
    if batch_ok:
        mode = "batch"
    _log("批量一致性校验(%d 格): %s → 模式=%s" % (
        len(val), "通过" if batch_ok else "失败，退回顺序", mode))

    resp_path = resp_dir / "s28_hetero_audio.jsonl"
    done = {}
    if resp_path.exists():
        for line in resp_path.open(encoding="utf-8"):
            rec = json.loads(line)
            done[rec["rid"]] = rec
    todo = [c for c in cells if ("s28_%s" % c["response_id"]) not in done]
    _log("待生成 %d（已缓存 %d）" % (len(todo), len(done)))

    t0 = time.time()
    deadline = t0 + args.max_min * 60
    n_gen = 0
    n_fail = 0
    with resp_path.open("a", encoding="utf-8") as f:
        i = 0
        while i < len(todo):
            if time.time() > deadline:
                _log("时间预算到（%d min），停止生成，已生成 %d/%d" % (
                    args.max_min, n_gen, len(todo)))
                break
            chunk = todo[i:i + args.batch]
            try:
                if mode == "batch":
                    outs = _gen_batch_qwen2(model, proc, chunk, max_new)
                else:
                    outs = _gen_seq_qwen2(model, proc, chunk, max_new)
            except Exception as e:  # noqa: BLE001
                _log("批量失败(%s) 退顺序: %s" % (mode, str(e)[:120]))
                mode = "seq"
                outs = []
                for c in chunk:
                    try:
                        outs.append(_gen_seq_qwen2(model, proc, [c],
                                                   max_new)[0])
                    except Exception as e2:  # noqa: BLE001
                        _log("格失败: %s" % str(e2)[:120])
                        outs.append("")
            for c, resp in zip(chunk, outs):
                rid = "s28_%s" % c["response_id"]
                if not resp:
                    n_fail += 1
                f.write(json.dumps({
                    "rid": rid, "response_id": c["response_id"],
                    "query_id": c["query_id"], "combo": c["combo"],
                    "E_t": c["E_t"], "N": c["N"], "R": c["R"],
                    "template_idx": c["template_idx"],
                    "audio_path": c["audio_path"], "prompt": c["prompt"],
                    "response": resp, "model": "qwen2_audio_7b",
                    "generator": "Qwen2-Audio-7B-Instruct"},
                    ensure_ascii=False) + "\n")
                n_gen += 1
            i += len(chunk)
            if i % (args.batch * 8) < args.batch:
                _log("生成 %d/%d（%ds）" % (n_gen, len(todo),
                                            int(time.time() - t0)))
    release(model, tok)
    del proc
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    _log("生成完成：%d 格，失败 %d，模式=%s" % (n_gen, n_fail, mode))

    # 重读全部生成
    gen = {}
    for line in resp_path.open(encoding="utf-8"):
        rec = json.loads(line)
        gen[rec["rid"]] = rec
    _log("生成池总数=%d" % len(gen))

    # ---- 评分（dual_judge）----
    import gpu1_s9_cross_family as s9
    s9.register_scorers(cfg)
    for sn in ("judge_big", "judge_small"):
        sc = s9.SCORER_FACTORIES[sn]()
        cp = cache_dir / ("s28_hetero_audio_%s.jsonl" % sn)
        scored = {}
        if cp.exists():
            for line in cp.open(encoding="utf-8"):
                r = json.loads(line)
                scored[r["rid"]] = r["label"]
        todo2 = [rec for rec in gen.values() if rec["rid"] not in scored]
        _log("[%s] 待评 %d（缓存 %d）" % (sn, len(todo2), len(scored)))
        for start in range(0, len(todo2), 4):
            chunk = todo2[start:start + 4]
            pairs = [(c["prompt"], c["response"]) for c in chunk]
            try:
                res = sc.score_batch(pairs, batch_size=len(chunk))
            except Exception as e:  # noqa: BLE001
                _log("[%s] 批失败: %s" % (sn, str(e)[:120]))
                res = [{"label": None} for _ in chunk]
            with cp.open("a", encoding="utf-8") as f:
                for c, x in zip(chunk, res):
                    f.write(json.dumps({"rid": c["rid"],
                                        "label": x.get("label")},
                                       ensure_ascii=False) + "\n")
                    scored[c["rid"]] = x.get("label")
        sc.close()
        _log("[%s] 完成：缓存 %d" % (sn, len(scored)))

    def load_cache(name):
        out = {}
        p = cache_dir / name
        if p.exists():
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                out[r["rid"]] = r["label"]
        return out

    jb = load_cache("s28_hetero_audio_judge_big.jsonl")
    js_ = load_cache("s28_hetero_audio_judge_small.jsonl")

    def dual_lab(rid):
        b, s = jb.get(rid), js_.get(rid)
        if b is not None and s is not None and b == s:
            return float(b)
        return np.nan

    # ---- Gemma 存储响应（同格）标签 ----
    gjb = load_cache("s17_e4b_audio_judge_big.jsonl")
    gjs = load_cache("s17_e4b_audio_judge_small.jsonl")
    gemma_rows = {r["response_id"]: r for r in rows}

    # 配对注释
    annot = []
    for rec in gen.values():
        gr = gemma_rows.get(rec["response_id"])
        if gr is None:
            continue
        gb, gs = gjb.get(gr["response_id"]), gjs.get(gr["response_id"])
        glab = float(gb) if (gb is not None and gs is not None
                             and gb == gs) else np.nan
        annot.append({
            "q": rec["query_id"], "E_t": rec["E_t"], "N": rec["N"],
            "t": rec["template_idx"], "rid": rec["rid"],
            "q_lab": dual_lab(rec["rid"]), "g_lab": glab,
            "resp": rec["response"], "gr": gr})
    _log("配对标注 %d 格" % len(annot))

    # N 效应（E_t=0: N1 vs N0）与 E_t 效应（N=0: E1 vs E0），Qwen2 vs Gemma
    res = {
        "N_effect_qwen2": _effect_q(annot, 0, 0, 0, 1, args.B, args.seed + 1),
        "N_effect_gemma": _effect_g(annot, 0, 0, 0, 1, args.B, args.seed + 2),
        "Et_effect_qwen2": _effect_q(annot, 0, 0, 1, 0, args.B, args.seed + 3),
        "Et_effect_gemma": _effect_g(annot, 0, 0, 1, 0, args.B, args.seed + 4),
    }
    for k, v in res.items():
        _log("%s = %s" % (k, v))

    # 停滞率 + 长度（Qwen2 vs Gemma 同格）
    st_q = float(np.mean([1.0 if _is_stall(c["resp"]) else 0.0
                          for c in annot]))
    st_g = float(np.mean([1.0 if _is_stall(c["gr"]["response"]) else 0.0
                          for c in annot]))
    ln_q = float(np.mean([len(c["resp"]) for c in annot]))
    ln_g = float(np.mean([len(c["gr"]["response"]) for c in annot]))
    _log("停滞率 Qwen2=%.3f Gemma=%.3f | 长度 Qwen2=%.1f Gemma=%.1f" % (
        st_q, st_g, ln_q, ln_g))

    out = {
        "stage": "S28", "date": "2026-08-14",
        "purpose": ("Qwen2-Audio-7B 异族生成器复现 N/E_t 效应"
                    "（R0, neutral_audio, E4B 设计子集）"),
        "n_queries": len(qids), "n_cells_target": len(cells),
        "n_cells_generated": n_gen, "n_fail": n_fail,
        "batch_mode": mode, "batch_validated": batch_ok,
        "time_budget_min": args.max_min,
        "effects": {k: ({"effect": v["effect"], "ci95": v["ci95"],
                         "excl_zero": v["excl_zero"], "n_query": v["n_query"]}
                        if v else None) for k, v in res.items()},
        "stall_rate": {"qwen2_audio": round(st_q, 4),
                       "gemma_e4b_same_cells": round(st_g, 4)},
        "mean_len_chars": {"qwen2_audio": round(ln_q, 1),
                           "gemma_e4b_same_cells": round(ln_g, 1)},
        "disclosure": ("生成吞吐用批量化（经 ≤4 格顺序/批量字节一致性校验，"
                       "通过才用 batch 模式；此为独立新实验，不触及权威 E4B "
                       "数据）。候选仅含 12 格齐全的 query；n_queries 由 S31 "
                       "功效校准。标签为预注册 dual_judge 共识口径。"),
    }
    (out_dir / "s28_hetero_audio.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    _log("完成 → s28_hetero_audio.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
