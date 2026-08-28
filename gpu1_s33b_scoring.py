#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S33b：抢救评分——对 S33 已生成的采样格补跑 dual_judge 评分与分析。

背景：S33 生成阶段完成 344 格（responses/s33_hetero_audio.jsonl），但评分阶段
恰逢 E4B≥10700 触发 _e4b_done 首查即退出，n_cells_scored=0。本脚本只做评分+
分析（不生成），复用 S33 的评分/效应逻辑，产出完整 s33_hetero_audio.json。

时序依据：E4B 已 ≥10700，orchestrator 退出填充循环；stage_p1_pilot 仍跑至
10800 后做 effects 才退出，capstone 等其退出才占 GPU1。因此本窗口 GPU1 空闲，
评分 ~10-15min 可完成，不与任何腿冲突。不再以 _e4b_done 提前收尾（E4B 已达
阈值，评分是为抢救已生成数据），但仍保留 GPU1 占用检测（_gpu1_busy，含排除
gpu1_s33b 自身），绝不与 orchestrator/capstone 并发抢。

纪律：
  - 只写 scorers_cache/s33_hetero_audio_judge_<x>.jsonl + s33_hetero_audio.json；
  - 零人工标注、零账本、不写 .complete/.done；CUDA_VISIBLE_DEVICES=1 由调用方注入。

用法：CUDA_VISIBLE_DEVICES=1 python gpu1_s33b_scoring.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu1_s33_temperature as s33  # noqa: E402 复用 _gpu1_busy/_effect/_is_stall
import gpu1_s9_cross_family as s9  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root, get_logger
    root = resolve_root(cfg)
    log = get_logger("s33b", root)
    out_dir = root / "results" / "gpu1_pipeline"
    cache_dir = out_dir / "scorers_cache"
    resp_path = out_dir / "responses" / "s33_hetero_audio.jsonl"

    def _log(m):
        print("[s33b] %s" % m, flush=True)

    # ---- 1. 读已生成格 ----
    if not resp_path.exists():
        _log("无 s33_hetero_audio.jsonl，退出")
        return 1
    gen = {}
    for line in resp_path.open(encoding="utf-8"):
        rec = json.loads(line)
        gen[rec["rid"]] = rec
    _log("生成池=%d（目标 360，未生成的 16 格跳过）" % len(gen))
    if not gen:
        _log("生成池为空，退出")
        return 1

    # ---- 2. 等待 GPU1 空闲（绝不并发抢）----
    _log("等待 GPU1 空闲 ...")
    while s33._gpu1_busy():
        _log("GPU1 被占，等待 ...")
        time.sleep(30)
    _log("GPU1 空闲，开始评分")

    # ---- 3. dual_judge 评分 ----
    s9.register_scorers(cfg)
    for sn in ("judge_big", "judge_small"):
        _log("加载评分器 %s ..." % sn)
        sc = s9.SCORER_FACTORIES[sn]()
        cp = cache_dir / ("s33_hetero_audio_%s.jsonl" % sn)
        scored = {}
        if cp.exists():
            for line in cp.open(encoding="utf-8"):
                r = json.loads(line)
                scored[r["rid"]] = r["label"]
        todo2 = [rec for rec in gen.values() if rec["rid"] not in scored]
        _log("[%s] 待评 %d（缓存 %d）" % (sn, len(todo2), len(scored)))
        for start in range(0, len(todo2), 4):
            if s33._gpu1_busy():
                while s33._gpu1_busy():
                    time.sleep(20)
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
            if len(scored) % 32 == 0:
                _log("[%s] 进度 %d/%d" % (sn, len(scored), len(gen)))
        sc.close()
        _log("[%s] 完成：缓存 %d" % (sn, len(scored)))

    # ---- 4. 分析（复用 S33 逻辑）----
    def load_cache(name):
        out = {}
        p = cache_dir / name
        if p.exists():
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                out[r["rid"]] = r["label"]
        return out

    jb = load_cache("s33_hetero_audio_judge_big.jsonl")
    js_ = load_cache("s33_hetero_audio_judge_small.jsonl")
    s28_jb = load_cache("s28_hetero_audio_judge_big.jsonl")
    s28_js = load_cache("s28_hetero_audio_judge_small.jsonl")

    def dual_lab(b, s):
        if b is not None and s is not None and b == s:
            return float(b)
        return np.nan

    def s28_lab(rid):
        return dual_lab(s28_jb.get(rid), s28_js.get(rid))

    annot = []
    for rec in gen.values():
        b, s = jb.get(rec["rid"]), js_.get(rec["rid"])
        g28 = s28_lab("s28_%s" % rec["response_id"])
        annot.append({"q": rec["query_id"], "E_t": rec["E_t"], "N": rec["N"],
                      "t": rec["template_idx"], "rid": rec["rid"],
                      "lab": dual_lab(b, s), "g28": g28, "resp": rec["response"]})
    cov = len([c for c in annot if not np.isnan(c["lab"])])
    _log("配对标注 %d 格（已评 %d）" % (len(annot), cov))

    res = {
        "N_effect_sample": s33._effect(annot, 0, 0, 0, 1, args.B, args.seed + 1),
        "Et_effect_sample": s33._effect(annot, 0, 0, 1, 0, args.B, args.seed + 3),
    }
    sub = [c for c in annot if not np.isnan(c["g28"])]
    res["N_effect_greedy_samesub"] = s33._effect(
        [dict(c, lab=c["g28"]) for c in sub], 0, 0, 0, 1, args.B, args.seed + 5)
    res["Et_effect_greedy_samesub"] = s33._effect(
        [dict(c, lab=c["g28"]) for c in sub], 0, 0, 1, 0, args.B, args.seed + 7)
    for k, v in res.items():
        _log("%s = %s" % (k, v))

    agree = [c for c in annot
             if not np.isnan(c["lab"]) and not np.isnan(c["g28"])
             and c["lab"] == c["g28"]]
    denom = len([c for c in annot
                 if not np.isnan(c["lab"]) and not np.isnan(c["g28"])])
    agr = len(agree) / max(1, denom)
    st_q = float(np.mean([1.0 if s33._is_stall(c["resp"]) else 0.0
                          for c in annot]))
    ln_q = float(np.mean([len(c["resp"]) for c in annot]))
    _log("greedy vs sampling 标签一致率=%.3f（%d/%d）" % (agr, len(agree), denom))
    _log("停滞率=%.3f 长度=%.1f" % (st_q, ln_q))

    out = {
        "stage": "S33", "date": "2026-08-15",
        "note": ("S33b 抢救评分：S33 生成 344 格后评分阶段因 E4B≥10700 提前收尾"
                 "（n_cells_scored=0），本脚本补评已生成格并产出完整报告"),
        "purpose": ("解码温度鲁棒性：Qwen2-Audio-7B 以 temperature=1.0/top_p=0.9/"
                    "do_sample=True 重生成 S28 同款 cells，检验 N/E_t 效应是否 greedy"
                    "解码伪影"),
        "decode": {"temperature": 1.0, "top_p": 0.9, "top_k": 50,
                   "do_sample": True, "seed": args.seed},
        "n_queries": len(set(c["q"] for c in annot)),
        "n_cells_target": 360, "n_cells_generated": len(gen),
        "n_cells_scored": cov,
        "effects_sample": {k: ({"effect": v["effect"], "ci95": v["ci95"],
                                "excl_zero": v["excl_zero"],
                                "n_query": v["n_query"]}
                               if v else None)
                           for k, v in res.items() if "sample" in k},
        "effects_greedy_samesub": {k: ({"effect": v["effect"], "ci95": v["ci95"],
                                        "excl_zero": v["excl_zero"],
                                        "n_query": v["n_query"]}
                                       if v else None)
                                   for k, v in res.items() if "greedy" in k},
        "decode_agreement_greedy_vs_sample": round(agr, 4),
        "n_agreement_pairs": len(agree),
        "stall_rate_sample": round(st_q, 4),
        "mean_len_chars_sample": round(ln_q, 1),
        "coordination": ("S33b 抢救评分：E4B 已达 10700，orchestrator 退出填充循环；"
                         "stage_p1_pilot 仍跑至 10800，capstone 等其退出。GPU1 空闲"
                         "窗口内完成评分，保留 GPU1 占用检测绝不并发抢。"),
        "disclosure": ("覆盖率 n_cells_scored/n_cells_target=%d/%d；生成阶段因 E4B"
                       "完成停在 344 格（未生成 16 格），评分经 S33b 补全。解码为"
                       "采样（seed 固定可复现），与 S28 greedy 配对比较为'同格'设计。"
                       % (cov, 360)),
    }
    (out_dir / "s33_hetero_audio.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    _log("完成 → s33_hetero_audio.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
