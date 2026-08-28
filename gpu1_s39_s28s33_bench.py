#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S39：S33（Qwen2-Audio-7B 异族生成器，温度鲁棒性 scope）补评分腿。

动机：跨生成器图景（附录 B.4）已完整——ShieldGemma 的 N 翻转特定于 Gemma 族
生成器，在 Qwen2-Audio 上不翻转。覆盖盘点：S28（1200 格）已有
judge_big/judge_small/strongreject/harmbench（+S37 shieldgemma）共 5 家；
**S33（344 格）仅 judge_big/judge_small（+S38 shieldgemma）**，缺官方基准
（strongreject/harmbench）与 qwen32 强锚点。本实验补齐 S33，使温度鲁棒性
scope 在每个评分家族都有覆盖（镜像 S20d/S21 对称覆盖）。

产出（只写独立新文件，未改写生产缓存/账本/done）：
  results/gpu1_pipeline/scorers_cache/s33_hetero_audio_strongreject.jsonl
  results/gpu1_pipeline/scorers_cache/s33_hetero_audio_harmbench.jsonl
  results/gpu1_pipeline/scorers_cache/s33_hetero_audio_qwen32.jsonl
  results/gpu1_pipeline/s39_s33_bench.json（汇总）

纪律：CUDA_VISIBLE_DEVICES=1；GPU1 串行检测（_gpu1_busy）；只读响应；幂等可续
（rid 键控，加 s33_ 前缀与既有 judge 缓存一致）；harmbench 用左填充 1-token
Yes/No logit wrapper；strongreject/qwen32 用 score_batch。

用法：CUDA_VISIBLE_DEVICES=1 /root/.venv/bin/python gpu1_s39_s28s33_bench.py
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s39 %s] %s" % (Path(__file__).stem, m), flush=True)


def _gpu1_busy(min_free_mib=8192):
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


def _read_cache(p):
    """读取缓存；仅把 label 非 None 的条目视为已评（null 是失败/污染行，需重试）。"""
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                if rec.get("label") is not None:
                    out[rec["rid"]] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
    return out


def _harmbench_batch(sc, pairs, bs=8):
    """HarmBench-13b-cls 批处理：左填充 + 1 token generate + Yes/No logit 比较。"""
    import torch
    texts = [sc._wrap_llama2(b, g) for b, g in pairs]
    enc = sc.tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048)
    enc = {k: v.to(sc.model.device) if hasattr(v, "to") else v
           for k, v in enc.items()}
    with torch.no_grad():
        out = sc.model.generate(
            **enc, max_new_tokens=1, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
            pad_token_id=sc.tokenizer.pad_token_id)
    res = []
    first_scores = out.scores[0]  # [B, vocab]
    for i in range(len(pairs)):
        row = first_scores[i]
        ly = row[sc.yes_ids].max().item()
        ln = row[sc.no_ids].max().item()
        label = 1 if ly > ln else 0
        res.append({"label": label})
    return res


def _score_all(scorer, todo, cache_path, tag, batch=4, is_harmbench=False,
               is_qwen32=False, hb_batch=2):
    done = _read_cache(cache_path)
    todo = [r for r in todo if r["rid"] not in done]
    _log("[%s] 待评 %d（缓存 %d）" % (tag, len(todo), len(done)))
    total = len(todo)
    bs = hb_batch if is_harmbench else batch
    for start in range(0, total, bs):
        chunk = todo[start:start + bs]
        pairs = [(r["prompt"], r["response"] or "") for r in chunk]
        try:
            if is_harmbench:
                res = _harmbench_batch(scorer, pairs, bs=len(chunk))
            elif is_qwen32:
                # Qwen32Scorer.score_batch 签名无 batch_size（内部自定 batch）
                res = scorer.score_batch(pairs)
            else:
                res = scorer.score_batch(pairs, batch_size=len(chunk))
        except Exception as e:  # noqa: BLE001
            _log("[%s] 批次失败: %s" % (tag, str(e)[:150]))
            res = [{"label": None} for _ in chunk]
        with cache_path.open("a", encoding="utf-8") as f:
            for r, x in zip(chunk, res):
                f.write(json.dumps({"rid": r["rid"],
                                    "label": x.get("label")},
                                   ensure_ascii=False) + "\n")
        if (start // bs) % 25 == 0:
            _log("[%s] %d/%d" % (tag, min(start + bs, total), total))
    return _read_cache(cache_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--scorers", default="strongreject,harmbench,qwen32")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--hb-batch", type=int, default=2,
                    help="harmbench 批大小（GPU1 被主链占用 4.86GB 后显存受限，"
                         "默认 2；标签逐条判定不受 batch 影响）")
    args = ap.parse_args()

    if _gpu1_busy():
        _log("GPU1 剩余显存不足，退出（串行纪律）")
        return 1
    _log("GPU1 空闲，开始")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    cache_dir = out_dir / "scorers_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 响应加载 + rid 前缀（与既有 judge 缓存一致，仅 S33）
    scopes = []
    key, rid_prefix = "s33_hetero_audio", "s33_"
    rows = [json.loads(l) for l in
            (out_dir / "responses" / (key + ".jsonl")).open(encoding="utf-8")]
    for r in rows:
        r["rid"] = rid_prefix + r["response_id"]
    scopes.append((key, rows))
    _log("%s=%d（S28 已有 strongreject/harmbench/judge_big/judge_small 完整缓存，"
         "只补 S33）" % (key, len(rows)))

    import gpu1_s9_cross_family as s9
    from scorer_utils import HarmBenchScorer
    s9.register_scorers(cfg)

    summary = {"stage": "S39", "date": "2026-08-16",
               "purpose": ("S33（Qwen2-Audio 温度鲁棒性 scope）补官方基准腿 "
                           "strongreject+harmbench + qwen32 强锚点"),
               "method": ("复用 S21 协议：strongreject=Gemma4ScorerBase.score_batch；"
                          "harmbench=左填充 1-token Yes/No logit wrapper；"
                          "qwen32=score_batch；rid 键控幂等，s33_ 前缀")}
    for key, rows in scopes:
        summary[key] = {}
        for sn in [s.strip() for s in args.scorers.split(",") if s.strip()]:
            cache_path = cache_dir / ("%s_%s.jsonl" % (key, sn))
            _log("评分 %s %s → %s" % (key, sn, cache_path.name))
            if sn == "harmbench":
                sc = HarmBenchScorer(cfg["scorers"]["harmbench_model"])
                try:
                    cache = _score_all(sc, rows, cache_path, "%s_%s" % (key, sn),
                                       batch=args.batch, is_harmbench=True,
                                       hb_batch=args.hb_batch)
                finally:
                    sc.close()
            elif sn == "qwen32":
                # qwen32 用 S9 的独立 Qwen32Scorer（AWQ 原生加载，score_batch）
                sc = s9.Qwen32Scorer(s9._discover_awq())
                try:
                    cache = _score_all(sc, rows, cache_path, "%s_%s" % (key, sn),
                                       batch=args.batch, is_harmbench=False,
                                       is_qwen32=True)
                finally:
                    sc.close()
            else:
                sc = s9.SCORER_FACTORIES[sn]()
                try:
                    cache = _score_all(sc, rows, cache_path, "%s_%s" % (key, sn),
                                       batch=args.batch, is_harmbench=False)
                finally:
                    sc.close()
            nonnull = sum(1 for v in cache.values() if v is not None)
            summary[key][sn] = {"cached": len(cache), "nonnull": nonnull,
                                "harmful_rate": round(
                                    sum(1 for v in cache.values() if v == 1) /
                                    max(1, nonnull), 4)}
            _log("[%s_%s] 完成：缓存 %d，非空 %d/%d，有害率 %.4f" % (
                key, sn, len(cache), nonnull, len(rows),
                summary[key][sn]["harmful_rate"]))
        _log("%s 评分完成" % key)

    (out_dir / "s39_s33_bench.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log("完成 → s39_s33_bench.json（独立新文件，未改写生产缓存）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
