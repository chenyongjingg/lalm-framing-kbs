#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S21：E2B 文本 strongreject + harmbench 官方基准评分腿（GPU1，2026-08-14）。

动机：S20d 在 E4B 文本侧补了 strongreject + harmbench 两条官方基准评分器腿。
E2B 侧此前只有 judge_big/judge_small/qwen32/cross_check 四口径（S11/S19），
官方基准未覆盖。本实验镜像 S20d 到 E2B 侧 3600 单元，使官方基准评分器在
双生成器（E2B/E4B 同格 3600）对称覆盖，论文"评分稳健性"章节闭环。

纪律：
  - 只评分 + 覆盖率，零分析。只读 E2B 文本 responses，只写
    results/gpu1_pipeline/scorers_cache/s17_e2b_text_<scorer>.jsonl。
  - 幂等可续（response_id 键控，E2B 响应含 response_id）；CUDA_VISIBLE_DEVICES=1
    由调用方注入；不写账本/done。
  - strongreject 用 Gemma4ScorerBase.score_batch(batch=4)；
    harmbench 原生仅 score_one（1 token generate），写批处理 wrapper（左填充 +
    Yes/No 首 token logit 比较，复用 HarmBenchScorer 的 __init__/_wrap_llama2）。

用法：CUDA_VISIBLE_DEVICES=1 python gpu1_s21_e2b_text_bench.py [--scorers strongreject,harmbench] [--batch 4]
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s21 %s] %s" % (Path(__file__).stem, m), flush=True)


def _read_cache(p):
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
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


def _score_all(scorer, todo, cache_path, tag, batch=4, is_harmbench=False):
    done = _read_cache(cache_path)
    todo = [r for r in todo if r["response_id"] not in done]
    _log("[%s] 待评 %d（缓存 %d）" % (tag, len(todo), len(done)))
    total = len(todo)
    bs = 8 if is_harmbench else batch
    for start in range(0, total, bs):
        chunk = todo[start:start + bs]
        pairs = [(r["prompt"], r["response"] or "") for r in chunk]
        try:
            if is_harmbench:
                res = _harmbench_batch(scorer, pairs, bs=len(chunk))
            else:
                res = scorer.score_batch(pairs, batch_size=len(chunk))
        except Exception as e:  # noqa: BLE001
            _log("[%s] 批次失败: %s" % (tag, str(e)[:150]))
            res = [{"label": None} for _ in chunk]
        with cache_path.open("a", encoding="utf-8") as f:
            for r, x in zip(chunk, res):
                f.write(json.dumps({"rid": r["response_id"],
                                    "label": x.get("label")},
                                   ensure_ascii=False) + "\n")
        if (start // bs) % 25 == 0:
            _log("[%s] %d/%d" % (tag, min(start + bs, total), total))
    return _read_cache(cache_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--scorers", default="strongreject,harmbench")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    cache_dir = root / "results" / "gpu1_pipeline" / "scorers_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e2b_responses.jsonl",
        encoding="utf-8")]
    sel = [r for r in rows if r.get("modality") == "text"]
    _log("E2B 文本=%d" % len(sel))

    import gpu1_s9_cross_family as s9
    from scorer_utils import HarmBenchScorer
    s9.register_scorers(cfg)
    for sn in [s.strip() for s in args.scorers.split(",") if s.strip()]:
        cache_path = cache_dir / ("s17_e2b_text_%s.jsonl" % sn)
        _log("评分 %s → %s" % (sn, cache_path.name))
        if sn == "harmbench":
            sc = HarmBenchScorer(cfg["scorers"]["harmbench_model"])
            try:
                cache = _score_all(sc, sel, cache_path, sn,
                                   batch=args.batch, is_harmbench=True)
            finally:
                sc.close()
        else:
            sc = s9.SCORER_FACTORIES[sn]()
            try:
                cache = _score_all(sc, sel, cache_path, sn,
                                   batch=args.batch, is_harmbench=False)
            finally:
                sc.close()
        nonnull = sum(1 for v in cache.values() if v is not None)
        _log("[%s] 完成：缓存 %d，非空 %d/%d" % (sn, len(cache), nonnull,
                                             len(sel)))
    _log("S21 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
