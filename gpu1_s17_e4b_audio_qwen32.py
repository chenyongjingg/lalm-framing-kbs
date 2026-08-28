#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S17-PartA：E4B 全量 qwen32 强锚增量预评分（文本/音频，2026-08-14）。

动机：S17（E4B 跨族核验，L3 权威主模态收官）的分析需 E4B 完成后执行；
但其 GPU 最贵的 qwen32 强锚评分腿可提前增量做掉——评分逐条独立、顺序无关，
cache 按 response_id 键控，最终分析只在全量数据上跑。

科学纪律：
  - 本阶段只评分 + 覆盖报告，零分析（不做部分数据统计，避免非科学口径）。
  - 只读 E4B responses（append-only，读当前 EOF），只写
    results/gpu1_pipeline/scorers_cache/s17_e4b_<modality>_qwen32.jsonl。
  - 不写任何账本 / .complete / done；CUDA_VISIBLE_DEVICES=1 由调用方注入。
  - 幂等可续：同一命令随时重跑，自动跳过已评 response_id（catch-up 到全量）。

用法：CUDA_VISIBLE_DEVICES=1 python gpu1_s17_e4b_audio_qwen32.py [--modality audio|text]
  （默认 audio，兼容既有 S17a 音频流）
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s17a %s] %s" % (Path(__file__).stem, m), flush=True)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["audio", "text"], default="audio")
    args = ap.parse_args()
    mod = args.modality

    cfg = yaml.safe_load(open("pipeline_config.yaml", encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    cache_dir = root / "results" / "gpu1_pipeline" / "scorers_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / ("s17_e4b_%s_qwen32.jsonl" % mod)

    resp_path = root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl"
    rows = [json.loads(l) for l in open(resp_path, encoding="utf-8")]
    sel = [r for r in rows if r.get("modality") == mod]
    done = _read_cache(cache_path)
    todo = [r for r in sel if r["response_id"] not in done]
    _log("E4B %s 总=%d 已评分=%d 待评=%d" % (mod, len(sel), len(done),
                                          len(todo)))
    if not todo:
        _log("无待评（已全量覆盖 %s），退出" % mod)
        return 0

    import gpu1_s9_cross_family as s9
    qw = s9.Qwen32Scorer(s9._discover_awq(), batch_size=8)
    try:
        pairs = [(r["prompt"], r["response"] or "") for r in todo]
        res = qw.score_batch(pairs)
        with cache_path.open("a", encoding="utf-8") as f:
            for r, x in zip(todo, res):
                f.write(json.dumps({"rid": r["response_id"],
                                    "label": x.get("label")},
                                   ensure_ascii=False) + "\n")
        nonnull = sum(1 for x in res if x.get("label") is not None)
        _log("本次评分 %d 条，非空 %d（总覆盖 %s %d/%d）" % (
            len(todo), nonnull, mod, len(done) + len(todo), len(sel)))
    finally:
        qw.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
