#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S20b：E4B 文本 judge_big + judge_small 增量评分（GPU1，2026-08-14）。

动机：S17b 只给 E4B 文本上了 qwen32 强锚单腿。为把 S20 的跨生成器（E4B vs E2B
同格 3600）从 qwen32 单腿升级到 pre-registered R2 的 dual_judge 共识口径
（judge_big==judge_small），需补 E4B 文本两条 judge 腿。此标签也是 E4B 完成后
S17 Part B 跨族核验（E4B 全量 judge/qwen32 一致率 ≥0.80）文本侧的前置。

纪律：
  - 本阶段只评分 + 覆盖率报告，零分析（不做部分数据统计）。
  - 只读 E4B 文本 responses（append-only，读当前 EOF），只写
    results/gpu1_pipeline/scorers_cache/s17_e4b_text_<judge>.jsonl。
  - 幂等可续：按 response_id 键控，同一命令重跑自动跳过已评（catch-up 到全量）。
  - CUDA_VISIBLE_DEVICES=1 由调用方注入；不写账本 / .complete / done。

用法：CUDA_VISIBLE_DEVICES=1 python gpu1_s20b_e4b_text_judges.py \
  [--judge both|judge_big|judge_small] [--modality text|audio|both]
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s20b %s] %s" % (Path(__file__).stem, m), flush=True)


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


def _score_judge(scorer, todo, cache_path, tag, batch_size=4):
    """score_batch 批量 + 每批追加缓存（幂等可续）。"""
    done = _read_cache(cache_path)
    todo = [r for r in todo if r["response_id"] not in done]
    _log("[%s] 待评 %d（缓存 %d）" % (tag, len(todo), len(done)))
    total = len(todo)
    for start in range(0, total, batch_size):
        chunk = todo[start:start + batch_size]
        pairs = [(r["prompt"], r["response"] or "") for r in chunk]
        try:
            res = scorer.score_batch(pairs, batch_size=len(chunk))
        except Exception as e:  # noqa: BLE001
            _log("[%s] 批次失败: %s" % (tag, str(e)[:150]))
            res = [{"label": None} for _ in chunk]
        with cache_path.open("a", encoding="utf-8") as f:
            for r, x in zip(chunk, res):
                f.write(json.dumps({"rid": r["response_id"],
                                    "label": x.get("label")},
                                   ensure_ascii=False) + "\n")
        if (start // batch_size) % 25 == 0:
            _log("[%s] %d/%d" % (tag, min(start + batch_size, total), total))
    return _read_cache(cache_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--judge", choices=["both", "judge_big", "judge_small"],
                    default="both")
    ap.add_argument("--modality", choices=["text", "audio", "both"],
                    default="text",
                    help="text=E4B 文本 3600；audio=E4B 音频增量（随生成推进）")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    cache_dir = root / "results" / "gpu1_pipeline" / "scorers_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    resp_path = root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl"
    rows = [json.loads(l) for l in open(resp_path, encoding="utf-8")]
    mods = ["text", "audio"] if args.modality == "both" else [args.modality]
    for mod in mods:
        sel = [r for r in rows if r.get("modality") == mod]
        _log("E4B %s=%d" % (mod, len(sel)))

        import gpu1_s9_cross_family as s9
        s9.register_scorers(cfg)
        targets = (["judge_big", "judge_small"] if args.judge == "both"
                   else [args.judge])
        for sn in targets:
            cache_path = cache_dir / ("s17_e4b_%s_%s.jsonl" % (mod, sn))
            _log("评分 %s[%s] → %s" % (sn, mod, cache_path.name))
            sc = s9.SCORER_FACTORIES[sn]()
            try:
                cache = _score_judge(sc, sel, cache_path, "%s/%s" % (mod, sn),
                                     batch_size=args.batch_size)
            finally:
                sc.close()
            nonnull = sum(1 for v in cache.values() if v is not None)
            _log("[%s] 完成：缓存 %d，非空 %d/%d" % (
                "%s/%s" % (mod, sn), len(cache), nonnull, len(sel)))
    _log("S20b 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
