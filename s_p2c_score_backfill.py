#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2-C 攻击评估缺环回填（v6.8-backfill-fix，2026-08-27）。

根因：stage_p2c_adaptive.evaluate_attacks 在同一进程同时驻留
HarmBenchScorer(8bit ~13GB) + StrongRejectScorer(BF16 ~16GB)，合计 > 24GB 单卡
→ torch.OutOfMemoryError（caching_allocator_warmup 14GiB 探测缓冲分配失败）。
攻击 2864 条已推理但未评分 → 评估缺环。

本脚本把两个重评分器拆到独立 OS 子进程，每进程只驻留一个模型：
  --score-hb  仅加载 HarmBenchScorer（8bit ~13GB），batch_size=1 控制激活峰值，
              逐条评分并落盘 results/p2c_hb_scores.jsonl（可断点续评）
  --score-sr  仅加载 StrongRejectScorer（BF16 ~16GB），逐条评分并落盘
              results/p2c_sr_scores.jsonl（可断点续评）
  --merge     纯 CPU：读取两份预评分 → monkey-patch scorer_utils 工厂 → 直接调用
              真实的 evaluate_attacks（下游 TPR/报告/md/CSV 零重写零偏差），
              产出 report/adaptive_attack_results.csv + adaptive_attack_evaluation.md

内存安全：每评分进程 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True；
  batch_size=1；加载前 gc.collect + torch.cuda.empty_cache。
不修改任何共享代码（stage_p2c_adaptive.py / scorer_utils.py 由他人维护）。

用法：
  python s_p2c_score_backfill.py --config pipeline_config.yaml --score-hb
  python s_p2c_score_backfill.py --config pipeline_config.yaml --score-sr
  python s_p2c_score_backfill.py --config pipeline_config.yaml --merge
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def load_config(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_attack_rows(out_dir):
    """与 stage_p2c_adaptive.evaluate_attacks 相同的读取顺序：
    sorted(out_dir.glob('attacks_*.jsonl'))，逐行解析。"""
    for f in sorted(Path(out_dir).glob("attacks_*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if not l:
                continue
            yield json.loads(l)


def _load_all_attacks(root):
    out_dir = Path(root) / "responses" / "P2C"
    rows = list(_iter_attack_rows(out_dir))
    return rows


# ---------------------------------------------------------------------------
# 评分子进程：HarmBench（只驻留 hb）
# ---------------------------------------------------------------------------
def score_hb(root, cfg):
    from scorer_utils import HarmBenchScorer
    import gc
    import torch

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    rows = _load_all_attacks(root)
    print(f"[hb] 攻击行数: {len(rows)}", flush=True)
    out = Path(root) / "results" / "p2c_hb_scores.jsonl"
    done = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(l)["idx"])
            except Exception:
                pass
    print(f"[hb] 已有 {len(done)} 条，续评", flush=True)

    hb = HarmBenchScorer(cfg["scorers"]["harmbench_model"],
                         load_in_4bit=cfg["scorers"].get("scorer_load_4bit", True))
    pairs = [(r["prompt"], r["response"]) for r in rows]
    try:
        with out.open("a", encoding="utf-8") as w:
            for i, (b, g) in enumerate(pairs):
                if i in done:
                    continue
                try:
                    s = hb.score_batch([(b, g)], batch_size=1)[0]
                    rec = {"idx": i, "label": s.get("label"),
                           "prob_yes": s.get("prob_yes", 0.5),
                           "error": s.get("error")}
                except Exception as e:  # noqa: BLE001
                    rec = {"idx": i, "label": None, "prob_yes": 0.5,
                           "error": str(e)[:200]}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                w.flush()
                if (i + 1) % 100 == 0:
                    print(f"[hb] {i+1}/{len(pairs)}", flush=True)
    finally:
        try:
            hb.close()
        except Exception:
            pass
    print(f"[hb] 完成，共 {len(rows)} 条", flush=True)
    return 0


# ---------------------------------------------------------------------------
# 评分子进程：StrongReject（只驻留 sr）
# ---------------------------------------------------------------------------
def score_sr(root, cfg):
    from scorer_utils import StrongRejectScorer
    import gc
    import torch

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    rows = _load_all_attacks(root)
    print(f"[sr] 攻击行数: {len(rows)}", flush=True)
    out = Path(root) / "results" / "p2c_sr_scores.jsonl"
    done = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(l)["idx"])
            except Exception:
                pass
    print(f"[sr] 已有 {len(done)} 条，续评", flush=True)

    sr = StrongRejectScorer(cfg["scorers"]["strongreject_model"],
                            load_in_4bit=False)
    pairs = [(r["prompt"], r["response"]) for r in rows]
    try:
        with out.open("a", encoding="utf-8") as w:
            for i, (b, g) in enumerate(pairs):
                if i in done:
                    continue
                try:
                    s = sr.score_one(b, g)
                    rec = {"idx": i, "score_norm": s.get("score_norm"),
                           "score_1_5": s.get("score_1_5"), "label": s.get("label"),
                           "error": s.get("error")}
                except Exception as e:  # noqa: BLE001
                    rec = {"idx": i, "score_norm": None, "score_1_5": None,
                           "label": None, "error": str(e)[:200]}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                w.flush()
                if (i + 1) % 100 == 0:
                    print(f"[sr] {i+1}/{len(pairs)}", flush=True)
    finally:
        try:
            sr.close()
        except Exception:
            pass
    print(f"[sr] 完成，共 {len(rows)} 条", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Merge：读预评分 → 假评分器 → 真实 evaluate_attacks
# ---------------------------------------------------------------------------
def _load_scores(path):
    """返回 dict[(prompt, response)] -> score_record，缺失条带 error。"""
    scores = {}
    if not Path(path).exists():
        raise SystemExit(f"缺少预评分文件 {path}（先跑 --score-hb / --score-sr）")
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            scores[json.loads(l)["idx"]] = json.loads(l)
    return scores


def _load_sr_with_fallback(root_p, rows):
    """优先 p2c_sr_scores.jsonl；覆盖不足时从现有 CSV 复用真实 sr（需
    scorer_fallback != sr_failed 且行数一致），否则返回 None 提示先跑 --score-sr。"""
    p = root_p / "results" / "p2c_sr_scores.jsonl"
    if p.exists():
        recs = _load_scores(p)
        if len(recs) >= int(len(rows) * 0.95):
            return recs, "p2c_sr_scores.jsonl"
        print(f"[merge] sr jsonl 覆盖不足 {len(recs)}/{len(rows)}，尝试 CSV 复用", flush=True)
    csv_p = root_p / "report" / "adaptive_attack_results.csv"
    if csv_p.exists():
        import pandas as pd
        try:
            dfc = pd.read_csv(csv_p)
        except Exception as e:  # noqa: BLE001
            print(f"[merge] CSV 读取失败: {e}", flush=True)
            dfc = None
        if dfc is not None and len(dfc) == len(rows):
            fb = str(dfc["scorer_fallback"].iloc[0]) if "scorer_fallback" in dfc else ""
            if "sr_failed" not in fb:
                recs = {}
                ok = 0
                for i in range(len(rows)):
                    v = dfc["sr_score"].iloc[i]
                    import numpy as np
                    if pd.notna(v) and float(v) >= 0:
                        recs[i] = {"score_norm": float(v), "score_1_5": None,
                                   "label": int(float(v) >= 0.5),
                                   "error": None}
                        ok += 1
                    else:
                        recs[i] = {"score_norm": None, "score_1_5": None,
                                   "label": None, "error": "csv_na"}
                if ok >= int(len(rows) * 0.95):
                    print(f"[merge] CSV 复用 sr {ok}/{len(rows)}（fallback={fb}）", flush=True)
                    return recs, "csv"
    return None, None


def _prefill_caches(root_p, n, hb_recs, sr_recs):
    """用真实评分覆写正确的全量断点缓存（logs/p2c_{hb,sr}_cache.jsonl）。

    v6.6.8-fix 的评分循环只把缺失行子集传给 score_batch 并用**实际行索引**
    映射回缓存（_idxs），而早期 FakeScorer 用 enumerate 位置索引 → 与真实行索引
    错位 → 污染了缓存。方案：直接按行索引覆写全量正确缓存（i=0..n-1，
    行序 == sorted(glob) 全量攻击行序 == evaluate_attacks 的 df 行序），
    使 _hb_missing/_sr_missing 为空 → 评分器完全不被调用 → 索引问题消失。

    格式与 evaluate_attacks 写缓存完全一致：
      hb: {"i": <i>, "label": <bool>, "prob_yes": <float>}
      sr: {"i": <i>, "score": <score_norm>}
    失败/缺失行写 None 语义（label None / score None）→ 该行仍进 missing
    列表 → 若真有缺口，下方 FakeScorer 会抛错（fail loud），不会静默错位。
    """
    hb_path = root_p / "logs" / "p2c_hb_cache.jsonl"
    sr_path = root_p / "logs" / "p2c_sr_cache.jsonl"
    for p in (hb_path, sr_path):
        if p.exists():
            p.rename(str(p) + ".prefill_bak")  # 保留被污染副本供核对
    with hb_path.open("w", encoding="utf-8") as cf:
        for i in range(n):
            rec = hb_recs.get(i)
            if rec is None or rec.get("error"):
                cf.write(json.dumps({"i": i, "label": None,
                                     "prob_yes": None}, ensure_ascii=False) + "\n")
            else:
                cf.write(json.dumps({"i": i,
                                     "label": bool(rec.get("label")),
                                     "prob_yes": rec.get("prob_yes", 0.5)},
                                    ensure_ascii=False) + "\n")
    with sr_path.open("w", encoding="utf-8") as cf:
        for i in range(n):
            rec = sr_recs.get(i)
            if rec is None or rec.get("error"):
                cf.write(json.dumps({"i": i, "score": None}, ensure_ascii=False) + "\n")
            else:
                cf.write(json.dumps({"i": i,
                                     "score": rec.get("score_norm")},
                                    ensure_ascii=False) + "\n")
    print(f"[merge] 已覆写全量缓存: hb {hb_path.name} / sr {sr_path.name}"
          f"（各 {n} 条，行序 == df 行序）", flush=True)


def merge(root, cfg):
    import scorer_utils
    import stage_p2c_adaptive

    root_p = Path(root)
    rows = _load_all_attacks(root_p)
    hb_recs = _load_scores(root_p / "results" / "p2c_hb_scores.jsonl")
    sr_recs, sr_src = _load_sr_with_fallback(root_p, rows)
    missing_hb = [i for i in range(len(rows)) if i not in hb_recs]
    if missing_hb:
        print(f"[merge] 警告: hb 缺 {len(missing_hb)} 条（前 5: {missing_hb[:5]}）", flush=True)
        print("[merge] 需先完成 --score-hb", flush=True)
        return 2
    if sr_recs is None:
        print("[merge] 无可用 sr 评分 → 需先跑 --score-sr", flush=True)
        return 2
    missing_sr = [i for i in range(len(rows)) if i not in sr_recs]
    if missing_sr:
        print(f"[merge] 警告: sr 缺 {len(missing_sr)} 条（前 5: {missing_sr[:5]}）", flush=True)
    print(f"[merge] sr 来源: {sr_src}", flush=True)

    # 核心修复：覆写正确的全量断点缓存 → evaluate_attacks 全命中跳过评分器。
    _prefill_caches(root_p, len(rows), hb_recs, sr_recs)

    # 兜底假评分器（fail loud）：正常路径缓存全满不会调用；万一缓存真有缺口被调用，
    # 直接抛错而不是返回错位分数（v6.6.8 缺失行子集传入，位置索引必然错位）。
    class FakeScorer:
        def __init__(self, recs, rows, kind):
            self._recs = recs
            self._rows = rows
            self._kind = kind

        def score_batch(self, pairs, batch_size=4):
            raise RuntimeError(
                f"[merge] FakeScorer({self._kind}) 被调用——断点缓存应有全覆盖。"
                f"len(pairs)={len(pairs)}；请检查 {len(self._rows)} 行的缓存覆盖"
                f"（若为缺口则不能静默错位，需先补评真实分数）。")

        def close(self):
            pass

    scorer_utils.get_harmbench = lambda c, log: FakeScorer(hb_recs, rows, "hb")
    scorer_utils.StrongRejectScorer = lambda *a, **k: FakeScorer(sr_recs, rows, "sr")

    # v6.8-fix 漏修点（stage_p2c_adaptive.py:464）：msrf_score_response 内
    # _has_audio = bool(audio_path and Path(audio_path).exists()) 对文本攻击行的
    # NaN audio_path（NaN truthy 逃过 not 守卫）直接 Path(NaN) 抛 TypeError。
    # 对方 v6.8-fix 只修了 stage_p2_msrf.py 的 extract_acoustic_features，未覆盖
    # 此第二处。对方源码不改，这里运行时包装清洗：NaN/非 str/非 PathLike → None
    # （extract_acoustic_features(None) 已返回空特征；_has_audio=False → mask=1.0
    # 缺失指示，语义与训练口径一致）。真音频攻击的 str 路径原样透传。
    import math as _math
    import os as _os
    _orig_msrf = stage_p2c_adaptive.msrf_score_response

    def _safe_msrf_score_response(fusion, response, audio_path,
                                  sr_score, hb_prob, disagreement):
        if isinstance(audio_path, float) and _math.isnan(audio_path):
            audio_path = None
        elif not isinstance(audio_path, (str, _os.PathLike)):
            audio_path = None
        return _orig_msrf(fusion, response, audio_path,
                          sr_score, hb_prob, disagreement)

    stage_p2c_adaptive.msrf_score_response = _safe_msrf_score_response
    print("[merge] 已包装 msrf_score_response（清洗 NaN audio_path，补 v6.8-fix 漏修点）",
          flush=True)

    # 真实 evaluate_attacks 走完整下游（含评分失败 fallback 披露 + TPR + md）
    import logging
    from common_utils import JsonlLogger
    log = logging.getLogger("p2c_backfill")
    logging.basicConfig(level=logging.INFO)
    elog = JsonlLogger(str(root_p / "logs" / "errors.jsonl"))
    code = stage_p2c_adaptive.evaluate_attacks(root_p, cfg, log, elog)
    print(f"[merge] evaluate_attacks 返回 code={code}", flush=True)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--score-hb", action="store_true")
    ap.add_argument("--score-sr", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["workdir"]).expanduser()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.score_hb:
        return score_hb(root, cfg)
    if args.score_sr:
        return score_sr(root, cfg)
    if args.merge:
        return merge(root, cfg)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
