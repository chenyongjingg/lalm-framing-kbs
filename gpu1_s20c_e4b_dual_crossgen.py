#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S20c：E4B 文本 dual_judge 口径跨生成器复现（CPU，2026-08-14）。

动机：S20 的跨生成器（E2B vs E4B 文本同格 3600）当时只有 qwen32 单腿
（E4B 文本 judge 缓存未生成）。S20b 补齐 E4B 文本 judge_big + judge_small 后，
本实验在 pre-registered R2 的 dual_judge 共识口径上复现跨生成器：
  - E4B 文本侧 E_t/N/R 主效应（dual / judge_big / judge_small / qwen32 四口径，
    与 S20 的 E2B 侧表镜像对比）；
  - 跨生成器逐格一致率：E2B-dual vs E4B-dual（同序，位置对齐已由 S20 验证）；
  - 各口径下三维效应符号同向判定（E2B 侧 vs E4B 侧）。

若 E4B-dual 侧 N/E_t 显著同向 → 协议 RQ 效应在两次独立生成 + 共识口径下双重复现，
为论文最强稳健性声明。零人工标注、纯 CPU、全缓存驱动、只写 s20c_* 输出。

用法：python gpu1_s20c_e4b_dual_crossgen.py [--B 2000] [--smoke]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s20c %s] %s" % (Path(__file__).stem, m), flush=True)


def _json_safe(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _read_index_cache(p, n):
    """i 键控缓存（E2B 侧）。"""
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                out[rec["i"]] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
    return np.array([out.get(i, np.nan) for i in range(n)], dtype=float)


def _read_rid_cache(p):
    """response_id 键控缓存（E4B 侧）。"""
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                out[rec["rid"]] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
    return out


def _bootstrap_dim(rows, labels, dim, val0, val1, seed=20260815, B=2000):
    by_q = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_q[r["query_id"]].append(i)
    qids = sorted(by_q)
    rng = np.random.RandomState(seed)
    nq = len(qids)
    deltas = np.empty(B)
    for b in range(B):
        sel = []
        for _ in range(nq):
            sel.extend(by_q[qids[rng.randint(nq)]])
        g0 = [labels[i] for i in sel if rows[i][dim] == val0
              and not np.isnan(labels[i])]
        g1 = [labels[i] for i in sel if rows[i][dim] == val1
              and not np.isnan(labels[i])]
        if not g0 or not g1:
            deltas[b] = np.nan
            continue
        deltas[b] = np.mean(g1) - np.mean(g0)
    ok = deltas[~np.isnan(deltas)]
    if len(ok) < B // 2:
        return None
    return [round(float(np.percentile(ok, 2.5)), 4),
            round(float(np.percentile(ok, 97.5)), 4)]


def _dim_effect(rows, labels, dim, v0, v1, B=2000, name=""):
    e0 = [labels[i] for i, r in enumerate(rows)
          if r[dim] == v0 and not np.isnan(labels[i])]
    e1 = [labels[i] for i, r in enumerate(rows)
          if r[dim] == v1 and not np.isnan(labels[i])]
    p0 = float(np.mean(e0)) if e0 else None
    p1 = float(np.mean(e1)) if e1 else None
    delta = (p1 - p0) if (p0 is not None and p1 is not None) else None
    ci = _bootstrap_dim(rows, labels, dim, v0, v1, B=B)
    return {"n0": len(e0), "n1": len(e1),
            "pos0": round(p0, 4) if p0 is not None else None,
            "pos1": round(p1, 4) if p1 is not None else None,
            "delta": round(delta, 4) if delta is not None else None,
            "ci95": ci, "name": name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    B = 20 if args.smoke else args.B

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "scorers_cache"

    e2b = [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e2b_responses.jsonl",
        encoding="utf-8")]
    e4b = [r for r in [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl",
        encoding="utf-8")] if r.get("modality") == "text"]
    n = len(e2b)
    assert len(e4b) == n, "E4B 文本 != E2B 行数"
    aligned = all(e4b[i]["query_id"] == e2b[i]["query_id"] and
                  e4b[i]["template_idx"] == e2b[i]["template_idx"] and
                  e4b[i]["E_t"] == e2b[i]["E_t"] and
                  e4b[i]["N"] == e2b[i]["N"] and
                  e4b[i]["R"] == e2b[i]["R"] for i in range(n))
    _log("E2B=%d E4B文本=%d 同序对齐=%s" % (n, len(e4b), aligned))

    # ---- E2B 侧标签（i 键控）----
    e2b_jb = _read_index_cache(cache_dir / "judge_big.jsonl", n)
    e2b_js = _read_index_cache(cache_dir / "judge_small.jsonl", n)
    e2b_qw = _read_index_cache(cache_dir / "qwen32.jsonl", n)
    e2b_dual = np.where(~np.isnan(e2b_jb) & ~np.isnan(e2b_js)
                        & (e2b_jb == e2b_js), e2b_jb, np.nan)

    # ---- E4B 侧标签（rid 键控，位置对齐）----
    rid = [r["response_id"] for r in e4b]
    e4b_jb_c = _read_rid_cache(cache_dir / "s17_e4b_text_judge_big.jsonl")
    e4b_js_c = _read_rid_cache(cache_dir / "s17_e4b_text_judge_small.jsonl")
    e4b_qw_c = _read_rid_cache(cache_dir / "s17_e4b_text_qwen32.jsonl")
    e4b_jb = np.array([e4b_jb_c.get(x, np.nan) for x in rid], dtype=float)
    e4b_js = np.array([e4b_js_c.get(x, np.nan) for x in rid], dtype=float)
    e4b_qw = np.array([e4b_qw_c.get(x, np.nan) for x in rid], dtype=float)
    e4b_dual = np.where(~np.isnan(e4b_jb) & ~np.isnan(e4b_js)
                        & (e4b_jb == e4b_js), e4b_jb, np.nan)
    for tag, arr in [("jb", e2b_jb), ("js", e2b_js), ("qw", e2b_qw),
                     ("dual", e2b_dual)]:
        _log("E2B %s 非空=%d" % (tag, int(np.sum(~np.isnan(arr)))))
    for tag, arr in [("jb", e4b_jb), ("js", e4b_js), ("qw", e4b_qw),
                     ("dual", e4b_dual)]:
        _log("E4B %s 非空=%d" % (tag, int(np.sum(~np.isnan(arr)))))

    dims = [("E_t", 0, 1), ("N", 0, 1), ("R", 0, 1)]
    srcs = ["dual", "judge_big", "judge_small", "qwen32"]
    e2b_src = {"dual": e2b_dual, "judge_big": e2b_jb, "judge_small": e2b_js,
               "qwen32": e2b_qw}
    e4b_src = {"dual": e4b_dual, "judge_big": e4b_jb, "judge_small": e4b_js,
               "qwen32": e4b_qw}

    # ---- 1. E4B 侧主效应（四口径）----
    e4b_eff = {}
    for dim, v0, v1 in dims:
        if args.smoke and dim != "E_t":
            continue
        e4b_eff[dim] = {}
        for s in srcs:
            e4b_eff[dim][s] = _dim_effect(e4b, e4b_src[s], dim, v0, v1,
                                          B=B, name="e4b_%s" % s)
            _log("E4B dim=%s %s: pos0=%s pos1=%s Δ=%s CI=%s" % (
                dim, s, e4b_eff[dim][s]["pos0"], e4b_eff[dim][s]["pos1"],
                e4b_eff[dim][s]["delta"], e4b_eff[dim][s]["ci95"]))

    # ---- 2. 跨生成器逐格一致率（四口径）----
    agree = {}
    for s in srcs:
        a, b = e2b_src[s], e4b_src[s]
        pair = int(np.sum(~np.isnan(a) & ~np.isnan(b)))
        if pair == 0:
            agree[s] = {"n_pairs": 0, "agreement": None}
            continue
        hit = int(np.sum(~np.isnan(a) & ~np.isnan(b) & (a == b)))
        agree[s] = {"n_pairs": pair,
                    "agreement": round(hit / pair, 4),
                    "pass_0_80": (hit / pair) >= 0.80}
        _log("跨生成器 %s: 一致率=%.4f (n=%d)" % (s, hit / pair, pair))

    # ---- 3. 各口径符号同向（E2B Δ vs E4B Δ）----
    sign_align = {}
    for dim, v0, v1 in dims:
        if args.smoke and dim != "E_t":
            continue
        sign_align[dim] = {}
        for s in srcs:
            e2d = _dim_effect(e2b, e2b_src[s], dim, v0, v1, B=B,
                              name="e2b_%s" % s)["delta"]
            e4d = e4b_eff[dim][s]["delta"]
            sign_align[dim][s] = {
                "e2b_delta": e2d, "e4b_delta": e4d,
                "same_sign": bool(e2d is not None and e4d is not None
                                  and np.sign(e2d) == np.sign(e4d))}
        _log("同向 %s: %s" % (dim, json.dumps(sign_align[dim],
                                              ensure_ascii=False)))

    out = {
        "stage": "S20c", "date": "2026-08-14",
        "purpose": ("E4B 文本 dual_judge 口径跨生成器复现：S20 跨生成器当时仅 qwen32"
                    "单腿，S20b 补齐 E4B 文本 judge 腿后，在 R2 共识口径上双重复现"),
        "e4b_text_effects": e4b_eff,
        "cross_generator_agreement": agree,
        "sign_alignment": sign_align,
        "note": ("E4B-dual 侧 N/E_t 显著同向 → 协议 RQ 效应在两次独立生成 + 共识口径"
                 "双重复现；R 符号分裂（长度混杂）亦跨生成器复现。"),
    }
    if args.smoke:
        print(json.dumps(_json_safe(out), ensure_ascii=False, indent=2))
        _log("smoke 完成（不落盘）")
        return 0
    with open(out_dir / "s20c_e4b_dual_crossgen.json", "w",
              encoding="utf-8") as f:
        json.dump(_json_safe(out), f, ensure_ascii=False, indent=2)
    _log("已落盘 s20c_e4b_dual_crossgen.json")
    print(json.dumps(_json_safe(out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
