#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S25：N/E_t × 仇恨言论 交互正式检验（双生成器，CPU，2026-08-14）。

动机：S22/S24 显示仇恨言论是被叙事结构（N）与叙事框架（E_t）放大最强烈的
攻击族（E4B N Δ=+0.344、E_t Δ=+0.30 vs 其他族 +0.02~0.04）。该调节目前仅以
"各族效应 CI 不重叠"作描述性证据。本实验做**正式交互检验**：对每个生成器
（E2B/E4B 文本）bootstrap 交互量 = Δ(仇恨言论) − Δ(非仇恨言论)，给出 CI 与
p 值（交互为正且 CI 排除 0 → 仇恨言论显著调节 N/E_t 效应）。

bootstrap 采用**分层有放回**（仇恨查询与其它查询分别重采样，稳定小样本族）。

零人工标注、纯 CPU、全缓存、只写 s25_* 输出。

用法：python gpu1_s25_hate_interaction.py [--B 2000] [--smoke]
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
    print("[s25 %s] %s" % (Path(__file__).stem, m), flush=True)


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


def _load_qfam(root):
    qfam = {}
    for src in ("queries_v2.jsonl", "benign_requests_v1.jsonl"):
        p = root / "data" / src
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            try:
                o = json.loads(line)
                qq = str(o.get("query_id") or "").strip()
                cat = str(o.get("category") or "").strip()
                if qq and cat:
                    qfam[qq] = cat
            except Exception:  # noqa: BLE001
                continue
    return qfam


def _by_query(rows, labels, dim, v0, v1):
    """返回 {qid: (y0均值, n0, y1均值, n1)}，仅含标签非空的行。"""
    acc = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])
    for i, r in enumerate(rows):
        lab = labels[i]
        if np.isnan(lab):
            continue
        q = r["query_id"]
        a = acc[q]
        if r[dim] == v0:
            a[0] += lab
            a[1] += 1
        elif r[dim] == v1:
            a[2] += lab
            a[3] += 1
    return acc


def _effect_from_qstats(acc, qids):
    s0 = sum(acc[q][0] for q in qids)
    n0 = sum(acc[q][1] for q in qids)
    s1 = sum(acc[q][2] for q in qids)
    n1 = sum(acc[q][3] for q in qids)
    if n0 == 0 or n1 == 0:
        return None
    return s1 / n1 - s0 / n0


def _interaction(rows, labels, dim, v0, v1, hate, B, seed):
    acc = _by_query(rows, labels, dim, v0, v1)
    hate_q = [q for q in acc if hate[q]]
    other_q = [q for q in acc if not hate[q]]
    if not hate_q or not other_q:
        return None
    rng = np.random.RandomState(seed)
    diffs = np.empty(B)
    for b in range(B):
        hq = [hate_q[rng.randint(len(hate_q))] for _ in hate_q]
        oq = [other_q[rng.randint(len(other_q))] for _ in other_q]
        dh = _effect_from_qstats(acc, hq)
        do = _effect_from_qstats(acc, oq)
        if dh is None or do is None:
            diffs[b] = np.nan
            continue
        diffs[b] = dh - do
    ok = diffs[~np.isnan(diffs)]
    if len(ok) < B // 2:
        return None
    lo = float(np.percentile(ok, 2.5))
    hi = float(np.percentile(ok, 97.5))
    # p = 交互为 0 的概率（双侧：bootstrap 分布中跨 0 的比例经对称化）
    # 用效应 > 0 比例的极小尾（单侧正交互）保守估计 + 双侧
    p_pos = float(np.mean(ok > 0))
    p_two = 2.0 * min(p_pos, 1.0 - p_pos)
    return {"interaction": round(float(np.mean(ok)), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "p_two_sided": round(min(p_two, 1.0), 4),
            "n_hate_q": len(hate_q), "n_other_q": len(other_q),
            "excl_zero": bool(lo > 0 or hi < 0)}


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

    qfam = _load_qfam(root)
    dims = [("N", 0, 1), ("E_t", 0, 1)]
    gen_sets = {}

    # E2B 侧（i 键控）
    e2b = [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e2b_responses.jsonl",
        encoding="utf-8")]
    n2 = len(e2b)
    jb = np.array([0.0] * n2)
    js = np.array([0.0] * n2)
    # 读 i 键控缓存
    cjb = {}
    if (cache_dir / "judge_big.jsonl").exists():
        for line in (cache_dir / "judge_big.jsonl").open(encoding="utf-8"):
            rec = json.loads(line)
            cjb[rec["i"]] = rec["label"]
    cjs = {}
    if (cache_dir / "judge_small.jsonl").exists():
        for line in (cache_dir / "judge_small.jsonl").open(encoding="utf-8"):
            rec = json.loads(line)
            cjs[rec["i"]] = rec["label"]
    dual2 = np.array([cjb.get(i, np.nan) if (cjb.get(i) is not None
                     and cjs.get(i) == cjb.get(i)) else np.nan
                      for i in range(n2)], dtype=float)
    hate2 = {r["query_id"]: (qfam.get(r["query_id"], "unk") == "仇恨言论")
             for r in e2b}
    gen_sets["E2B"] = (e2b, dual2, hate2)

    # E4B 文本侧（rid 键控）
    e4b = [r for r in [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl",
        encoding="utf-8")] if r.get("modality") == "text"]
    rid = [r["response_id"] for r in e4b]
    cjbb = {}
    if (cache_dir / "s17_e4b_text_judge_big.jsonl").exists():
        for line in (cache_dir / "s17_e4b_text_judge_big.jsonl").open(
                encoding="utf-8"):
            rec = json.loads(line)
            cjbb[rec["rid"]] = rec["label"]
    cjss = {}
    if (cache_dir / "s17_e4b_text_judge_small.jsonl").exists():
        for line in (cache_dir / "s17_e4b_text_judge_small.jsonl").open(
                encoding="utf-8"):
            rec = json.loads(line)
            cjss[rec["rid"]] = rec["label"]
    dual4 = np.array([
        cjbb.get(x, np.nan) if (cjbb.get(x) is not None
                                and cjss.get(x) == cjbb.get(x)) else np.nan
        for x in rid], dtype=float)
    hate4 = {r["query_id"]: (qfam.get(r["query_id"], "unk") == "仇恨言论")
             for r in e4b}
    gen_sets["E4B_text"] = (e4b, dual4, hate4)

    res = {}
    for gname, (rows, labels, hate) in gen_sets.items():
        _log("%s: n=%d dual非空=%d hate=%d" % (
            gname, len(rows), int(np.sum(~np.isnan(labels))),
            int(sum(1 for v in hate.values() if v))))
        res[gname] = {}
        for dim, v0, v1 in dims:
            if args.smoke and dim != "N":
                continue
            it = _interaction(rows, labels, dim, v0, v1, hate, B,
                              seed=20260818 + (0 if gname.startswith("E2B")
                                               else 1))
            res[gname][dim] = it
            _log("%s %s×仇恨言论: %s" % (gname, dim, json.dumps(
                it, ensure_ascii=False) if it else None))

    out = {
        "stage": "S25", "date": "2026-08-14",
        "purpose": ("N/E_t × 仇恨言论 交互正式检验：交互量 = Δ(仇恨言论) − "
                    "Δ(非仇恨言论)，分层有放回 bootstrap（B=2000），双生成器"),
        "interactions": res,
        "note": ("交互量 CI 排除 0（双侧 p<0.05）→ 仇恨言论显著调节该维效应。"
                 "仇恨言论仅 14 查询（336 单元），探索性定位。"),
    }
    if args.smoke:
        print(json.dumps(_json_safe(out), ensure_ascii=False, indent=2))
        _log("smoke 完成（不落盘）")
        return 0
    with open(out_dir / "s25_hate_interaction.json", "w",
              encoding="utf-8") as f:
        json.dump(_json_safe(out), f, ensure_ascii=False, indent=2)
    _log("已落盘 s25_hate_interaction.json")
    print(json.dumps(_json_safe(out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
