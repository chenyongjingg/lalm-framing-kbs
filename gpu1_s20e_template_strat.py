#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S20e：N/E_t/R 按模板分层（t2 停滞伪影检验，CPU，2026-08-14）。

动机：S19 已报 E_t 按模板分层（t0 +0.071 / t1 +0.044 / t2 −0.015），但 S20 的
三维分解未按模板分层。t2 是停滞热点（S12b：E_t=1/t2 停滞 56.3%），审稿人必问：
"协议 RQ（N）主效应是否主要由某个模板驱动？t2 停滞伪影是否污染 N？"。
本实验按模板 t0/t1/t2 分层报告 N/E_t/R 效应（dual_judge + qwen32 口径），
并给出"效应是否跨模板稳健"判定。

零人工标注、纯 CPU、全缓存、只写 s20e_* 输出。

用法：python gpu1_s20e_template_strat.py [--B 1000] [--smoke]
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
    print("[s20e %s] %s" % (Path(__file__).stem, m), flush=True)


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
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                out[rec["i"]] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
    return np.array([out.get(i, np.nan) for i in range(n)], dtype=float)


def _bootstrap_dim(rows, labels, dim, val0, val1, seed=20260815, B=1000):
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


def _stratum(rows, labels, mask, dim, v0, v1, B):
    sub_i = [i for i in range(len(rows)) if mask[i]]
    e0 = [labels[i] for i in sub_i
          if rows[i][dim] == v0 and not np.isnan(labels[i])]
    e1 = [labels[i] for i in sub_i
          if rows[i][dim] == v1 and not np.isnan(labels[i])]
    p0 = float(np.mean(e0)) if e0 else None
    p1 = float(np.mean(e1)) if e1 else None
    delta = (p1 - p0) if (p0 is not None and p1 is not None) else None
    # 该模板内 query 聚类 bootstrap（对子集）
    sub_rows = [rows[i] for i in sub_i]
    sub_lab = np.array([labels[i] for i in sub_i])
    ci = _bootstrap_dim(sub_rows, sub_lab, dim, v0, v1, B=B)
    return {"n0": len(e0), "n1": len(e1),
            "pos0": round(p0, 4) if p0 is not None else None,
            "pos1": round(p1, 4) if p1 is not None else None,
            "delta": round(delta, 4) if delta is not None else None,
            "ci95": ci}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    B = 20 if args.smoke else args.B

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "scorers_cache"

    rows = [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e2b_responses.jsonl",
        encoding="utf-8")]
    n = len(rows)
    jb = _read_index_cache(cache_dir / "judge_big.jsonl", n)
    js = _read_index_cache(cache_dir / "judge_small.jsonl", n)
    qw = _read_index_cache(cache_dir / "qwen32.jsonl", n)
    dual = np.where(~np.isnan(jb) & ~np.isnan(js) & (jb == js), jb, np.nan)
    _log("E2B=%d dual 非空=%d qwen32 非空=%d" % (
        n, int(np.sum(~np.isnan(dual))), int(np.sum(~np.isnan(qw)))))

    dims = [("E_t", 0, 1), ("N", 0, 1), ("R", 0, 1)]
    srcs = {"dual_judge": dual, "qwen32": qw}
    templates = [0, 1, 2]

    res = {}
    for dim, v0, v1 in dims:
        if args.smoke and dim != "N":
            continue
        res[dim] = {}
        # 总体（池化）
        res[dim]["overall"] = {}
        for s, lab in srcs.items():
            res[dim]["overall"][s] = _stratum(rows, lab, np.ones(n, bool),
                                              dim, v0, v1, B)
        # 按模板
        for t in templates:
            mask = np.array([r["template_idx"] == t for r in rows])
            res[dim]["t%d" % t] = {}
            for s, lab in srcs.items():
                res[dim]["t%d" % t][s] = _stratum(rows, lab, mask, dim,
                                                  v0, v1, B)
            _log("dim=%s t%d: dual Δ=%s qw Δ=%s" % (
                dim, t, res[dim]["t%d" % t]["dual_judge"]["delta"],
                res[dim]["t%d" % t]["qwen32"]["delta"]))
        # 跨模板 Δ 的极差（衡量模板异质性）
        deltas_dual = [res[dim]["t%d" % t]["dual_judge"]["delta"]
                       for t in templates]
        res[dim]["template_delta_spread_dual"] = {
            "max": max(deltas_dual), "min": min(deltas_dual),
            "range": round(max(deltas_dual) - min(deltas_dual), 4)}
        _log("dim=%s 模板 Δ 极差(dual)=%s" % (
            dim, res[dim]["template_delta_spread_dual"]))

    out = {
        "stage": "S20e", "date": "2026-08-14",
        "purpose": ("N/E_t/R 按模板分层：t2 停滞热点（E_t=1/t2 停滞 56.3%，S12b）"
                    "是否污染协议 RQ（N）与 E_t/R 效应"),
        "by_template": res,
        "note": ("若 N/E_t 效应在 t0/t1/t2 各层内均同向（尤其 t2 亦显著）→ 主效应"
                 "非模板伪影；若 t2 层消失/反向 → 如实披露 t2 停滞污染（须结合 "
                 "S12/S18 停滞量化解读）。"),
    }
    if args.smoke:
        print(json.dumps(_json_safe(out), ensure_ascii=False, indent=2))
        _log("smoke 完成（不落盘）")
        return 0
    with open(out_dir / "s20e_template_strat.json", "w",
              encoding="utf-8") as f:
        json.dump(_json_safe(out), f, ensure_ascii=False, indent=2)
    _log("已落盘 s20e_template_strat.json")
    print(json.dumps(_json_safe(out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
