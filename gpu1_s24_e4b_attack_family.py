#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S24：E4B 文本 N/E_t/R 效应按攻击族分层（跨生成器复现 S22，CPU，2026-08-14）。

动机：S22 在 E2B 侧证明 N 效应跨攻击族泛化（5/5 同向、3/5 显著）。S23 已显示
E4B 与 E2B 在模板维度存在差异（N 的 t2 归零 E2B 特有，E4B 全模板显著）。
审稿人必问："跨攻击族泛化是否在权威生成器 E4B 上复现？还是 E2B 特有？"
本实验在 E4B 文本 3600 上镜像 S22：按攻击族分层报告 N/E_t/R 效应
（dual_judge + qwen32，rid 键控缓存），并输出 E2B vs E4B 跨生成器对比。

零人工标注、纯 CPU、全缓存、只写 s24_* 输出。

用法：python gpu1_s24_e4b_attack_family.py [--B 1000] [--smoke]
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
    print("[s24 %s] %s" % (Path(__file__).stem, m), flush=True)


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


def _read_rid_cache(p):
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                out[rec["rid"]] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
    return out


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


def _bootstrap_dim(rows, labels, dim, val0, val1, seed=20260817, B=1000):
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

    rows = [r for r in [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl",
        encoding="utf-8")] if r.get("modality") == "text"]
    n = len(rows)
    qfam = _load_qfam(root)
    fams = [qfam.get(r["query_id"], "unk") for r in rows]
    unk = sum(1 for f in fams if f.startswith("unk"))
    _log("E4B文本=%d 攻击族映射未命中=%d" % (n, unk))

    rid = [r["response_id"] for r in rows]
    jb_c = _read_rid_cache(cache_dir / "s17_e4b_text_judge_big.jsonl")
    js_c = _read_rid_cache(cache_dir / "s17_e4b_text_judge_small.jsonl")
    qw_c = _read_rid_cache(cache_dir / "s17_e4b_text_qwen32.jsonl")
    jb = np.array([jb_c.get(x, np.nan) for x in rid], dtype=float)
    js = np.array([js_c.get(x, np.nan) for x in rid], dtype=float)
    qw = np.array([qw_c.get(x, np.nan) for x in rid], dtype=float)
    dual = np.where(~np.isnan(jb) & ~np.isnan(js) & (jb == js), jb, np.nan)
    _log("dual 非空=%d qwen32 非空=%d" % (
        int(np.sum(~np.isnan(dual))), int(np.sum(~np.isnan(qw)))))

    order = sorted({f for f in fams})
    dims = [("N", 0, 1), ("E_t", 0, 1), ("R", 0, 1)]
    srcs = {"dual_judge": dual, "qwen32": qw}

    res = {}
    for dim, v0, v1 in dims:
        res[dim] = {"overall": {}}
        for s, lab in srcs.items():
            res[dim]["overall"][s] = _stratum(rows, lab, np.ones(n, bool),
                                              dim, v0, v1, B)
        res[dim]["by_family"] = {}
    for fam in order:
        mask = np.array([f == fam for f in fams])
        ncells = int(mask.sum())
        nq = len({r["query_id"] for i, r in enumerate(rows) if mask[i]})
        for dim, v0, v1 in dims:
            if args.smoke and dim != "N":
                continue
            fam_blk = {"n_cells": ncells, "n_queries": nq}
            for s, lab in srcs.items():
                fam_blk[s] = _stratum(rows, lab, mask, dim, v0, v1, B)
            res[dim]["by_family"][fam] = fam_blk
            _log("fam=%s dim=%s dual Δ=%s CI=%s" % (
                fam, dim, fam_blk["dual_judge"]["delta"],
                fam_blk["dual_judge"]["ci95"]))
    for dim, v0, v1 in dims:
        if args.smoke and dim != "N":
            continue
        dels = [(f, res[dim]["by_family"][f]["dual_judge"]["delta"])
                for f in order
                if res[dim]["by_family"][f]["dual_judge"]["delta"] is not None]
        pos = [d for _, d in dels if d > 0]
        neg = [d for _, d in dels if d < 0]
        n_sig = 0
        n_sig_pos = 0
        for f, d in dels:
            ci = res[dim]["by_family"][f]["dual_judge"]["ci95"]
            if ci is not None and (ci[0] > 0 or ci[1] < 0):
                n_sig += 1
                if ci[0] > 0:
                    n_sig_pos += 1
        res[dim]["across_family"] = {
            "n_families_valid": len(dels),
            "n_pos": len(pos), "n_neg": len(neg),
            "n_sig": n_sig, "n_sig_pos": n_sig_pos,
            "all_same_sign": (len(pos) == len(dels)) or (len(neg) == len(dels)),
            "note": ("跨族方向一致性判定：若多数族同向且若干族显著 → "
                     "效应非单攻击族驱动。")}

    out = {
        "stage": "S24", "date": "2026-08-14",
        "purpose": ("E4B 文本 N/E_t/R 效应按攻击族分层：跨生成器复现 S22，"
                    "验证跨攻击族泛化在权威生成器 E4B 上成立"),
        "family_order": order,
        "coverage": {"n_cells": n, "n_families": len(order),
                     "missing_families": ["网络攻击", "良性请求"]},
        "by_dim": res,
        "note": ("E4B 侧跨族方向与显著性是否复现 E2B（S22：N 5/5 同向 3/5 显著）"
                 "；若差异须与 S23 模板差异一并如实披露。"),
    }
    if args.smoke:
        print(json.dumps(_json_safe(out), ensure_ascii=False, indent=2))
        _log("smoke 完成（不落盘）")
        return 0
    with open(out_dir / "s24_e4b_attack_family.json", "w",
              encoding="utf-8") as f:
        json.dump(_json_safe(out), f, ensure_ascii=False, indent=2)
    _log("已落盘 s24_e4b_attack_family.json")
    print(json.dumps(_json_safe(out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
