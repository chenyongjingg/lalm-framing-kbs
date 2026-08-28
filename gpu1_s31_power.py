#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S31：功效分析 / 最小可检测效应（CPU，2026-08-14）。

动机：S28 将用 Qwen2-Audio 异族生成器在 E4B 设计子集上复现 N/E_t 效应。复现
要成功，抽样量必须给足功效。本实验用真实 E4B 文本数据的查询级聚类结构（同
query 内标签相关），模拟论文所用的 query 聚类 bootstrap 检验，估计不同
n_queries 下各效应量（总效应 = 真实 + 注入）的功效，并给最小可检测效应（MDE，
80% 功效）。

设计（匹配 S28 的 N 效应测试结构）：
  - N0 单元 = combo (E_t=0, N=0, R=0) × 3 模板；N1 = (E_t=0, N=1, R=0) × 3。
  - 每个 query 有真实 pos_rate（dual_judge，E4B 文本），cell 计数 n0/n1。
  - 模拟：按 n_queries 有放回抽 query，对 N1 注入效应 δ（叠加在真实概率上，
    封顶 1），Bernoulli 生成 cell 标签 → pooled Δ → B=1000 次 query 聚类
    bootstrap 得 95%CI → CI 排除 0 即检出。N_sim 次重复统计功效。

输出 s31_power.json + report/s31_power.md；纯 CPU、零生成、只写独立产物。

用法：python gpu1_s31_power.py [--ns 30,60,100,150] [--ds 0,0.02,0.04,0.06,0.08,0.10] [--B 1000] [--nsim 60]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s31] %s" % m, flush=True)


def _load_dual_e4b_text(root):
    """E4B 文本 + dual_judge 标签（rid 键控缓存）。返回行列表与 dual 数组。"""
    rows = [json.loads(l) for l in open(
        root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl",
        encoding="utf-8")]
    rows = [r for r in rows if r.get("modality") == "text"]
    cache_dir = root / "results" / "gpu1_pipeline" / "scorers_cache"
    cjb, cjs = {}, {}
    for key, out in (("s17_e4b_text_judge_big.jsonl", cjb),
                     ("s17_e4b_text_judge_small.jsonl", cjs)):
        p = cache_dir / key
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            rec = json.loads(line)
            out[rec["rid"]] = rec["label"]
    dual = []
    for r in rows:
        rid = r["response_id"]
        b, s = cjb.get(rid), cjs.get(rid)
        if b is not None and s is not None and b == s:
            dual.append(b)
        else:
            dual.append(np.nan)
    return rows, np.asarray(dual, dtype=float)


def _query_aggs(rows, dual):
    """按 query 聚合 combo(0,0,0) 与 (0,1,0) 的 pos/count。返回 dict[q]=(p0,n0,p1,n1)。"""
    acc = {}
    for r, lab in zip(rows, dual):
        if np.isnan(lab):
            continue
        if r["combo"] != [0, 0, 0] and r["combo"] != [0, 1, 0]:
            continue
        q = r["query_id"]
        if q not in acc:
            acc[q] = [0.0, 0, 0.0, 0]  # s0,n0,s1,n1
        a = acc[q]
        if r["combo"] == [0, 0, 0]:
            a[0] += lab
            a[1] += 1
        else:
            a[2] += lab
            a[3] += 1
    # 只保留 n0,n1 都 >=1 的 query
    return {q: (s0 / n0, n0, s1 / n1, n1) for q, (s0, n0, s1, n1) in acc.items()
            if n0 >= 1 and n1 >= 1}


def _sim_power(aggs, n_q, delta, B, n_sim, rng):
    qids = list(aggs.keys())
    p0s = np.array([aggs[q][0] for q in qids])
    n0s = np.array([aggs[q][1] for q in qids])
    p1s = np.array([aggs[q][2] for q in qids])
    n1s = np.array([aggs[q][3] for q in qids])
    p1s_inj = np.minimum(1.0, p1s + delta)
    m = len(qids)
    det = 0
    for _ in range(n_sim):
        idx = rng.randint(0, m, size=n_q)
        y0 = rng.binomial(n0s[idx], p0s[idx]) / n0s[idx]
        y1 = rng.binomial(n1s[idx], p1s_inj[idx]) / n1s[idx]
        obs = y1.sum() / n1s[idx].sum() - y0.sum() / n0s[idx].sum()
        if not np.isfinite(obs):
            continue
        diffs = np.empty(B)
        for b in range(B):
            j = rng.randint(0, n_q, size=n_q)
            diffs[b] = (y1[j].sum() / n1s[idx][j].sum()
                        - y0[j].sum() / n0s[idx][j].sum())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        if lo > 0 or hi < 0:
            det += 1
    return det / n_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--ns", default="30,60,100,150")
    ap.add_argument("--ds", default="0,0.02,0.04,0.06,0.08,0.10")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--nsim", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    ds = [float(x) for x in args.ds.split(",") if x.strip()]

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, dual = _load_dual_e4b_text(root)
    aggs = _query_aggs(rows, dual)
    n_q_pool = len(aggs)
    _log("E4B 文本 dual 有效行=%d，含 (0,0,0)/(0,1,0) 的 query=%d" % (
        int(np.sum(~np.isnan(dual))), n_q_pool))
    # 查询级 pos 方差（ICC 描述）
    v0 = np.var([aggs[q][0] for q in aggs], ddof=1) if n_q_pool > 1 else np.nan
    v1 = np.var([aggs[q][2] for q in aggs], ddof=1) if n_q_pool > 1 else np.nan
    _log("查询级 pos 方差：N0 var=%.4f, N1 var=%.4f" % (v0, v1))

    rng = np.random.RandomState(args.seed)
    rows_out = []
    for n_q in ns:
        if n_q > n_q_pool:
            _log("跳过 n_q=%d（> 池 %d）" % (n_q, n_q_pool))
            continue
        for d in ds:
            pw = _sim_power(aggs, n_q, d, args.B, args.nsim, rng)
            rows_out.append({"n_queries": n_q, "effect_delta": d,
                             "power": round(pw, 4)})
            _log("n_q=%d δ=%+.2f 功效=%.3f" % (n_q, d, pw))

    # MDE：对每个 n_q，最小 δ 使功效 >= 0.80
    mde = []
    for n_q in ns:
        sub = [r for r in rows_out if r["n_queries"] == n_q]
        cand = [r["effect_delta"] for r in sub if r["power"] >= 0.8]
        mde.append({"n_queries": n_q,
                    "mde_0_80": min(cand) if cand else None,
                    "max_power_seen": max(
                        (r["power"] for r in sub), default=None)})

    out = {
        "stage": "S31", "date": "2026-08-14",
        "method": ("真实 E4B 文本查询级聚类结构 + 注入效应 δ（叠加于 N1 真实 "
                   "概率）→ Bernoulli cell 标签 → query 聚类 bootstrap（B=%d, "
                   "N_sim=%d）→ 功效 = CI 排除 0 比例" % (args.B, args.nsim)),
        "n_query_pool": n_q_pool,
        "query_level_pos_var": {"N0": round(v0, 4), "N1": round(v1, 4)},
        "results": rows_out,
        "mde_80": mde,
        "note": ("δ 为 N1 相对 N0 的真实 pos 差（叠加于真实 +0.06 效应之上）。"
                 "δ=0 行反映'仅真实效应'下的检出功效（复现力上限）。"),
    }
    (out_dir / "s31_power.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# S31：功效分析 / 最小可检测效应（CPU）\n",
        "- 结构：真实 E4B 文本 dual_judge 查询级聚类（n_query 池=%d）" % n_q_pool,
        "- 方法：注入效应 δ（叠加于真实概率）→ Bernoulli 标签 → query 聚类 "
        "bootstrap（B=%d, N_sim=%d）" % (args.B, args.nsim),
        "- 查询级 pos 方差：N0=%.4f, N1=%.4f\n" % (v0, v1),
        "## 功效表（δ=真实+注入的 N 效应 pos 差）\n",
        "| n_queries | δ | 功效 |" , "|---|---|---|",
    ]
    for r in rows_out:
        lines.append("| %d | %+.2f | %.3f |" % (
            r["n_queries"], r["effect_delta"], r["power"]))
    lines.append("\n## MDE（80% 功效）\n| n_queries | MDE |")
    lines.append("|---|---|")
    for m in mde:
        lines.append("| %d | %s |" % (m["n_queries"],
                                      ("%+.3f" % m["mde_0_80"])
                                      if m["mde_0_80"] is not None
                                      else ">%.2f(未达)" % (
                                          max(ds) if ds else 0)))
    lines.append("\n> 判读：δ=0 功效反映仅真实效应下的复现检出率；S28 应选择 "
                 "功效≥0.8 的 n_queries 与预期效应量匹配的组合。")
    (root / "report" / "s31_power.md").write_text(
        "\n".join(lines), encoding="utf-8")
    _log("已落盘 s31_power.json + report/s31_power.md")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
