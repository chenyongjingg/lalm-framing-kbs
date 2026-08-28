#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S39（第二部分）：S33（Qwen2-Audio 温度鲁棒性 scope）全家族效应计算。

S39 已补齐 S33 的 strongreject + harmbench 官方基准腿（harmbench 用
--hb-batch 2 在 GPU1 上完成，因主链占用 GPU1 4.86GB 需降批）。本脚本用
纯 CPU 计算 S33 全部 5 家评分族（judge_big/judge_small/strongreject/
harmbench/shieldgemma）的 N/E_t 效应，镜像 s28_five_family_effects.json
结构，使附录 B.4 的 S28/S33 两行对称。

产出：
  results/gpu1_pipeline/s33_five_family_effects.json
纪律：零 GPU、零写生产缓存；只读响应与缓存。

用法：/root/.venv/bin/python gpu1_s39_s33_effects.py
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu1_s28_hetero_audio as s28  # noqa: E402 _bootstrap_pair


def _load_rid(p):
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            d = json.loads(line)
            r = d["rid"]
            out[r[4:] if r.startswith("s28_") or r.startswith("s33_") else r] = \
                d.get("label")
    return out


def _family_effects(rows, labs, B, seed, families, out):
    by_rid = {r["rid"]: r for r in rows}
    for fam in families:
        lab = labs[fam]
        out[fam] = {}
        for key, (sela, selb) in {
            "N_Et0": (lambda r: r["E_t"] == 0 and r["N"] == 0,
                      lambda r: r["E_t"] == 0 and r["N"] == 1),
            "Et_N0": (lambda r: r["E_t"] == 0 and r["N"] == 0,
                      lambda r: r["E_t"] == 1 and r["N"] == 0),
            "Et_N1": (lambda r: r["E_t"] == 0 and r["N"] == 1,
                      lambda r: r["E_t"] == 1 and r["N"] == 1),
        }.items():
            a = [(by_rid[r]["query_id"], lab.get(r)) for r in by_rid
                 if sela(by_rid[r]) and lab.get(r) is not None]
            b = [(by_rid[r]["query_id"], lab.get(r)) for r in by_rid
                 if selb(by_rid[r]) and lab.get(r) is not None]
            eff = s28._bootstrap_pair(a, b, B, seed + hash(fam) % 1000)
            out[fam][key] = eff
    return out


def main():
    cfg = yaml.safe_load(open("pipeline_config.yaml", encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    cache_dir = out_dir / "scorers_cache"
    B = cfg.get("seeds", {}).get("bootstrap", 2000)
    seed = cfg.get("seeds", {}).get("bootstrap", 42)

    rows = [json.loads(l) for l in
            (out_dir / "responses" / "s33_hetero_audio.jsonl").open(
                encoding="utf-8")]
    # 统一 rid 为缓存键（缓存 rid 带 s33_ 前缀）
    for r in rows:
        r["rid"] = r["response_id"]
    n_query = len(set(r["query_id"] for r in rows))
    print("[s39fx] S33=%d 格, %d 查询" % (len(rows), n_query), flush=True)

    sg = {}
    sg_all = [json.loads(l) for l in
              (out_dir / "s38_shieldgemma_labels.jsonl").open(encoding="utf-8")]
    for rec in sg_all:
        if rec.get("scope") == "s33_hetero_audio":
            sg[rec["rid"][4:] if rec["rid"].startswith("s33_") else rec["rid"]] = \
                rec.get("label")

    families = ["judge_big", "judge_small", "strongreject", "harmbench",
                "shieldgemma"]
    labs = {}
    for fam in families:
        if fam == "shieldgemma":
            labs[fam] = sg
        else:
            labs[fam] = _load_rid(cache_dir / ("s33_hetero_audio_%s.jsonl" % fam))
        nn = sum(1 for v in labs[fam].values() if v is not None)
        print("[s39fx] %s 非空 %d/%d" % (fam, nn, len(rows)), flush=True)

    effects = _family_effects(rows, labs, B, seed, families, {})
    out = {"stage": "S39", "date": "2026-08-16",
           "scope": "s33_hetero_audio",
           "generator": "Qwen2-Audio-7B",
           "n_cells": len(rows), "n_query": n_query,
           "effects": effects}
    (out_dir / "s33_five_family_effects.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 终端摘要：N(E_t=0) 每家
    print("\n=== S33 N effect at E_t=0（N1-N0，正=N1 更有害）===")
    for fam in families:
        e = effects[fam]["N_Et0"]
        print("%-12s %+.4f [%s,%s] %s" % (
            fam, e["effect"], e["ci95"][0], e["ci95"][1],
            "✓" if e["excl_zero"] else "✗"))
    print("完成 → s33_five_family_effects.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
