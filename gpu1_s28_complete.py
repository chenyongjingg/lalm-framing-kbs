#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S28 补齐：重评 judge_small 空标签 + 4 评分器合并一致率分析。

背景（2026-08-15）：post21 编排链已自动补完 S28 的 strongreject（14:17）与
harmbench（14:54）腿，judge_big/judge_small 早于 08-14 完成。当前唯一显存缺口：
judge_small 缓存 12 条 label=None（评分失败未重试）。本脚本：
  1. 用 GemmaJudgeScorer（s9 工厂）重评 judge_small 空标签，追加写入缓存
     （幂等：已有 label 的 rid 跳过，与 s28_sr_leg 同款续传模式）；
  2. 读取全部 4 评分器缓存（judge_big/judge_small/strongreject/harmbench），
     归一化 rid（judge 缓存带 "s28_" 前缀），计算 pairwise Cohen's κ +
     Spearman ρ，检验"独立评分器在 S28 上方向是否一致"（KBS 审查核心追问）；
  3. 用权威 dual_judge（big==small）为真值，报告各评分器一致率 + 争议率；
  4. 重算 S28 跨生成器 N/E_t 效应（dual_judge / strongreject / harmbench
     三口径），检验独立评分器下效应是否同向复现。

纪律：
  - 零人工标注、零账本、不写 .complete/.done；CUDA_VISIBLE_DEVICES=1 由调用方注入。
  - 只读 responses/s28_hetero_audio.jsonl + scorers_cache/s28_*；只写
    scorers_cache/s28_hetero_audio_judge_small.jsonl（追加）+
    results/gpu1_pipeline/s28_hetero_audio_full.json + report/s28_hetero_audio_full.md。
  - GPU1 串行纪律：启动前检测 GPU1 占用（排除自身），被占则等待/退出。

用法：CUDA_VISIBLE_DEVICES=1 python gpu1_s28_complete.py [--config pipeline_config.yaml]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu1_s9_cross_family as s9  # noqa: E402 复用 SCORER_FACTORIES/_bootstrap_pair
import gpu1_s28_hetero_audio as s28  # noqa: E402 复用 _effect_q/_effect_g/_is_stall


def _gpu1_busy(min_free_mib=8192):
    """GPU1（index 1）是否可跑：按剩余显存判断，避免把 GPU0 进程误判为占用。

    主链 stage_p1_pilot（505447）是多 GPU 程序，在 GPU1 上有 ~4.8GB 常驻
    分配且不增长——它不阻塞 S 系列小评分器（judge_small 4bit ~3GB）。
    真正的冲突是 GPU1 剩余显存不足（>8GB free 才视为可跑）。
    """
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines()
    for ln in out:
        if not ln.strip():
            continue
        idx, free = [x.strip() for x in ln.split(",")]
        if idx == "1":
            return int(free) < min_free_mib
    return True


def _own_pids():
    me = {str(__import__("os").getpid())}
    try:
        import os
        import subprocess
        ppid = os.getppid()
        me.add(str(ppid))
        out = subprocess.run(["pgrep", "-P", str(os.getpid())],
                             capture_output=True, text=True).stdout.split()
        me.update(out)
    except Exception:  # noqa: BLE001
        pass
    return me


def _load_cache(p):
    out = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                r = d["rid"]
                out[r[4:] if r.startswith("s28_") else r] = d.get("label")
    return out


def _norm_rid(r):
    return r[4:] if r.startswith("s28_") else r


def _pairwise_agreement(lab1, lab2):
    """two binary label dicts → Cohen κ + Spearman ρ + 一致率。"""
    ids = [r for r in lab1 if lab1[r] is not None and lab2.get(r) is not None]
    if len(ids) < 10:
        return None
    a = np.array([float(lab1[r]) for r in ids])
    b = np.array([float(lab2[r]) for r in ids])
    agree = float((a == b).mean())
    # Cohen's kappa
    n = len(a)
    p0 = float((a == b).mean())
    p1 = float(a.mean()) * float((1 - b).mean()) + (1 - float(a.mean())) * float(b.mean())
    kappa = (p0 - p1) / (1 - p1) if (1 - p1) > 1e-9 else 1.0
    # Spearman (rank correlation)
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    da = ar - ar.mean()
    db = br - br.mean()
    rho = float((da * db).sum() / np.sqrt((da ** 2).sum() * (db ** 2).sum()))
    return {"n": n, "agreement": round(agree, 4), "kappa": round(kappa, 4),
            "spearman": round(rho, 4)}


def _effect_by_label(rows, label_fn, E0, N0, E1, N1, B, seed):
    """rows: 响应记录；label_fn(rid)->label 或 None。query 聚类 bootstrap。"""
    a = [(r["query_id"], label_fn(r["response_id"]))
         for r in rows if r["E_t"] == E0 and r["N"] == N0
         and label_fn(r["response_id"]) is not None]
    b = [(r["query_id"], label_fn(r["response_id"]))
         for r in rows if r["E_t"] == E1 and r["N"] == N1
         and label_fn(r["response_id"]) is not None]
    return s28._bootstrap_pair(a, b, B, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root, get_logger
    root = resolve_root(cfg)
    log = get_logger("s28_complete", root)
    out_dir = root / "results" / "gpu1_pipeline"
    cache_dir = out_dir / "scorers_cache"
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    def _log(m):
        print("[s28c] %s" % m, flush=True)

    # ---- 0. GPU1 显存检测（主链 505447 在 GPU1 有 4.8GB 常驻，不阻塞小评分器）----
    if _gpu1_busy():
        _log("GPU1 剩余显存不足，退出（串行纪律）")
        return 1
    _log("GPU1 空闲，开始")

    # ---- 1. 读 S28 响应 ----
    resp_path = out_dir / "responses" / "s28_hetero_audio.jsonl"
    rows = [json.loads(l) for l in resp_path.open(encoding="utf-8")]
    _log("S28 响应=%d 条" % len(rows))
    rids = [r["response_id"] for r in rows]

    # ---- 2. 重评 judge_small 空标签（显存缺口）----
    js_path = cache_dir / "s28_hetero_audio_judge_small.jsonl"
    js_cache = _load_cache(js_path)
    todo = [r for r in rows
            if r["response_id"] in js_cache and js_cache[r["response_id"]] is None]
    _log("judge_small 空标签 %d 条待重评" % len(todo))
    if todo:
        s9.register_scorers(cfg)
        _log("加载 judge_small 评分器 ...")
        sc = s9.SCORER_FACTORIES["judge_small"]()
        for start in range(0, len(todo), 4):
            chunk = todo[start:start + 4]
            pairs = [(r.get("prompt") or "", str(r.get("response") or ""))
                     for r in chunk]
            try:
                res = sc.score_batch(pairs, batch_size=len(chunk))
            except Exception as e:  # noqa: BLE001
                _log("batch fail: %s" % str(e)[:150])
                res = [{"label": None} for _ in chunk]
            with js_path.open("a", encoding="utf-8") as f:
                for r, x in zip(chunk, res):
                    f.write(json.dumps(
                        {"rid": r["response_id"], "label": x.get("label")},
                        ensure_ascii=False) + "\n")
            _log("judge_small 重评进度 %d/%d" % (min(start + 4, len(todo)), len(todo)))
        sc.close()
        _log("judge_small 重评完成，刷新缓存")
        js_cache = _load_cache(js_path)
    else:
        _log("judge_small 无空标签，跳过")

    # ---- 3. 加载全部 4 评分器缓存 ----
    jb = _load_cache(cache_dir / "s28_hetero_audio_judge_big.jsonl")
    js = js_cache
    sr = _load_cache(cache_dir / "s28_hetero_audio_strongreject.jsonl")
    hb = _load_cache(cache_dir / "s28_hetero_audio_harmbench.jsonl")
    _log("judge_big=%d judge_small=%d strongreject=%d harmbench=%d"
         % (len([v for v in jb.values() if v is not None]),
            len([v for v in js.values() if v is not None]),
            len([v for v in sr.values() if v is not None]),
            len([v for v in hb.values() if v is not None])))

    def dual_lab(rid):
        b, s = jb.get(rid), js.get(rid)
        if b is not None and s is not None and b == s:
            return float(b)
        return None

    # ---- 4. 权威 dual_judge 真值 + 各评分器一致率 ----
    auth = {}
    for r in rows:
        dl = dual_lab(r["response_id"])
        if dl is not None:
            auth[r["response_id"]] = dl
    _log("权威 dual_judge 共识=%d（judge_big==judge_small）" % len(auth))
    dispute_n = sum(1 for r in rows
                    if jb.get(r["response_id"]) is not None
                    and js.get(r["response_id"]) is not None
                    and jb[r["response_id"]] != js[r["response_id"]])
    _log("双 judge 争议格=%d" % dispute_n)

    agree = {}
    for name, lab in [("judge_big", jb), ("judge_small", js),
                      ("strongreject", sr), ("harmbench", hb)]:
        agree[f"auth_vs_{name}"] = _pairwise_agreement(auth, lab)
    agree["auth_vs_dual_self"] = {
        "n": len(auth), "agreement": 1.0, "kappa": 1.0, "spearman": 1.0}
    # 4 评分器两两一致率
    scs = {"judge_big": jb, "judge_small": js,
           "strongreject": sr, "harmbench": hb}
    keys = list(scs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ag = _pairwise_agreement(scs[keys[i]], scs[keys[j]])
            if ag:
                agree["%s_vs_%s" % (keys[i], keys[j])] = ag

    # ---- 5. 三口径跨生成器效应复现 ----
    B = cfg.get("seeds", {}).get("bootstrap", 2000)
    seed = cfg.get("seeds", {}).get("bootstrap", 42)
    colmap = {"dual_judge": dual_lab, "strongreject": lambda r: sr.get(r),
              "harmbench": lambda r: hb.get(r)}
    effects = {}
    for scope, fn in colmap.items():
        n_qwen = _effect_by_label(rows, fn, 0, 0, 0, 1, B, seed + 1)
        e_qwen = _effect_by_label(rows, fn, 0, 0, 1, 0, B, seed + 3)
        effects[scope] = {"N_effect_qwen2": n_qwen, "Et_effect_qwen2": e_qwen}
    # 与已产出的 strongreject 分析对照
    sr_existing = None
    sp = out_dir / "s28_hetero_audio_strongreject.json"
    if sp.exists():
        sr_existing = json.loads(sp.read_text(encoding="utf-8"))

    # ---- 6. 落盘 ----
    out = {
        "stage": "S28-COMPLETE", "date": "2026-08-15",
        "purpose": ("S28 补齐：重评 judge_small 空标签 + 4 独立评分器"
                    "（judge_big/judge_small/strongreject/harmbench）合并一致率"
                    " + 跨生成器效应三口径复现"),
        "n_cells": len(rows), "n_dual_consensus": len(auth),
        "n_dispute": dispute_n,
        "scorer_agreement": agree,
        "effects": {k: {kk: ({"effect": vv["effect"], "ci95": vv["ci95"],
                              "excl_zero": vv["excl_zero"],
                              "n_query": vv["n_query"]} if vv else None)
                        for kk, vv in v.items()} for k, v in effects.items()},
        "strongreject_existing": sr_existing,
        "disclosure": ("judge 缓存 rid 带 s28_ 前缀已归一化；judge_small 12 空经"
                       "本脚本重评（幂等续传）。权威口径=judge_big==judge_small 共识，"
                       "争议格排除。效应为 Qwen2-Audio-7B 生成器（跨生成器第 3 复现）。"
                       "CI=query 聚类 bootstrap。"),
    }
    (out_dir / "s28_hetero_audio_full.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 7. 人类可读报告 ----
    def _fmt(eff):
        if not eff:
            return "N/A"
        return "%.4f [%s,%s] %s" % (eff["effect"], eff["ci95"][0],
                                    eff["ci95"][1],
                                    "✓" if eff["excl_zero"] else "✗")

    lines = ["# S28 补齐：4 评分器合并一致率 + 三口径效应复现\n",
             "- 日期：2026-08-15\n",
             "- 单元：%d，权威 dual_judge 共识 %d，双 judge 争议 %d\n" % (
                 len(rows), len(auth), dispute_n),
             "\n## 各评分器 vs 权威 dual_judge 一致率\n",
             "| 评分器 | n | 一致率 | κ | Spearman |"]
    for k, v in agree.items():
        if v:
            lines.append("| %s | %d | %.3f | %.3f | %.3f |" % (
                k, v["n"], v["agreement"], v["kappa"], v["spearman"]))
    lines += ["\n## 跨生成器效应（Qwen2-Audio vs 权威/独立评分器）\n",
              "| 口径 | N_effect_qwen2 | Et_effect_qwen2 |"]
    for scope, eff in effects.items():
        lines.append("| %s | %s | %s |" % (
            scope, _fmt(eff["N_effect_qwen2"]), _fmt(eff["Et_effect_qwen2"])))
    lines += ["\n- 详情：`results/gpu1_pipeline/s28_hetero_audio_full.json`\n"]
    (report_dir / "s28_hetero_audio_full.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    _log("完成 → s28_hetero_audio_full.json + report/s28_hetero_audio_full.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
