#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S29：音频生成确定性大样本复核（GPU1，2026-08-14）。

动机：S2 已对文本路径做了 40/40 字节级确定性（同设备两次 + 跨设备 vs GPU0）。
审稿人可能问："音频路径（wav 输入 + 多模态编码）确定性是否同样成立？" 本实验
扩展至音频：从 E4B 音频响应中确定性抽取 N=120 格（跨 combo×template），在
GPU1 上以与权威路径相同的代码（stage_p0c._lalm_audio_one gemma_4 分支）每格
生成两次，检验：
  - 同设备（GPU1 run1 vs run2）字节一致率；
  - 跨设备（GPU1 run1 vs GPU0 存储响应）字节一致率（同权重 bf16 + greedy）。
  - 逐字节 diff 定位（首处差异位置/长度），供如实披露"确定性程度"。

纪律：
  - 只读 E4B 音频行（抽样）；独立写 results/gpu1_pipeline/s29_*.json / md。
  - 零人工标注、零账本、不写 .complete/.done。CUDA_VISIBLE_DEVICES=1 由调用方注入。
  - 与权威路径同代码 → 可比；但产物独立（不写回 E4B responses）。

用法：CUDA_VISIBLE_DEVICES=1 python gpu1_s29_determinism_audio.py [--n 120] [--seed 20260814]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL = "gemma_4_e4b"


def _log(m):
    print("[s29] %s" % m, flush=True)


def _diff_pos(a, b):
    """返回首处差异位置；相等返回 None。"""
    if a == b:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n = 2

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root, get_logger
    root = resolve_root(cfg)
    log = get_logger("s29", root)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_new = cfg.get("gpu", {}).get("max_new_tokens", 512)

    # 抽样：E4B 音频行（含存储响应），跨 combo×template 确定性均衡
    rows = []
    for line in open(root / "responses" / "P1_PILOT" /
                     "gemma_4_e4b_responses.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("modality") == "audio" and r.get("response"):
            rows.append(r)
    rows.sort(key=lambda r: (tuple(r["combo"]), r["template_idx"],
                             r["query_id"]))
    # 分 (combo, template) 桶均衡抽取
    buckets = {}
    for r in rows:
        buckets.setdefault((tuple(r["combo"]), r["template_idx"]), []).append(r)
    rng = np.random.RandomState(args.seed)
    sel = []
    while len(sel) < args.n and buckets:
        for k in list(buckets.keys()):
            pool = buckets[k]
            if not pool:
                del buckets[k]
                continue
            idx = rng.randint(len(pool))
            sel.append(pool.pop(idx))
            if len(sel) >= args.n:
                break
    _log("抽样 %d 格（%d 桶）" % (len(sel), len(buckets)))

    from gpu1_common import load_generation_model, release
    mconf = cfg["models"][MODEL]
    _log("加载 %s ..." % MODEL)
    model, tok = load_generation_model(MODEL, mconf, cfg, log)

    # 复用权威路径 _lalm_audio_one（import 权威函数，保证与 GPU0 同代码可比）
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage_p0c_mod", "stage_p0c.py")
    sp0c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp0c)
    _lalm_audio_one = sp0c._lalm_audio_one

    recs = []
    n_same_dev_eq = n_cross_eq = n_err = 0
    n_same_dev = n_cross = 0
    for r in sel:
        wav, prompt = r["audio_path"], r["prompt"]
        run1 = run2 = None
        try:
            run1 = _lalm_audio_one(MODEL, model, tok, wav, prompt, max_new)
            run2 = _lalm_audio_one(MODEL, model, tok, wav, prompt, max_new)
        except Exception as e:  # noqa: BLE001
            log.warning("格 %s 生成失败: %s", r["response_id"], str(e)[:150])
            n_err += 1
        stored = r.get("response")
        if run1 is None:
            recs.append({"response_id": r["response_id"],
                         "combo": r["combo"], "template_idx": r["template_idx"],
                         "error": True})
            continue
        if run2 is not None:
            n_same_dev += 1
            eq_same = run1 == run2
            n_same_dev_eq += int(eq_same)
        else:
            eq_same = None
        if stored:
            n_cross += 1
            eq_cross = run1 == stored
            n_cross_eq += int(eq_cross)
        else:
            eq_cross = None
        recs.append({
            "response_id": r["response_id"],
            "combo": r["combo"], "template_idx": r["template_idx"],
            "query_id": r["query_id"],
            "same_dev_eq": bool(eq_same) if eq_same is not None else None,
            "cross_dev_eq": bool(eq_cross) if eq_cross is not None else None,
            "diff_pos_same": _diff_pos(run1, run2 or ""),
            "diff_pos_cross": _diff_pos(run1, stored or ""),
            "len_run1": len(run1), "len_stored": len(stored or "")})
        _log("%s 同设备=%s 跨设备=%s" % (
            r["response_id"],
            "Y" if eq_same else ("N" if eq_same is False else "-"),
            "Y" if eq_cross else ("N" if eq_cross is False else "-")))
    release(model, tok)

    same_rate = n_same_dev_eq / n_same_dev if n_same_dev else None
    cross_rate = n_cross_eq / n_cross if n_cross else None
    _log("同设备字节一致=%s（%d/%d） 跨设备字节一致=%s（%d/%d）" % (
        same_rate, n_same_dev_eq, n_same_dev, cross_rate, n_cross_eq, n_cross))

    out = {
        "stage": "S29", "date": "2026-08-14",
        "purpose": ("音频生成确定性大样本复核（stage_p0c._lalm_audio_one "
                    "gemma_4 分支，GPU1 重跑两次 + 跨设备 vs GPU0 存储）"),
        "model": MODEL, "n_cells": len(sel), "n_error": n_err,
        "same_device_byte_eq": round(same_rate, 4) if same_rate is not None
        else None,
        "same_device_n": n_same_dev, "same_device_eq_n": n_same_dev_eq,
        "cross_device_byte_eq": round(cross_rate, 4)
        if cross_rate is not None else None,
        "cross_device_n": n_cross, "cross_device_eq_n": n_cross_eq,
        "per_cell": recs,
        "disclosure": ("同一 GPU 上 bf16 + greedy 解码理论应字节一致；跨 GPU "
                       "可能存在浮点归约的极小非确定性，逐字节 diff 如实报告。"
                       "若字节一致率 < 1，披露差异位置分布以证'实践不可察觉'。"),
    }
    (out_dir / "s29_determinism_audio.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# S29：音频生成确定性大样本复核（GPU1）\n",
        "- 模型：%s；抽样 %d 格（跨 combo×template 桶均衡）" % (MODEL, len(sel)),
        "- 同设备（GPU1 run1 vs run2）字节一致率：**%s**（%d/%d）" % (
            ("%.4f" % same_rate) if same_rate is not None else "N/A",
            n_same_dev_eq, n_same_dev),
        "- 跨设备（GPU1 vs GPU0 存储）字节一致率：**%s**（%d/%d）\n" % (
            ("%.4f" % cross_rate) if cross_rate is not None else "N/A",
            n_cross_eq, n_cross),
        "## 逐格结果",
        "| response_id | combo | t | 同设备 | 跨设备 | diff_pos_同 | diff_pos_跨 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        lines.append("| %s | %s | %d | %s | %s | %s | %s |" % (
            r["response_id"], r["combo"], r["template_idx"],
            ("✓" if r.get("same_dev_eq") else
             ("✗" if r.get("same_dev_eq") is False else "-")),
            ("✓" if r.get("cross_dev_eq") else
             ("✗" if r.get("cross_dev_eq") is False else "-")),
            r.get("diff_pos_same"), r.get("diff_pos_cross")))
    lines.append("\n## 披露\n> %s" % out["disclosure"])
    (root / "report" / "s29_determinism_audio.md").write_text(
        "\n".join(lines), encoding="utf-8")
    _log("已落盘 s29_determinism_audio.json + report/s29_determinism_audio.md")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
