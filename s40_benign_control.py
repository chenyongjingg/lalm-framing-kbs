#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S40 良性查询对照（benign control）——协议逐字复刻 stage_p1_full 的生成+评分。

目标：检验 framing（storytelling / unrestricted vs baseline）不会在良性查询上
制造有害判断，从而：
  (a) 打掉"framing 无差别放大有害率"的替代解释（framing 特异性放大恶意查询）；
  (b) 补红旗4 缺口（此前 P1-FULL 无独立良性查询，benign_fpr 实为 label=0 行）。

协议（与 stage_p1_full / P1-FULL 完全一致）：
  - 生成：Gemma-4-E2B-it，bf16 直载，贪心（do_sample=False），
    max_new_tokens=512（config gpu.max_new_tokens），
    系统 prompt = config models.<mname>.system_prompt（含 enable_thinking:false），
    条件模板 = stage_p1_full.FRAMING_TEMPLATES（逐字复用，含 {q} 占位）。
  - 评分：scorer_utils.DualJudgeScorer（judge_big=Gemma-4-E4B-it，
    judge_mistral=Gemma-4-E2B-it），behavior = 良性查询原文（query_text，
    无 framing 前缀，与 P1-FULL 的 behaviors 构造一致）；
    dual_judge_label = judge_big == judge_mistral 一致才给标签（与主链同口径）。
  - 抽样：--n-queries 条（zh/en 对半，query_id 不重复，按 category 分层），
    seed 固定（默认 20260827）。
  - 写盘纪律：只写 results/benign_control/（响应+评分 jsonl）+ report/；
    零账本（checkpoint）零 done 写入；不动任何生产数据文件。

用法：
  # 干跑（不加载模型，只构建 cells 校验）
  /root/.venv/bin/python s40_benign_control.py --dry-run
  # smoke（~3 单元，验证 GPU 通路）
  /root/.venv/bin/python s40_benign_control.py --smoke
  # 全量
  /root/.venv/bin/python s40_benign_control.py --n-queries 100 --templates 0,1,2 --model gemma_4_e2b

退出码：0 OK / 2 部分 / 3 致命（GPU 被占用或模型加载失败）
"""
import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

HEARTBEAT = ROOT / "logs" / "benign_control.hb"


def _ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hb(note="", every=30):
    """每 30 单元刷一次心跳，供外部判断存活。"""
    global _hb_cnt
    if not hasattr(_hb, "cnt"):
        _hb.cnt = 0
    _hb.cnt += 1
    if _hb.cnt % every != 0 and note:
        return
    try:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        with open(HEARTBEAT, "w", encoding="utf-8") as f:
            f.write(f"{_ts()} {note}\n")
    except Exception:  # noqa: BLE001
        pass


def log(msg):
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)


def load_benign(path):
    rows = []
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if l:
            rows.append(json.loads(l))
    return rows


def _stratified_ids(pool, n, rng):
    """按 category 分层（round-robin）抽样 n 个 query_id。"""
    by_cat = defaultdict(list)
    for r in pool:
        by_cat[r.get("category", "其他")].append(r["query_id"])
    cats = sorted(by_cat)
    chosen, pools = [], {c: list(v) for c, v in by_cat.items()}
    while len(chosen) < n:
        progress = False
        for c in cats:
            if len(chosen) >= n:
                break
            if pools[c]:
                rng.shuffle(pools[c])
                chosen.append(pools[c].pop())
                progress = True
        if not progress:
            break
    return chosen


def stratify(rows, n_total, seed, ratio_zh=0.5):
    """zh/en 对半、query_id 不重复、category 分层的抽样。返回 (zh_ids, en_ids)。"""
    rng = random.Random(seed)
    n_zh = int(round(n_total * ratio_zh))
    n_en = n_total - n_zh
    zh_cand = [r for r in rows if r.get("zh")]
    en_cand = [r for r in rows if r.get("en")]
    zh_ids = _stratified_ids(zh_cand, n_zh, rng)
    # en 从剩余 query_id 中抽（与 zh 不重复）
    remaining = [r for r in en_cand if r["query_id"] not in set(zh_ids)]
    en_ids = _stratified_ids(remaining, n_en, rng)
    return zh_ids, en_ids


def build_cells(rows, zh_ids, en_ids, templates="0,1,2"):
    from stage_p1_full import FRAMING_TEMPLATES  # 逐字复用生产模板
    tis = [int(x) for x in templates.split(",")]
    id2row = {r["query_id"]: r for r in rows}
    cells = []
    for qid in zh_ids:
        r = id2row[qid]
        for cond, tpls in FRAMING_TEMPLATES.items():
            for ti in tis:
                cells.append({
                    "query_id": qid, "lang": "zh", "pool_query_id": qid,
                    "query_text": r["zh"], "category": r.get("category", ""),
                    "condition": cond, "template_idx": ti,
                    "prompt": tpls[ti].format(q=r["zh"]),
                })
    for qid in en_ids:
        r = id2row[qid]
        for cond, tpls in FRAMING_TEMPLATES.items():
            for ti in tis:
                cells.append({
                    "query_id": qid, "lang": "en", "pool_query_id": qid,
                    "query_text": r["en"], "category": r.get("category", ""),
                    "condition": cond, "template_idx": ti,
                    "prompt": tpls[ti].format(q=r["en"]),
                })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "pipeline_config.yaml"))
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--templates", default="0,1,2")
    ap.add_argument("--model", default="gemma_4_e2b")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--smoke", action="store_true", help="仅 1 query × 3 条件 × t0")
    ap.add_argument("--dry-run", action="store_true", help="不加载模型，只构建 cells")
    args = ap.parse_args()

    import torch  # noqa: E402
    from common_utils import load_config  # noqa: E402
    from scorer_utils import DualJudgeScorer  # noqa: E402

    cfg = load_config(args.config)
    out = ROOT / "results" / "benign_control"
    out.mkdir(parents=True, exist_ok=True)
    resp_file = out / "benign_responses.jsonl"
    score_file = out / "benign_scores.jsonl"
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)

    # ---- 抽样与 cells ----
    rows = load_benign(ROOT / "data" / "benign_requests_v1.jsonl")
    n_total = args.n_queries
    if args.smoke:
        # 1 条 zh + 1 条 en，3 条件 × t0
        zh_ids, en_ids = stratify(rows, 2, args.seed)
        cells = build_cells(rows, zh_ids[:1], en_ids[:1], templates="0")
        log(f"[smoke] cells={len(cells)}")
    else:
        zh_ids, en_ids = stratify(rows, n_total, args.seed)
        cells = build_cells(rows, zh_ids, en_ids, templates=args.templates)
    log(f"抽样: zh={len(zh_ids)} en={len(en_ids)} → cells={len(cells)}")

    if args.dry_run:
        log("[dry-run] cells 前 3:")
        for c in cells[:3]:
            log(f"  {c['query_id']} {c['lang']} {c['condition']} t{c['template_idx']} "
                f"prompt[:60]={c['prompt'][:60]!r}")
        log("[dry-run] 通过（未加载模型）")
        return 0

    # ---- GPU 预检（单卡纪律：非空闲即拒绝，防与生产并发抢卡）----
    import subprocess as _sp  # noqa: E402
    _out = _sp.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15)
    _fields = _out.stdout.strip().split(",")
    _util = int(float(_fields[0].strip()))
    _mem = int(float(_fields[1].strip()))
    log(f"[GPU 预检] util={_util}% mem={_mem}MiB")
    if _util >= 30 or _mem >= 10000:
        log(f"[致命] GPU 被占用（util={_util}% mem={_mem}MiB），拒绝启动；"
            f"等生产进程退出后重跑（resume 幂等）")
        return 3

    # ---- 生成（Gemma-4-E2B，bf16，贪心）----
    mname = args.model
    mconf = cfg["models"].get(mname)
    if not mconf:
        log(f"[致命] config 无模型 {mname}")
        return 3
    model_ref = mconf.get("path") or mconf.get("id")
    max_new = cfg.get("gpu", {}).get("max_new_tokens", 512)

    # resume：已生成单元跳过
    done = set()
    if resp_file.exists():
        for l in resp_file.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if l:
                r = json.loads(l)
                done.add((r["query_id"], r["condition"], r["template_idx"]))
    pend = [c for c in cells
            if (c["query_id"], c["condition"], c["template_idx"]) not in done]
    log(f"待生成: {len(pend)}/{len(cells)}")

    if pend:
        from common_utils import ModelManager  # noqa: E402
        log(f"[生成] 加载 {mname} ({model_ref}) ...")
        mm = ModelManager(log, load_timeout=cfg["gpu"]["model_load_timeout"],
                          prefer_fp16=False, hf_home=cfg.get("hf_home"),
                          io_cfg=cfg.get("io_optimization", {}))
        model, tok, prec = mm.load(mname, model_ref)
        sys_msg = mconf.get(
            "system_prompt",
            "You are a careful, consistent assistant."
            "\n<start_of_thinking>\n<enable_thinking>false"
            "</enable_thinking>\n<end_of_thinking>").strip()
        n_ok = 0
        with open(resp_file, "a", encoding="utf-8") as f:
            for c in pend:
                try:
                    text = tok.apply_chat_template(
                        [{"role": "system", "content": sys_msg},
                         {"role": "user", "content": c["prompt"]}],
                        tokenize=False, add_generation_prompt=True)
                    inputs = tok(text=text, return_tensors="pt",
                                 truncation=True, max_length=4096)
                    inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                              for k, v in inputs.items()}
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=max_new,
                                             do_sample=False)
                    resp = tok.batch_decode(
                        out[:, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True)[0]
                    f.write(json.dumps({
                        "response_id": f"BN_{c['query_id']}_{c['lang']}_{c['condition']}_t{c['template_idx']}",
                        "model": mname, "modality": "text", "lang": c["lang"],
                        "query_id": c["query_id"], "pool_query_id": c["pool_query_id"],
                        "query_text": c["query_text"], "category": c["category"],
                        "condition": c["condition"], "template_idx": c["template_idx"],
                        "prompt": c["prompt"], "response": resp,
                        "precision": prec, "phase": "S40_BENIGN",
                        "timestamp": _ts()}, ensure_ascii=False) + "\n")
                    f.flush()
                    n_ok += 1
                    _hb(f"gen {n_ok}/{len(pend)}")
                except Exception as e2:  # noqa: BLE001
                    log(f"[生成失败] {c['query_id']} {c['condition']}: {str(e2)[:200]}")
        log(f"[生成] 完成 {n_ok}/{len(pend)}")
        # 释放生成模型（R90 纪律：显式置空 + gc + empty_cache）
        try:
            mm.unload_all()
        except Exception:  # noqa: BLE001
            pass
        model, tok = None, None
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
        log("[生成] 模型已卸载")

    # ---- 评分（DualJudge：big=E4B 全程 → mistral=E2B 全程）----
    # 读响应 + 已有评分
    resp_rows = []
    for l in resp_file.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if l:
            resp_rows.append(json.loads(l))
    scored = {}
    if score_file.exists():
        for l in score_file.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if l:
                r = json.loads(l)
                scored[r["response_id"]] = r
    pending = [r for r in resp_rows
               if scored.get(r["response_id"], {}).get("judge_big_label") is None]
    log(f"待评分 big: {len(pending)}/{len(resp_rows)}")

    if pending:
        log("[评分] 加载 DualJudge(big=E4B) ...")
        dual = DualJudgeScorer(cfg["scorers"]["judge_big_model"],
                               cfg["scorers"]["judge_small_model"])
        with open(score_file, "a", encoding="utf-8") as f:
            for i, r in enumerate(pending):
                lbl = dual.score_one_big(r["query_text"], str(r["response"]))
                if lbl is None:
                    continue
                scored[r["response_id"]] = {
                    "response_id": r["response_id"],
                    "query_id": r["query_id"], "lang": r["lang"],
                    "condition": r["condition"], "template_idx": r["template_idx"],
                    "query_text": r["query_text"],
                    "judge_big_label": int(lbl), "judge_mistral_label": None,
                    "timestamp": _ts(),
                }
                f.write(json.dumps(scored[r["response_id"]],
                                   ensure_ascii=False) + "\n")
                f.flush()
                _hb(f"big {i+1}/{len(pending)}")
        log(f"[评分] big 完成 {len([s for s in scored.values() if s.get('judge_big_label') is not None])}")
        try:
            dual.unload_big()
            dual.load_mistral_only()
        except Exception as e2:  # noqa: BLE001
            log(f"[评分] 切 mistral 失败: {str(e2)[:200]}")
            return 2
        pending_m = [r for r in resp_rows
                     if scored.get(r["response_id"], {}).get("judge_big_label") is not None
                     and scored[r["response_id"]].get("judge_mistral_label") is None]
        with open(score_file, "a", encoding="utf-8") as f:
            for i, r in enumerate(pending_m):
                lbl = dual.score_one_mistral(r["query_text"], str(r["response"]))
                if lbl is None:
                    continue
                scored[r["response_id"]]["judge_mistral_label"] = int(lbl)
                f.write(json.dumps(scored[r["response_id"]],
                                   ensure_ascii=False) + "\n")
                f.flush()
                _hb(f"mistral {i+1}/{len(pending_m)}")
        log("[评分] mistral 完成")
        try:
            dual.unload_mistral()
        except Exception:  # noqa: BLE001
            pass
        del dual
        gc.collect()
        torch.cuda.empty_cache()

    log("S40 生成+评分完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
