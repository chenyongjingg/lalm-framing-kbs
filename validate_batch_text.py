# -*- coding: utf-8 -*-
"""
validate_batch_text.py — 选项 B 验证协议（启用门槛，v6.5 宪法下）

在空闲 GPU（调用时 CUDA_VISIBLE_DEVICES=1）上，用**真实单元**（复用
stage_p1_full 数据构造：queries_v2 + advbench + FRAMING_TEMPLATES +
config system_prompt，与生产同 seed 抽样）做 逐条 vs 批量 A/B：

  - 逐条 = stage_p1_full.py L380-416 / stage_p0c.py L216-251 Gemma-4 文本
          路径的**精确复刻**（apply_chat_template system+user →
          proc(text=...) → greedy generate → proc.batch_decode[0]）
  - 批量 = gemma4_batch_text.infer_text_batch（逐单元同 tokenize + 左 padding）

判定（AUDIT #165 定义）：**100% 字符串逐字节一致 → PASS（批量启用前提）**；
任一不一致 / 异常 → FAIL（弃 B，逐条保留，无损失）。

用法：
  CUDA_VISIBLE_DEVICES=1 python validate_batch_text.py \
      --config pipeline_config.yaml --n-e4b 200 --n-e2b 100 --bs 8 --seed 42
  python validate_batch_text.py --config pipeline_config.yaml --dry-run
  # --dry-run 仅构建单元清单并校验采样，不加载模型/不推理（冒烟用）
  # 产物写入 audit/validate_batch/（数据隔离 AUDIT #172，不与 results/ 混居）
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from pathlib import Path

from stage_p1_full import (FRAMING_TEMPLATES, build_cells,
                           load_queries_pool, load_queries_v1, sample_queries)

DEFAULT_SYS = ("You are a careful, consistent assistant.\n"
               "<start_of_thinking>\n<enable_thinking>false"
               "</enable_thinking>\n<end_of_thinking>")

MODEL_ORDER = ["gemma_4_e4b", "gemma_4_e2b"]


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_advbench_robust(csv_path: Path, n: int, seed: int) -> list:
    """AdvBench 有害行为 CSV 健壮加载（兼容 goal/Goal/behavior 列名）。

    背景（audit_log.md AUDIT #166）：stage_p1_full.load_advbench 只读
    behavior/Goal，而阶段 D 实际写入的 advbench_sample_v1.csv 列为 goal,target
    （源 advbench_harmful_behaviors.csv 原始列名）→ P1-FULL 的 AdvBench 锚定集
    在运行时将静默为空（§6.2 违反风险）。本加载器为**验证协议专用**，覆盖
    预期完整输入分布（含英文 AdvBench），不修改任何阶段脚本。
    """
    import csv

    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = (r.get("behavior") or r.get("Goal") or r.get("goal") or "").strip()
            if b:
                rows.append(b)
    rng = random.Random(seed)
    idx = rng.sample(range(len(rows)), min(n, len(rows)))
    return [rows[i] for i in idx]


def build_real_cells(cfg, root: Path):
    """与 stage_p1_full.main 相同口径构造真实设计单元（PILOT 零重叠排除）。

    返回 (all_cells, meta) ；不推理，纯数据层。
    """
    from common_utils import JsonlLogger
    elog = JsonlLogger(str(root / "logs" / "errors.jsonl"))
    seeds = cfg.get("seeds", {})
    p1f = cfg.get("p1_full", {})
    zh_pool, en_pool = load_queries_pool(root, "queries_v2.jsonl")
    if not zh_pool:
        zh_pool = load_queries_v1(root / "data" / "queries_v1.jsonl", "zh")
    if not en_pool:
        en_pool = load_queries_v1(root / "data" / "queries_v1.jsonl", "en")
    adv_q = []
    adv_csv = root / "data" / "advbench_sample_v1.csv"
    if adv_csv.exists():
        adv_q = load_advbench_robust(adv_csv, p1f.get("advbench_n", 200),
                                     seeds.get("sampling_advbench", 20260806))
    n_zh = p1f.get("queries_n_zh", 200)
    n_en = p1f.get("queries_n_en", 200)
    pilot_used = set()
    pilot_f = root / "results" / "p1_pilot_queries_zh.json"
    if pilot_f.exists():
        try:
            pilot_used = set(json.loads(
                pilot_f.read_text(encoding="utf-8")).get("queries", []))
        except Exception:  # noqa: BLE001
            pilot_used = set()
    zh_pool = [q for q in zh_pool if q not in pilot_used]
    en_pool = [q for q in en_pool if q not in pilot_used]
    zh_s = sample_queries(zh_pool, n_zh, seeds.get("sampling_full_zh", 20260804))
    en_s = sample_queries(en_pool, n_en, seeds.get("sampling_full_en", 20260805))
    if not en_s and zh_s:
        elog.event(stage="validate_batch", event="lang_fallback",
                   note="英文池为空，en 槽位降级用中文（仅验证，报告侧披露）")
        en_s = sample_queries(zh_pool, n_en, seeds.get("sampling_full_en", 20260805))
    cells = (build_cells(zh_s, "zh") + build_cells(en_s, "en")
             + build_cells(adv_q, "adv"))
    meta = {"zh_queries": len(zh_s), "en_queries": len(en_s),
            "adv_queries": len(adv_q), "total_cells": len(cells)}
    return cells, meta


def stratified_sample(cells, n_total, seed):
    """按 (lang × condition) 分层均衡抽样。每组 ceil(n_total/9) 个。

    返回 list[cell]，确定性（seed）。
    """
    rng = random.Random(seed)
    groups = [(lang, cond) for lang in ("zh", "en", "adv")
              for cond in FRAMING_TEMPLATES.keys()]
    per = (n_total + len(groups) - 1) // len(groups)
    sel = []
    for g in groups:
        pool = [c for c in cells
                if c["lang"] == g[0] and c["condition"] == g[1]]
        pool = sorted(pool, key=lambda c: (c["query_id"], c["template_idx"]))
        rng.shuffle(pool)
        sel.extend(pool[:per])
    return sel[:n_total]


def infer_single_prod(model, tok, text, max_new):
    """stage_p1_full/stage_p0c Gemma-4 逐条路径精确复刻。"""
    import torch as _t
    inputs = tok(text=text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v
              for k, v in inputs.items()}
    with _t.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    resp = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)[0]
    return resp, inputs["input_ids"].shape[1]


def run_model(model_name, mconf, cfg, root, cells, n, bs, seed, out_dir):
    import torch as _t
    from common_utils import ModelManager
    from gemma4_batch_text import infer_text_batch

    from common_utils import setup_logging
    log = setup_logging(str(root / "logs" / "validate_batch.log"),
                        f"validate_batch_{model_name}")
    mm = ModelManager(log, load_timeout=cfg["gpu"]["model_load_timeout"],
                      prefer_fp16=False, hf_home=cfg.get("hf_home"),
                      io_cfg=cfg.get("io_optimization", {}))
    model_ref = mconf.get("path") or mconf.get("id")
    sys_msg = (mconf.get("system_prompt", DEFAULT_SYS)).strip()
    max_new = cfg.get("gpu", {}).get("max_new_tokens", 512)

    model, tok, prec = mm.load(model_name, model_ref)
    # R102 修复（2026-08-12）：加载预检——拒绝「多模态模型被当 CausalLM 加载」的
    # 静默错误路径（架构检测 BF16 直载 OOM → 回退 CausalLM fp16/4bit → 首个
    # generate 崩溃/挂死；实例：验证进程 278319 静默死亡，日志止于「开始验证」）。
    # 权重未驻留 CUDA 或类名非条件生成模型 → 显式报错退出，不静默带病生成。
    _cls = model.__class__.__name__.lower()
    _dev = getattr(getattr(model, "device", None), "type", None)
    if _dev != "cuda" or ("condition" not in _cls and "causallm" in _cls):
        _t.cuda.empty_cache()
        gc.collect()
        raise RuntimeError(
            f"[{model_name}] 加载预检失败 class={model.__class__.__name__} "
            f"device={_dev!r}——多模态模型被当 CausalLM 加载的静默错误路径，"
            "拒绝进入生成。请以 CUDA_VISIBLE_DEVICES=<空闲GPU> 启动使 BF16 "
            "直载命中（不得依赖回退路径）")
    log.info("[%s] 加载预检通过 class=%s device=%s prec=%s",
             model_name, model.__class__.__name__, _dev, prec)
    log.info("[%s] 加载完成 prec=%s，开始验证 %d 单元",
             model_name, prec, n)

    sample = stratified_sample(cells, n, seed)
    texts = []
    for c in sample:
        texts.append(tok.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": c["prompt"]}],
            tokenize=False, add_generation_prompt=True))

    # 逐条基准
    single_resp, single_prompt_len, single_err = [], [], []
    for i, text in enumerate(texts):
        try:
            r, pl = infer_single_prod(model, tok, text, max_new)
            single_resp.append(r)
            single_prompt_len.append(pl)
            single_err.append(None)
        except Exception as e:  # noqa: BLE001
            single_resp.append(None)
            single_prompt_len.append(None)
            single_err.append(f"single:{str(e)[:200]}")
        if (i + 1) % 50 == 0:
            log.info("[%s] single %d/%d", model_name, i + 1, n)

    # 批量（按 bs 分块，完整覆盖 chunk 边界）
    batch_resp, batch_err = [], []
    for s in range(0, len(texts), bs):
        chunk = texts[s:s + bs]
        try:
            r = infer_text_batch(model, tok, chunk, max_new, bs=bs)
            batch_resp.extend(r)
            batch_err.extend([None] * len(chunk))
        except Exception as e:  # noqa: BLE001
            batch_resp.extend([None] * len(chunk))
            batch_err.extend([f"batch:{str(e)[:200]}"] * len(chunk))
        log.info("[%s] batch chunk %d-%d/%d", model_name, s,
                 min(s + bs, len(texts)), len(texts))

    recs = []
    n_equal = n_mismatch = n_err = 0
    for i, c in enumerate(sample):
        s1, s2 = single_resp[i], batch_resp[i]
        byte_equal = (s1 is not None and s2 is not None and s1 == s2)
        if single_err[i] or batch_err[i]:
            n_err += 1
            byte_equal = False
        elif byte_equal:
            n_equal += 1
        else:
            n_mismatch += 1
        rec = {
            "model": model_name, "prec": prec,
            "query_id": c["query_id"], "lang": c["lang"],
            "condition": c["condition"], "template_idx": c["template_idx"],
            "prompt": c["prompt"],
            "single_sha256": sha256_hex(s1) if s1 is not None else None,
            "batch_sha256": sha256_hex(s2) if s2 is not None else None,
            "single_len": len(s1) if s1 is not None else -1,
            "batch_len": len(s2) if s2 is not None else -1,
            "byte_equal": byte_equal,
            "single_err": single_err[i], "batch_err": batch_err[i],
        }
        recs.append(rec)
    with open(out_dir / f"result_{model_name}.jsonl", "w",
              encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "model": model_name, "n_cells": len(recs),
        "n_byte_equal": n_equal, "n_mismatch": n_mismatch, "n_errors": n_err,
        "pass": (n_equal == len(recs) and n_mismatch == 0 and n_err == 0),
        "bs": bs, "seed": seed,
    }
    with open(out_dir / f"summary_{model_name}.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("[%s] 验证汇总: equal=%d mismatch=%d err=%d pass=%s",
             model_name, n_equal, n_mismatch, n_err, summary["pass"])

    del model, tok
    _t.cuda.empty_cache()
    gc.collect()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--n-e4b", type=int, default=200)
    ap.add_argument("--n-e2b", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from common_utils import load_config
    cfg = load_config(args.config)
    root = Path(cfg["workdir"]).expanduser()
    # 数据隔离（AUDIT #172，fix item 9）：validate_batch 产物写入 audit/ 而非
    # results/——batch/single 隔离纪律要求验证批产物永不与正式批结果混居/被
    # result_index 索引/被下游分析误用。旧产物（results/validate_batch/）保持
    # 原位不迁移（不修改已落盘数据），但已从 result_index 排除。
    out_dir = root / "audit" / "validate_batch"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells, meta = build_real_cells(cfg, root)
    print("[validate] 真实单元构造:", json.dumps(meta, ensure_ascii=False))
    if args.dry_run:
        for mn in MODEL_ORDER:
            s = stratified_sample(cells, getattr(args, "n_" + mn.split("_")[2]),
                                  args.seed)
            langs = sorted({c["lang"] for c in s})
            conds = sorted({c["condition"] for c in s})
            tpls = sorted({c["template_idx"] for c in s})
            print(f"[dry-run] {mn}: 抽样 {len(s)} 单元 | "
                  f"lang={langs} cond={conds} tpl={tpls}")
            for c in s[:2]:
                print("  样例:", c["query_id"], c["condition"],
                      "t", c["template_idx"], "|", c["prompt"][:60])
        return 0

    # 仅验证 config 注册表中的 gemma_4 模型
    models_cfg = {k: cfg["models"][k] for k in MODEL_ORDER
                  if k in cfg.get("models", {})}
    if not models_cfg:
        print("[validate] 无 gemma_4 模型可验证，FAIL")
        return 3
    summaries = []
    for mn, mconf in models_cfg.items():
        n = args.n_e4b if mn == "gemma_4_e4b" else args.n_e2b
        s = run_model(mn, mconf, cfg, root, cells, n, args.bs,
                      args.seed, out_dir)
        summaries.append(s)
    overall = all(s["pass"] for s in summaries)
    (out_dir / "overall.json").write_text(
        json.dumps({"overall_pass": overall, "summaries": summaries},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("[validate] 总体判定:", "PASS" if overall else "FAIL")
    for s in summaries:
        print("  ", s)
    return 0 if overall else 2


if __name__ == "__main__":
    sys.exit(main())
