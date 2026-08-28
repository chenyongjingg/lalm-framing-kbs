#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refill_p0c_nulls.py — AUDIT #187b：补评 P0-C GJ/DJM null 行。

根因（事实）：s_p0c_score.py load_done 只收集 response_id（不看 label），
`if len(done) < len(rows)` 对 null 行也成立 len(done)==len(rows) → "已完整，跳过"。
因此 222 个 null 行（GJ=111: part0 70+part1 41；DJM=111: part0 70+part1 41）
不会被 resume 补评。

本脚本：从 results/p0c_{gj,djm}_part{0,1}.jsonl 找出 label=None 的行，
从 responses/P0C/lalm_responses.jsonl 取 behavior=prompt / generation=response
（与 s_p0c_score.py 语义复刻一致），用 GemmaJudge / DualJudge(mistral) 重新评分，
重试至多 3 次；仍 None 则保留 None 并打印 STUBBORN（绝不凭空填值）。
写回原子重建原 part 文件（保持行序与 response_id 对齐，其他行原样不动）。

纪律：#1 禁止凭空生成数据。只写 results/p0c_{gj,djm}_part*.jsonl 的 null 行，
不触碰主链文件 / 其他评分器产物 / 运行中进程。GPU 由调用方指定 CUDA_VISIBLE_DEVICES。
"""
import json
import gc
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path("/root/lalm_framing_revision_v6")
RESP = ROOT / "responses" / "P0C" / "lalm_responses.jsonl"
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))

from scorer_utils import GemmaJudgeScorer, DualJudgeScorer  # noqa: E402

cfg = yaml.safe_load(open(ROOT / "pipeline_config.yaml", encoding="utf-8"))

# ---- 读响应（行为=prompt 已验等价原始有害 query；按 response_id 索引） ----
rows = [json.loads(l) for l in open(RESP, encoding="utf-8")]
rid2row = {r["response_id"]: r for r in rows}

# ---- 收集 null 行（保留原对象以便原位写回） ----
targets = {}  # col -> [(path, objs, null_rids)]
for tag, col in (("gj", "gemma_label"), ("djm", "judge_mistral_label")):
    for part in (0, 1):
        p = RES / ("p0c_%s_part%d.jsonl" % (tag, part))
        if not p.exists():
            continue
        objs = []
        nulls = []
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            objs.append(o)
            if o.get(col) is None:
                nulls.append(o["response_id"])
        if nulls:
            targets.setdefault(col, []).append((p, objs, nulls))
            print("[%s] part=%s null=%d" % (tag, p.name[-6:-5], len(nulls)))

if not targets:
    print("NO_NULLS — 无需补评")
    sys.exit(0)


def atomic_write(path, objs, col):
    """重建 part 文件（保持行序与 response_id 对齐，其他行原样）。"""
    tmp = path.with_suffix(".jsonl.tmp_refill")
    with tmp.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    tmp.replace(path)


def retry_score(fn, r, attempts=3):
    """评分 + 至多 3 次重试；仍失败返回 None（不凭空填值）。

    fn 必须已包装为返回最终 label（GJ 需 int(score_one().get("label"))——
    label=None 时 int(None) 抛异常 → 触发重试）。"""
    for _ in range(attempts):
        try:
            lab = fn(r["prompt"], r["response"])
            if lab is not None:
                return lab
        except Exception:
            continue
    return None


# ---- 1) GemmaJudge：补 gemma_label ----
if "gemma_label" in targets:
    gj = GemmaJudgeScorer(cfg["scorers"]["gemma_judge_model"], load_in_4bit=False)

    def _gj_lab(prompt, resp):
        # score_one 返回 dict；label=None → int(None) 抛异常 → 触发重试
        return int(gj.score_one(prompt, resp).get("label", 0))

    for p, objs, nulls in targets["gemma_label"]:
        stubborn = 0
        for rid in nulls:
            r = rid2row[rid]
            lab = retry_score(_gj_lab, r)
            if lab is None:
                stubborn += 1
            for o in objs:
                if o["response_id"] == rid:
                    o["gemma_label"] = lab
                    break
        atomic_write(p, objs, "gemma_label")
        print("[gj] %s 补评 %d 行，STUBBORN=%d" % (p.name, len(nulls), stubborn))
    gj.close()
    del gj
    gc.collect()
    torch.cuda.empty_cache()
    print("GJ refill done")

# ---- 2) DualJudge(mistral)：补 judge_mistral_label ----
if "judge_mistral_label" in targets:
    dj = DualJudgeScorer(
        cfg["scorers"]["judge_big_model"],
        cfg["scorers"].get("judge_small_model",
                           cfg["scorers"].get("judge_mistral_model")),
        load_in_4bit=False,
    )
    dj.load_mistral_only()
    for p, objs, nulls in targets["judge_mistral_label"]:
        stubborn = 0
        for rid in nulls:
            r = rid2row[rid]
            # score_one_mistral 直接返回 label（int 或 None）
            lab = retry_score(dj.score_one_mistral, r)
            if lab is None:
                stubborn += 1
            for o in objs:
                if o["response_id"] == rid:
                    o["judge_mistral_label"] = lab
                    break
        atomic_write(p, objs, "judge_mistral_label")
        print("[djm] %s 补评 %d 行，STUBBORN=%d" % (p.name, len(nulls), stubborn))
    dj.close()
    del dj
    gc.collect()
    print("DJM refill done")

print("REFILL_COMPLETE")
