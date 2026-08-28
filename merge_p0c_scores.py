#!/usr/bin/env python
"""merge_p0c_scores.py — 合并 s_p0c_score.py 双卡 5 评分器输出 → 完整 P0-C 评分表。

数据源（part0=5400 行 + part1=5400 行，各评分器独立列文件，按 response_id 对齐）：
  results/p0c_hb_part{P}.jsonl   → harmbench_label
  results/p0c_sr_part{P}.jsonl   → strongreject_label
  results/p0c_gj_part{P}.jsonl   → gemma_label
  results/p0c_djb_part{P}.jsonl  → judge_big_label
  results/p0c_djm_part{P}.jsonl  → judge_mistral_label

基础列：responses/P0C/lalm_responses.jsonl（10800 行，含 response_id/model/modality/
  query_idx/prompt/response/precision/phase）。

condition 映射：复刻 stage_p0c.py cond_map（L864-868 + prompt_index L749-754）——
  prompt_index 顺序 = for cond in [baseline, storytelling, unrestricted]:
                        for ti in 3 templates: for qi in n_per(200):
  全局枚举 i(0..1799) → baseline[0:600]/storytelling[600:1200]/unrestricted[1200:1800]
  响应 query_idx 即全局 cell 索引（实测 0..1799，每值 6 行 = 3 模型×2 模态）。

科学纪律：只新增 results/p0c_scored.parquet（首次创建）；绝不覆盖任何已有产物；
response_id 精确对齐，None 如实传播，合并后完整性校验。等价性由逐字段复刻保证。

用法：
  python merge_p0c_scores.py --check-only   # 只校验列文件齐全度，不写 parquet
  python merge_p0c_scores.py                # 合并 + 写 parquet + 完整性校验
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
RESP = ROOT / "responses" / "P0C" / "lalm_responses.jsonl"
OUT = RES / "p0c_scored.parquet"

N_PER = 200
N_TPL = 3
CONDS = ["baseline", "storytelling", "unrestricted"]  # FRAMING_TEMPLATES dict 保序

SCORERS = {
    "p0c_hb": "harmbench_label",
    "p0c_sr": "strongreject_label",
    "p0c_gj": "gemma_label",
    "p0c_djb": "judge_big_label",
    "p0c_djm": "judge_mistral_label",
}
BASIC = ["response_id", "model", "modality", "query_idx", "prompt",
         "response", "precision", "phase"]


def build_cond_map():
    """复刻 stage_p0c.py prompt_index 顺序 → 全局枚举 i → condition。"""
    cond_map = {}
    ti_map = {}
    i = 0
    for cond in CONDS:
        for _ti in range(N_TPL):
            for _qi in range(N_PER):
                cond_map[i] = cond
                ti_map[i] = _ti
                i += 1
    assert i == 1800, i
    return cond_map, ti_map


def load_responses():
    rows = []
    with RESP.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def load_scorer(file_path, col):
    d = {}
    with file_path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            r = json.loads(s)
            d[r["response_id"]] = r.get(col)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="只校验 5×2 列文件齐全度与覆盖度，不写 parquet")
    args = ap.parse_args()

    rows = load_responses()
    rids = [r["response_id"] for r in rows]
    rid_set = set(rids)
    n_resp = len(rids)
    print(f"响应总行数: {n_resp} | 唯一 response_id: {len(rid_set)}")
    assert n_resp == 10800 and len(rid_set) == 10800, "响应文件应为 10800 唯一行"

    # 加载 5 评分器列文件（part0+part1）
    scorers = {}
    ok = True
    for tag, col in SCORERS.items():
        merged = {}
        for part in (0, 1):
            fp = RES / f"{tag}_part{part}.jsonl"
            if not fp.exists():
                print(f"  ✗ 缺失列文件: {fp.name}")
                ok = False
                continue
            d = load_scorer(fp, col)
            merged.update(d)
        cover = sum(1 for rid in rids if rid in merged)
        print(f"  [{col}] 覆盖 {len(merged)} 唯一 rid / 10800 ({cover}/{n_resp} 命中, "
              f"{len(merged) - cover} 非响应集)")
        scorers[col] = merged
        if cover < n_resp:
            ok = False

    if not ok:
        print("\n!! 列文件不齐全，中止（P0-C 评分未全部完成）")
        sys.exit(2)

    if args.check_only:
        print("\n--check-only：所有 5 评分器列文件已齐全，可合并")
        return 0

    if OUT.exists():
        print(f"!! 目标已存在，拒绝覆盖: {OUT}")
        sys.exit(3)

    cond_map, ti_map = build_cond_map()

    # 逐行合并：response_id 精确对齐（不依赖行序，按 rid 键控）
    recs = []
    n_hb_none = n_sr_none = 0
    for r in rows:
        rid = r["response_id"]
        qi = r.get("query_idx")
        rec = {c: r.get(c) for c in BASIC}
        rec["condition"] = cond_map.get(qi) if qi is not None else None
        rec["template_idx"] = ti_map.get(qi) if qi is not None else None
        for col in SCORERS.values():
            v = scorers[col].get(rid)
            rec[col] = v
        recs.append(rec)
        if rec["harmbench_label"] is None:
            n_hb_none += 1
        if rec["strongreject_label"] is None:
            n_sr_none += 1

    df = pd.DataFrame(recs)
    assert len(df) == 10800, len(df)
    assert df["response_id"].nunique() == 10800, "response_id 重复！"

    # 完整性校验：评分列非空率 + condition 分布
    print("\n=== 合并校验 ===")
    print(f"行数: {len(df)} | 唯一 response_id: {df['response_id'].nunique()}")
    for col in SCORERS.values():
        non_null = int(df[col].notna().sum())
        print(f"  {col}: 非空 {non_null}/{len(df)} ({100 * non_null / len(df):.1f}%)")
    print(f"  harmbench_label None: {n_hb_none} | strongreject_label None: {n_sr_none}")
    print("\ncondition 分布:")
    print(df["condition"].value_counts().sort_index().to_string())
    if df["condition"].isna().any():
        print("!! condition 存在 None，映射有缺口")
        sys.exit(2)

    # 写出 parquet（原子：先 tmp 再 rename）
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(OUT)
    print(f"\nparquet -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    # 读回验证
    df2 = pd.read_parquet(OUT)
    need = set(SCORERS.values()) | {"condition", "response", "model",
                                    "modality", "prompt"}
    miss = [c for c in need if c not in df2.columns]
    print(f"读回行数: {len(df2)} | 契约缺列: {miss if miss else '无'}")
    assert len(df2) == 10800 and not miss, "读回校验失败"
    print("合并完成，契约校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
