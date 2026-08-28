# -*- coding: utf-8 -*-
"""固化完整 PILOT 查询文件（决策 D1 落地，可复现脚本）。

背景：决策 D1（zh_n 300→400，RESEARCH_PROTOCOL 冻结修订表 §11）需阶段 D 重跑；
池规模变化会使 stage_p1_pilot 的 sample_queries（rng.sample(range(len(pool)), n)）
结果漂移 → PILOT 查询集改变 → 已真实推理的 E4B text 响应与 7200 个音频全部作废。
本脚本从冻结文本（results/p1_pilot_queries_zh.json，PILOT 实际使用顺序）匹配回
当前池字典（保留 query_id/en/category），生成 results/p1_pilot_queries_full.json，
供 stage_p1_pilot 在池变化后保持 PILOT 集稳定（v6.5.26-fix）。

用法：在流水线根目录执行：
  python freeze_pilot.py [--root /root/lalm_framing_revision_v6] [--queries data/queries_v1.jsonl]

输出：results/p1_pilot_queries_full.json（含 version="v6.5"、seed、n、queries 全字段）。
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="流水线根目录（默认从 config workdir 解析或相对路径）")
    ap.add_argument("--queries", default="data/queries_v1.jsonl",
                    help="查询池 jsonl（PILOT 抽样源）")
    args = ap.parse_args()

    root = Path(args.root or ".").expanduser().resolve()
    pool_f = root / args.queries
    if not pool_f.exists():
        # 回退到 config 解析 workdir
        try:
            import yaml
            cfg = yaml.safe_load(open(root / "pipeline_config.yaml", encoding="utf-8"))
            workdir = Path(str(cfg.get("workdir", ""))).expanduser()
            if workdir.is_absolute():
                root = workdir
                pool_f = root / args.queries
        except Exception as e:  # noqa: BLE001
            print(f"查询池解析失败（--root 指定更稳妥）: {e}")
    if not pool_f.exists():
        print(f"FATAL: 查询池缺失 {pool_f}")
        return 2

    # 1. 当前池（PILOT 抽样源）
    pool = [json.loads(l) for l in pool_f.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    zh_by_text = {r.get("zh", ""): r for r in pool}

    # 2. 冻结文本（已抽样 150，PILOT 实际使用顺序）
    frozen_f = root / "results" / "p1_pilot_queries_zh.json"
    if not frozen_f.exists():
        print(f"FATAL: 冻结文件缺失 {frozen_f}（需先跑过 stage_p1_pilot 完成抽样）")
        return 2
    frozen = json.loads(frozen_f.read_text(encoding="utf-8"))
    texts = frozen["queries"]

    # 3. 匹配
    queries = []
    missed = []
    for t in texts:
        r = zh_by_text.get(t)
        if r is None:
            missed.append(t)
        else:
            queries.append({
                "query_id": r.get("query_id"),
                "zh": t,
                "en": r.get("en", ""),
                "en_from": r.get("en_from", ""),
                "category": r.get("category", ""),
                "source": r.get("source", ""),
            })
    print(f"匹配成功: {len(queries)}/{len(texts)}；缺失: {len(missed)}")
    for m in missed[:10]:
        print("  MISS:", m[:60])
    if missed:
        return 2

    # 4. query_id 校验
    bad_ids = [q["query_id"] for q in queries
               if not q.get("query_id") or not q["query_id"].startswith("q")]
    if bad_ids:
        print("FATAL: query_id 异常", bad_ids[:5])
        return 2

    # 5. 落盘完整 PILOT 文件
    out_f = root / "results" / "p1_pilot_queries_full.json"
    out_f.write_text(json.dumps({
        "version": "v6.5",
        "seed": frozen.get("seed"),
        "n": len(queries),
        "source": "frozen_texts_matched_to_pool",
        "queries": queries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写: {out_f}（{len(queries)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
