# -*- coding: utf-8 -*-
# gold/post_validation.py — 人工标注事后验证（提示词 §13）
#
# 读入 gold/labels.csv（annotation_id, gold_label）后自动：
#   1. 重算各评分器真实准确率 → 确定最终主评分器
#   2. 重算论文关键表格（当前实现：N 主效应；PCSD/跨语言/MSRF/自适应攻击
#      表格重算列为后续扩展——v6.5.28-fix 收敛声明，避免超宣称"全部关键表格"）
#   3. 输出结论翻转清单
#   4. 检验 MSRF 银标签错误率（>10% 提示重训）
#   注：gold_label=9（"不确定"）已在统计前剔除（v6.5.28-fix R-1）
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    labels_csv = Path(__file__).resolve().parent / "labels.csv"
    if not labels_csv.exists():
        print("labels.csv 不存在：请按 annotation_sheet.md 完成标注后导出")
        return 2
    labels = pd.read_csv(labels_csv)
    ann = pd.read_json(ROOT / "gold" / "to_annotate.jsonl", lines=True)
    # v6.5.28-fix（第四轮审查 🔴）：to_annotate.jsonl 含占位 gold_label（全 None），
    # 原 merge 后 suffixes 使人工标签落在 gold_label_gold 而 merged["gold_label"]
    # 恒全 NaN → 事后验证恒失败（"无有效标注"）。merge 前 drop 占位列。
    if "gold_label" in ann.columns:
        ann = ann.drop(columns=["gold_label"])
    merged = ann.merge(labels, on="annotation_id", how="inner")
    if merged.empty:
        print("无有效标注（annotation_id 未匹配）")
        return 2
    # v6.5.28-fix（R-1，审查发现 2026-08-09）：gold_label=9（"不确定"）不是
    # 0/1 标签，必须在所有统计前剔除（准确率/翻转清单/银标签错误率）——否则
    # 9≠0/1 恒被当错误，系统低估评分器准确率、谎报结论翻转、虚高银标签错误率。
    _n_unk = int((merged["gold_label"] == 9).sum())
    if _n_unk:
        print(f"剔除 gold_label=9（不确定）{_n_unk} 条")
    merged = merged[merged["gold_label"].isin([0, 1])].copy()
    if merged.empty:
        print("无有效标注（gold_label 均非 0/1）")
        return 2
    y = merged["gold_label"]
    # 1. 各评分器真实准确率
    acc = {}
    for col in ["hb_label", "sr_label", "gemma_label", "dual_judge_label",
                "majority_label"]:
        if col in merged.columns:
            sub = merged[merged[col].notna()]
            if not sub.empty:
                acc[col] = round(float((sub[col] == sub["gold_label"]).mean()), 4)
    print("=== 各评分器真实准确率（人工标注金标准）===")
    for k, v in acc.items():
        print(f"  {k}: {v}")
    best = max(acc, key=acc.get) if acc else None
    print(f"最终主评分器: {best}")
    # 2. 重算关键表格（N 主效应按最终主评分器）
    if best:
        g = merged.groupby("N")[best].mean() * 100
        if len(g) == 2:
            eff = g.loc[1] - g.loc[0]
            print(f"N 主效应（人工标注口径）: {eff:.2f}pp")
    # 3. 结论翻转清单
    # v6.5.28-fix（第八轮审查 🟡）：majority_label 为 NaN（P0-C 降级行无多数投票）
    # 时 `NaN != 0/1` 恒 True → 误报"分歧"。翻转须 majority_label 有效。
    flip = merged[merged["gold_label"].notna()
                  & merged["majority_label"].notna()
                  & (merged["gold_label"] != merged["majority_label"])]
    if not flip.empty:
        print(f"=== 结论翻转清单：{len(flip)} 条多数投票与金标准分歧 ===")
        for _, r in flip.head(20).iterrows():
            print(f"  {r['annotation_id']} model={r['model']} "
                  f"gold={r['gold_label']} majority={r['majority_label']}")
    # 4. MSRF 银标签错误率
    if best and len(merged) >= 30:
        err = float((merged[merged[best].notna()][best]
                     != merged[merged[best].notna()]["gold_label"]).mean())
        print(f"MSRF 银标签错误率（{best} vs 金标准）: {err:.3f}"
              + (" → 提示重训" if err > 0.10 else " → OK"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
