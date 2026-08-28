#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0 DualJudge 英文验证补跑（v6.5.20 更新：适配 v6.5 双 judge E4B+E2B）。

历史背景（v6.4 审计追溯）：08-05 两次 P0 运行中，judge_big(32B-AWQ) 验证
成功（acc=0.8621）后，Mistral-24B 以 8bit 加载（≈26-30G）超 24GB 显存 →
CPU offload → DualJudge 段失败，judge_mistral 英文验证缺失。

v6.5 现状：双 judge = Gemma-4-E4B-it + Gemma-4-E2B-it（BF16 直载，16G/10G
顺序加载不超 24G）；validation 键为 judge_big/judge_small（judge_mistral 为
v6.4 残留键，已废弃）。本脚本保留为幂等补跑窗口（仅当 validation 缺
judge_small 时触发，见 pipeline.sh）。

本脚本：只补跑 dual 段（不重跑全部评分器）——
  1. 加载 E4B → 公开基准验证（幂等：已有则跳过）
  2. 卸载 E4B → 加载 E2B → 公开基准验证
  3. 双 judge 一致率（前 200 条）
  4. 写回 gates/P0_scorers.json（合并现有 validation + judge_small + dual 一致率）
  5. 更新 report/scorer_validation_on_public_benchmarks.md

纪律：真实加载推理，失败如实记录，不降级 CPU offload 不造假。
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common_utils import load_config, setup_logging  # noqa: PLC0415
    from stage_p0_measure import (load_original_responses,  # noqa: PLC0415
                                  validate_single_scorer, _DualJudgeOne)

    cfg = load_config(args.config)
    root = Path(cfg["workdir"]).expanduser()
    log = setup_logging(str(root / "logs" / "p0_dual_refill.log"), "P0_DUAL_REFILL")
    elog_path = root / "logs" / "errors.jsonl"

    log.info("=== P0 DualJudge 英文验证补跑（v6.5，E4B+E2B）启动 ===")

    # 1. 公开基准
    bench_rows = load_original_responses(
        Path(cfg["original_data_dir"]).expanduser(), log)
    log.info("公开基准行: %d", len(bench_rows))
    if not bench_rows:
        log.error("公开基准为空 → 致命 3")
        return 3

    # 2. 延迟导入评分器
    try:
        from scorer_utils import DualJudgeScorer  # noqa: PLC0415
    except ImportError as e:
        log.error("评分器导入失败: %s", e)
        return 3

    gates_file = root / "gates" / "P0_scorers.json"
    rpt_file = root / "report" / "scorer_validation_on_public_benchmarks.md"

    # 3. 加载现有 P0_scorers.json（保留已有验证数据）
    gates = {}
    if gates_file.exists():
        try:
            gates = json.loads(gates_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            gates = {}
    validation = dict(gates.get("validation", {}))

    # 4. DualJudge 补跑（E4B 已验证则跳过，重点补 E2B；v6.5 §4.1）
    # v6.5.20-fix（问题 70）：键名 judge_mistral → judge_small（v6.5 双 judge
    # small = Gemma-4-E2B-it；judge_mistral 为 v6.4 Mistral-24B 残留键）。
    dual = None
    try:
        dual = DualJudgeScorer(cfg["scorers"]["judge_big_model"],
                               cfg["scorers"].get(
                                   "judge_small_model",
                                   cfg["scorers"].get("judge_mistral_model")),
                               load_in_4bit=False)  # BF16 直载（v6.5 §10.4）
        # 4a. E4B（已有则跳过验证）
        if "judge_big" not in validation:
            m_big = validate_single_scorer(_DualJudgeOne(dual, "big"),
                                           bench_rows, log)
            if m_big and "error" not in m_big:
                validation["judge_big"] = m_big
                log.info("judge_big(E4B) 英文验证: %s", m_big)
            else:
                log.warning("judge_big 验证无数据: %s", m_big)
        dual.unload_big()
        # 4b. E2B（v6.5 新配置，重点补）
        # v6.5.14-fix 2026-08-08（问题 16）：DualJudgeScorer 无 _load_mistral
        # 方法（v6.5 迁移 Mistral-24B → Gemma-4-E2B-it 时方法已删），
        # 改为 load_mistral_only（先 unload_big 再 _load_small）。
        try:
            dual.load_mistral_only()
            m_small = validate_single_scorer(_DualJudgeOne(dual, "small"),
                                             bench_rows, log)
            if m_small and "error" not in m_small:
                validation["judge_small"] = m_small
                log.info("judge_small(E2B) 英文验证: %s", m_small)
            else:
                log.warning("judge_small 验证无数据: %s", m_small)
        except Exception as e:  # noqa: BLE001
            log.error("judge_small(E2B) 加载/验证失败: %s", str(e)[:300])
            with open(elog_path, "a", encoding="utf-8") as ef:
                ef.write(json.dumps({
                    "stage": "P0_DUAL_REFILL", "event": "e2b_load_failed",
                    "reason": str(e)[:500],
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S")},
                    ensure_ascii=False) + "\n")
        dual.unload_mistral()

        # 4c. 双 judge 一致率（前 200 条，顺序加载）
        dual_stats = {}
        try:
            sample = [r for r in bench_rows[:200]
                      if r.get("behavior") and r.get("response")]
            dual.load_big_only()
            big_labels = []
            for r in sample:
                try:
                    big_labels.append(dual.score_one_big(
                        r["behavior"], r["response"]))
                except Exception:  # noqa: BLE001
                    big_labels.append(None)
            dual.unload_big()
            dual.load_mistral_only()
            mist_labels = []
            for r in sample:
                try:
                    mist_labels.append(dual.score_one_mistral(
                        r["behavior"], r["response"]))
                except Exception:  # noqa: BLE001
                    mist_labels.append(None)
            dual.unload_mistral()
            agree = total = 0
            for lb, lm in zip(big_labels, mist_labels):
                if lb is not None and lm is not None:
                    total += 1
                    if lb == lm:
                        agree += 1
            if total:
                dual_stats["agreement_rate"] = round(agree / total, 4)
                dual_stats["n"] = total
                dual_stats["dispute_rate"] = round(1 - agree / total, 4)
                log.info("双 judge 一致率: %.4f (n=%d)", dual_stats["agreement_rate"],
                         total)
        except Exception as e:  # noqa: BLE001
            log.warning("双 judge 一致率失败: %s", str(e)[:200])
    finally:
        if dual is not None:
            try:
                dual.close()
            except Exception:  # noqa: BLE001
                pass

    # 5. 写回 gates/P0_scorers.json
    gates["stage"] = "P0"
    gates["n_scorers"] = len(validation)
    gates["validation"] = validation
    # primary = acc 最高的正式评分器
    best_name = None
    best_acc = -1
    for nm, m in validation.items():
        if isinstance(m, dict) and "acc" in m:
            if m["acc"] > best_acc:
                best_acc, best_name = m["acc"], nm
    if best_name:
        gates["primary"] = best_name
    gates["dual_judge"] = dual_stats
    # v6.5.20-fix（问题 70）：note 原为 v6.4 时代表述（StrongREJECT-14B /
    # 双judge 32B+Mistral-24B 均已不在 v6.5 评分链）→ 改 v6.5 实际口径。
    gates["note"] = ("v6.5 纯本地评分：4 评分器（HarmBench/StrongREJECT-E4B/"
                     "Gemma-E2B/双judge E4B+E2B）全 Gemma 4 家族 + 1 异构交叉验证"
                     "（Qwen2.5-3B，仅公开基准+争议子集核对，不参与主推断）；"
                     "keyword 仅辅助基线；全评分器 BF16 直载（QAT 仓库 404）")
    gates_file.parent.mkdir(parents=True, exist_ok=True)
    gates_file.write_text(json.dumps(gates, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    log.info("P0_scorers.json 写回: %s (n_scorers=%d)", gates_file, len(validation))

    # 6. 更新验证报告
    # v6.5.20-fix（问题 70）：标题 v6.4 → v6.5
    val_lines = ["# 评分器公开基准验证（v6.5）\n",
                 f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 f"- 基准行数: {len(bench_rows)}",
                 f"- 主评分器: {gates.get('primary', 'N/A')}\n",
                 "| 评分器 | acc | recall | FPR | FNR | n |\n",
                 "|---|---|---|---|---|---|\n"]
    for name, m in validation.items():
        if isinstance(m, dict) and "acc" in m:
            val_lines.append(f"| {name} | {m['acc']} | {m['recall']} | "
                             f"{m['fpr']} | {m['fnr']} | {m['n']} |\n")
    val_lines.append(f"\n## 双 judge 一致率基线\n")
    if dual_stats:
        val_lines.append(f"- 一致率: {dual_stats.get('agreement_rate', 'N/A')} "
                         f"(n={dual_stats.get('n', 0)})\n")
        val_lines.append(f"- 争议率: {dual_stats.get('dispute_rate', 'N/A')}\n")
    else:
        val_lines.append("- 计算失败（见日志）\n")
    rpt_file.parent.mkdir(parents=True, exist_ok=True)
    rpt_file.write_text("".join(val_lines), encoding="utf-8")
    log.info("验证报告更新: %s", rpt_file)

    # 7. 结论
    n_ok = sum(1 for m in validation.values()
               if isinstance(m, dict) and "acc" in m)
    log.info("=== P0 DualJudge 补跑完成（validation %d 项）===", n_ok)
    return 0 if n_ok >= 4 else 2


if __name__ == "__main__":
    sys.exit(main())
