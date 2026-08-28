# -*- coding: utf-8 -*-
"""
gate_g1.py — 闸门 G1（v6.5 三口径判据 · 异质性软化为稳健性佐证）

输入: results/p1_pilot_effects.json（stage_p1_pilot.py 输出）
判据（v6.5 提示词 §5.3 / STAGE_CONTRACTS §G1）：
  C1. 三口径方向一致：主评分器 / 双judge一致 / 多数投票 的 N_main 方向同号
  C2. 效应量 ≥ 10pp（决策口径 = dual_judge；v6.7-r5-fix 终审 CRIT-1：
      中文唯一决策口径为 dual_judge，原 primary=judge_big 仅敏感性）
  C3. CI 不含 0（决策口径 = dual_judge，bootstrap；缺失即 fail-closed）
  C4. 双 judge 一致子集可用且无严重争议（争议率 < 0.5 或可解释）
  C5. 模型异质性（v6.5 软化）：仅作稳健性佐证（strong_evidence/acceptable/
      warning/unknown），不作 through/not-through 硬判据（v6.5 §0 判断 3 + §5.3 G1-e）
附加：操纵检验通过（manipulation_check）

输出: gates/G1.json（机器可读）
退出: 0=通过 / 1=不通过（转探索性分析）/ 3=致命（数据缺失）
"""

import argparse
import json
import sys
from pathlib import Path

from common_utils import load_config, setup_logging, JsonlLogger

STAGE = "g1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--effects", default=None,
                    help="p1_pilot_effects.json 路径（默认 config.workdir/results/...）")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["workdir"]).expanduser()
    log = setup_logging(str(root / "logs" / f"{STAGE}.log"), STAGE)
    jlog = JsonlLogger(str(root / "logs" / f"{STAGE}.jsonl"))

    if args.effects:
        eff_path = Path(args.effects)
    else:
        eff_path = root / "results" / "p1_pilot_effects.json"
    if not eff_path.exists():
        log.error("G1 输入缺失: %s → 致命 3", eff_path)
        return 3
    eff = json.loads(eff_path.read_text(encoding="utf-8"))
    log.info("G1 输入: %s", eff_path)

    # ---- v6.7-r5-fix（终审 Major B）：移除 recalc_v64.py 自动重算 ----
    # 原自愈路径 (a) 调用含 C-1 聚类 bootstrap 坍缩 bug（isin(set(...)) →
    # CI 窄约 31%）的 legacy 脚本，(b) 以 v6.4 6 票旧口径覆写 v6.5 effects。
    # 非 v6.5 effects 一律 fail-closed 披露，交由操作者用当前代码重算，
    # 不得静默覆写/沿用。
    if eff.get("version") != "v6.5":
        log.error("G1 输入 effects version=%s 非 v6.5（预注册口径）→ 不自动重算；"
                  "已禁用废弃 recalc_v64.py（含聚类 bootstrap 坍缩 bug，审计 CRIT 修复）。"
                  "须用当前 stage_p1_pilot 重算后再判，G1 判据如实披露。",
                  eff.get("version"))
        jlog.event(stage=STAGE, event="non_v65_effects", version=eff.get("version"),
                   note="非 v6.5 effects，已移除 recalc_v64 自动重算（终审 Major B）")

    # ---- v6.7-r5-fix（终审 Major C）：数据完整性硬判据 ----
    # P1-PILOT 推理缺口（音频/文本/query 短少）原仅 WARNING → effects 可能
    # 基于部分数据。stage_p1_pilot 现写 data_complete 标记（0 缺口才 True）；
    # 缺失或 False → G1 fail-closed，不静默判定。
    _dc = eff.get("data_complete")
    data_complete = bool(_dc)
    if _dc is None:
        log.warning("G1: effects 无 data_complete 标记（旧产物）→ 按不完整披露")
    elif not data_complete:
        log.error("G1: P1-PILOT 数据未完整（data_complete=False，存在推理缺口）"
                  "→ 按不通过披露，须补齐后重判")

    n_main = eff.get("N_main", {}) or {}
    n_x_as = eff.get("N_x_A_s", {}) or {}
    manip = eff.get("manipulation_check", {}) or {}

    # v6.5.26-fix（D2 裁决落地，审查发现 2026-08-08）：G1(a)"N 主效应或 N×A_s
    # 交互方向双模型一致"必须为硬判据。v6.5 仅软化 (e)（协议 §5.3 明示），(a) 未软化。
    # both_models.consistent 由 stage_p1_pilot 真实写入（E4B/E2B 对 N_main 主评分器
    # 口径方向的同号判定）；数据缺失（None）→ fail-closed 披露，不得静默通过。
    g1_cfg = cfg.get("g1", {})
    _min_pp = g1_cfg.get("min_effect_pp", 10)
    _ci_must_exclude = not g1_cfg.get("ci_include_zero", False)

    # C1: 三口径方向一致（primary/dual_judge/majority 的 direction）
    # v6.5.28-fix（M8，审查发现 2026-08-09）：协议 §5.3(d) "三口径一致"要求
    # 三口径齐全且同号。原 `len(dirs)>=2` 允许一口径缺失仍判通过（宽松）。
    # 三口径任一缺失 → fail-closed 披露（与 (a)/(c) 缺失 fail-closed 对称）。
    dirs = []
    for m in ("primary", "dual_judge", "majority"):
        v = n_main.get(m)
        if isinstance(v, dict) and v.get("direction") in ("up", "down", "none"):
            dirs.append(v["direction"])
    direction_consistent = len(dirs) == 3 and len(set(dirs)) == 1
    if len(dirs) < 3:
        log.warning("三口径不齐（%d/3: %s）→ C1(d) 按不满足披露", len(dirs), dirs)
        jlog.event(stage=STAGE, event="three_way_incomplete",
                   n_metrics=len(dirs), directions=dirs,
                   note="G1(d) 三口径一致要求三口径齐全，缺失按不满足披露")

    # C2/C3: 决策口径 —— v6.7-r5-fix（终审 CRIT-1）：PILOT 全中文，预注册
    # （prompt.md L190-193 + §5.3(d)）规定中文唯一决策口径 = dual_judge。
    # 原实现读 n_main["primary"]（= P0 单评分器 judge_big，中文 FNR≈0.4，
    # 预注册已降级为仅敏感性）→ 决策建立在被降级口径上。现改读 dual_judge
    # 的 effect_pp/ci（stage_p1_pilot L1562-1574 已按 query 配对 bootstrap 计算）。
    decision_caliber = "dual_judge"
    decision_v = (n_main.get(decision_caliber)
                  if isinstance(n_main.get(decision_caliber), dict) else {})
    eff_pp = decision_v.get("effect_pp")
    # C2: 效应量 ≥ 10pp（决策口径；阈值读 config g1.min_effect_pp）
    effect_ge_10pp = (eff_pp is not None and abs(eff_pp) >= _min_pp)
    # C3: CI 不含 0（决策口径；读 config g1.ci_include_zero）
    ci = decision_v.get("ci")
    if ci is None:
        # ci 缺失（数据不足/未计算）不得静默跳过——显式告警 + jlog 披露
        log.warning("G1 C3: 决策口径 %s 的 CI 缺失（数据不足/未计算）→ 按不满足披露",
                    decision_caliber)
        jlog.event(stage=STAGE, event="ci_missing_disclosure",
                   caliber=decision_caliber,
                   note="decision_caliber ci=None，C3 无法核验，按不满足披露")
    ci_excludes_zero = bool(ci and len(ci) == 2 and not (ci[0] <= 0 <= ci[1]))
    # primary（judge_big）仅作敏感性披露（预注册 R2），不参与 C2/C3 决策
    primary_v = n_main.get("primary") if isinstance(n_main.get("primary"), dict) else {}

    # C4: 双 judge 可用且无严重争议（争议率 < 0.5）
    # v6.4 协议 §6 / STAGE_CONTRACTS §G1：
    #   可用性 = 双 judge 均评分的样本数 > 0 且 effect_pp 可算
    #   争议判据 = 严格争议率 < 0.5（双 judge 均评分样本中不一致比例）。
    #   争议率缺失时降级为"可解释"（披露不足，不判死），协议允许"或可解释"。
    dual_v = n_main.get("dual_judge")
    dual_available = isinstance(dual_v, dict) and dual_v.get("effect_pp") is not None
    dispute_rate = dual_v.get("dispute_rate") if isinstance(dual_v, dict) else None
    dual_coverage = dual_v.get("dual_coverage") if isinstance(dual_v, dict) else None
    if dispute_rate is None:
        # 旧 effects（无 dispute_rate 字段）→ 无法核验争议率，如实披露并降级
        dispute_ok = None
        dispute_note = ("effects 无 dispute_rate 字段（旧口径产物）→ 争议判据无法核验，"
                        "按可解释处理（非否决项）")
    else:
        dispute_ok = dispute_rate < 0.5
        dispute_note = (f"争议率={dispute_rate}（双 judge 均评分样本 "
                        f"{dual_v.get('dual_total_n')} 条，覆盖率={dual_coverage}）"
                        f"→ {'<0.5 通过' if dispute_ok else '≥0.5 不通过'}")

    # C5: 模型异质性（v6.5 软化 — 仅作稳健性佐证，不作硬性判据）
    # v6.5 提示词 §0 顶层判断 3："模型间异质性不作主贡献结论的唯一依据，仅作稳健性佐证，
    #   主效应以三口径一致 + 混合效应模型为准"
    # v6.5 §5.3 闸门 G1 判据 (e)："同族模型的异质性弱，v6.5 起 (e) 仅作稳健性佐证，
    #   不作通过与否的硬性判据；方向一致但量级差异由规模/架构差异解释即为可接受"
    # 实现：从 passed 必要条件移除；方向一致(consistent=True)记"强佐证"，
    #   方向一致但量级差异(consistent=False 且可归因模态)记"可接受佐证"，
    #   不可解释差异记"警示（降级表述）"——均不阻塞 G1 通过。
    both = eff.get("both_models", {}) or {}
    # G1 (a)（v6.5.26-fix，D2 裁决）：N 主效应方向双模型一致 = 硬判据。
    #   both_models.consistent=True → 通过；False → 硬性不通过；
    #   None（数据缺失，无法核验）→ fail-closed 披露（纪律 #2 不静默通过）。
    # 注：N×A_s 交互仅 E4B（音频单元唯一模型，协议 §1(c)）可算，"双模型一致"
    # 无法对 N×A_s 独立核验，故 (a) 以 N_main 双模型方向一致为可核验形式。
    if both.get("consistent") is True:
        n_dir_consistent = True
    elif both.get("consistent") is False:
        n_dir_consistent = False
    else:
        n_dir_consistent = None
        log.warning("G1 (a): both_models.consistent 缺失（无法核验双模型方向一致）"
                    "→ 按不通过披露")
        jlog.event(stage=STAGE, event="both_models_missing",
                   note="G1(a) 双模型方向一致无法核验，fail-closed 披露")
    # v6.5.13-fix 2026-08-08：stage_p1_pilot 输出 N_x_A_s.<metric>.direction（嵌套），
    # 原读 nx.get("direction") 顶层键恒 None → consistent=False 时异质性恒标 warning。
    # 改为读 primary 口径方向（与 G1 主评分器口径一致）。
    n_x_as_direction = None
    if isinstance(n_x_as.get("primary"), dict):
        n_x_as_direction = n_x_as["primary"].get("direction")
    heterogeneity_explainable = both.get("consistent") is True or (
        both.get("consistent") is False and n_x_as_direction in
        ("audio_stronger", "text_stronger"))
    if both.get("consistent") is True:
        hetero_soft = "strong_evidence"   # 双模型方向一致 → 强稳健性佐证
    elif heterogeneity_explainable:
        hetero_soft = "acceptable"        # 方向一致但量级差异可归因于模态/规模/架构
    elif both.get("consistent") is False:
        hetero_soft = "warning"           # 不可解释的模型间差异 → 结论降级表述（不阻塞）
    else:
        hetero_soft = "unknown"           # effects 无 both_models 字段 → 披露缺失

    # 操纵检验（强制）
    # v6.5.23-fix（问题 88，2026-08-08）：原判据
    #   `bool(manip.get("passed")) or bool(manip.get("baseline_vs_full_differs"))`
    #   使 passed=False 时仍可过闸——"干预未改变目标属性/改变了保持变量"的失败
    #   单元混入下游，违反提示词 §5 操纵检查判据（"每因子配操纵检查；失败条件
    #   剔除重造"）。修复：只读 passed 为硬判据；baseline_vs_full_differs 仅作
    #   详情披露（写入 details），不作为通过依据。若 passed 键缺失 → 如实记为
    #   不通过并披露（纪律 #2，不静默视为通过）。
    _manip_passed = manip.get("passed")
    manip_ok = bool(_manip_passed)
    if _manip_passed is None:
        log.warning("manipulation_check.passed 缺失（effects 无操纵检验字段）→ "
                    "操纵检验按不通过披露（G1 判据不静默跳过）")

    criteria = {
        "direction_consistent": bool(direction_consistent),
        # v6.5.26-fix（D2）：G1(a) 双模型方向一致（硬判据）
        "n_direction_consistent": n_dir_consistent,
        "effect_ge_10pp": bool(effect_ge_10pp),
        "ci_excludes_zero": bool(ci_excludes_zero),
        "dual_judge_available": bool(dual_available),
        "dual_dispute_ok": dispute_ok,
        # v6.7-r5-fix（终审 CRIT-1/Major C）：决策口径与数据完整性
        "decision_caliber": decision_caliber,
        "data_complete": bool(data_complete),
        # v6.5: heterogeneity 从硬判据降为 soft_evidence（§0 判断 3 / §5.3 G1-e）
        "heterogeneity_soft": hetero_soft,
        "heterogeneity_explainable": bool(heterogeneity_explainable),
        "manipulation_check": bool(manip_ok),
        "details": {
            "directions": dirs,
            "effect_pp": eff_pp,
            "ci": ci,
            # primary（judge_big）仅敏感性披露，不参与 C2/C3 决策
            "primary_effect_pp": primary_v.get("effect_pp"),
            "primary_ci": primary_v.get("ci"),
            "both_models": both,
            # v6.6.1-fix（问题 53）：N_x_A_s 为嵌套结构（primary/dual_judge/
            # majority），原读顶层 n_x_as.get("direction") 恒 None（与 L132-134
            # v6.5.13 修复同源）；改为读 primary 口径方向。
            "n_x_As_direction": n_x_as_direction,
            "scorers_n": n_main.get("scorers_n"),
            "dual_dispute": dispute_note,
            "dual_coverage": dual_coverage,
            # v6.5.23-fix（问题 88）：baseline_vs_full_differs 仅作详情披露，
            # 不作为通过判据（操纵检验以 passed 为唯一硬判据）
            "manip_baseline_vs_full_differs": manip.get("baseline_vs_full_differs"),
        },
    }
    # v6.5 §5.3 G1: (a)(b)(c)(d) 为硬判据；(e) 模型间异质性仅作稳健性佐证（soft_evidence）
    # 硬判据集合：三口径方向一致 / (a)双模型方向一致 / 决策口径效应量≥10pp /
    #   双judge可用 / 操纵检验通过 / 数据完整（v6.7-r5-fix 终审 Major C）
    # v6.5.26-fix（D2）：(a) 纳入硬判据；n_dir_consistent=None（缺失）→ 不通过披露
    passed = all([criteria["direction_consistent"], criteria["effect_ge_10pp"],
                  criteria["dual_judge_available"], criteria["manipulation_check"],
                  criteria["n_direction_consistent"] is True])
    # C4 争议率判据（v6.4 协议 §6）：可核验时强制；缺失时降级可解释
    if dispute_ok is not None:
        passed = passed and dispute_ok
    # CI 判据为必要条件；ci=None（数据不足无法核验）时按不满足处理
    # （v6.5.28-fix M3：ci_excludes_zero 此时为 False → fail-closed，与 (a)
    # 的 None→fail-closed 对称；§5.3 (c) "bootstrap CI 不含 0"为硬判据）。
    passed = passed and criteria["ci_excludes_zero"]
    # 数据完整性（终审 Major C）：不完整（含缺口）→ 不通过
    passed = passed and criteria["data_complete"]
    # v6.7-r5-fix（终审 Major B，部署核验补强）：非 v6.5 输入不得静默沿用——
    # 披露之外，判定结果也强制不通过（不得用旧口径数据授权进入 P1-FULL/P0-C）。
    passed = passed and (eff.get("version") == "v6.5")

    out = {
        "gate": "G1",
        "passed": passed,
        "criteria": criteria,
        "evidence": [str(eff_path)],
        "note": (f"通过（决策口径={decision_caliber}）→ 进入 P1-FULL ∥ P0-C；"
                 f"不通过 → 转探索性分析（探索性分支仍产出报告）"
                 if passed else
                 "不通过：决策口径(dual_judge)效应不足/CI含0/三口径不一致/操纵检验失败/"
                 "数据不完整 → 转探索性分析"),
    }
    gates = root / "gates"
    gates.mkdir(parents=True, exist_ok=True)
    (gates / "G1.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    log.info("G1 判定: passed=%s criteria=%s", passed, criteria)
    jlog.event(stage=STAGE, event="decided", passed=passed, criteria=criteria)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
