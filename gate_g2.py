# -*- coding: utf-8 -*-
"""
gate_g2.py — 闸门 G2（v6.4）

输入: results/msrf_evaluation.json（stage_p2_msrf.py 输出，gates/G2_input.json 副本）
判据（v6.4 提示词 §8 / STAGE_CONTRACTS §G2）：
  C1. 固定 FPR=5% 下 TPR 较最佳单分支提升 ≥3pp（min_gain_pp）
  C2. AUPRC/benign FPR/ECE 不劣化（AUC 增益 > 0；ECE ≤ 0.15）
  C3. ≥3 分支有独立贡献（消融：任一分支移除后 AUC 下降）
  C4. 三口径一致（P1-PILOT/FULL 的 N_main 三口径方向一致——读 p1_full_stats.json）
  C5. 5 种子结果稳定（标准差不改变显著性结论：均值-2σ 仍 ≥ 增益阈值）

输出: gates/G2.json
退出: 0=通过（KBS 绿灯）/ 1=不通过（机制主导版改写建议）/ 3=致命
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np  # v6.6.1-fix（问题 52）：C5 种子级 TPR 增益均值/标准差

from common_utils import load_config, setup_logging, JsonlLogger

STAGE = "g2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["workdir"]).expanduser()
    log = setup_logging(str(root / "logs" / f"{STAGE}.log"), STAGE)
    jlog = JsonlLogger(str(root / "logs" / f"{STAGE}.jsonl"))

    # 输入 1：MSRF 评估
    g2_input = root / "gates" / "G2_input.json"
    if not g2_input.exists():
        g2_input = root / "results" / "msrf_evaluation.json"
    if not g2_input.exists():
        log.error("G2 输入缺失: %s → 致命 3", g2_input)
        return 3
    ev = json.loads(g2_input.read_text(encoding="utf-8"))
    log.info("G2 输入: %s", g2_input)

    # 输入 2：P1-FULL 三口径（结论稳健性）
    p1f_stats = root / "results" / "p1_full_stats.json"
    p1f = {}
    if p1f_stats.exists():
        p1f = json.loads(p1f_stats.read_text(encoding="utf-8"))
    crosslingual = p1f.get("crosslingual", {}) or {}

    # ---- 种子级 TPR 增益（C1 显著性 + C5 稳定性共用；提前计算）----
    # v6.5.26-fix（G2 C1/C5）：协议 §8.10 "TPR 提升显著且 ≥3pp" + "5 种子稳定"。
    # 原实现 C1 纯点估计（无显著性检验）、C5 单列；现种子级增益统一计算。
    seed_gains = []
    for sr_ in ev.get("seed_detail", []) or []:
        bl = sr_.get("best_lr")
        fus = (sr_.get("fusion") or {}).get(bl) if bl is not None else None
        sb = sr_.get("single_branch") or {}
        if fus is None or not sb:
            continue
        fus_tpr = fus.get("tpr_at_fpr")
        best_s_tpr = max(
            (v.get("tpr_at_fpr") for v in sb.values()
             if v.get("tpr_at_fpr") is not None), default=None)
        if fus_tpr is not None and best_s_tpr is not None:
            seed_gains.append((fus_tpr - best_s_tpr) * 100)  # pp
    if len(seed_gains) >= 2:
        _g_mean = float(np.mean(seed_gains))
        _g_std = float(np.std(seed_gains))
        _gain_lb = _g_mean - 2 * _g_std  # 2σ 下界
        _signif_ok = _gain_lb > 0
        _signif_note = (f"种子级 TPR 增益 {len(seed_gains)} 个: "
                        f"均值 {_g_mean:.2f}pp ± {_g_std:.2f}pp, "
                        f"2σ 下界 {_gain_lb:.2f}pp")
        # v6.5.29-fix（铁律版阶段1，KBS 统计检验补全）：配对 t-test（单样本
        # t-test 检验增益均值 >0）——补充 2σ 下界之外的正式统计检验，KBS 审稿
        # 人可复核。样本 5 个种子（5 种子稳定性协议）。
        try:
            from scipy import stats as _sts  # noqa: PLC0415
            _tval, _pval = _sts.ttest_1samp(seed_gains, 0.0)
            _signif_note += (f" | t-test: t={_tval:.2f} p={_pval:.4f} "
                             f"({'显著(p<0.05)' if _pval < 0.05 else '不显著'})")
        except Exception:  # noqa: BLE001
            pass
    elif len(seed_gains) == 1:
        _g_mean = float(seed_gains[0]); _g_std = None
        _gain_lb = None; _signif_ok = None
        _signif_note = f"仅 1 个种子可算 TPR 增益（{_g_mean:.2f}pp）→ 显著性无法核验"
    else:
        _g_mean = None; _g_std = None; _gain_lb = None; _signif_ok = None
        _signif_note = "seed_detail 无可用 TPR 数据 → 显著性无法核验"

    # C1: TPR 增益 ≥ 3pp 且显著（种子级增益 2σ 下界 > 0；不可核验 → fail-closed 披露）
    g2 = ev.get("g2", {}) or {}
    tpr_gain_pp = g2.get("tpr_gain_pp")
    min_gain = g2.get("min_gain_pp", cfg.get("p2", {}).get("eval", {}).get(
        "min_tpr_gain_pp", 3))
    c1 = (tpr_gain_pp is not None and tpr_gain_pp >= min_gain
          and _signif_ok is True)
    if _signif_ok is None and tpr_gain_pp is not None:
        log.warning("G2 C1: 种子级增益显著性无法核验（%s）→ 按不通过披露",
                    _signif_note)

    # C2: AUPRC 不劣化（融合 ap ≥ 最佳单分支 ap - 容差）+ benign FPR ≤ 5%
    #      + ECE 不劣化（融合 ece ≤ 最佳单分支 ece + 容差）
    # v6.5.26-fix（审查发现 2026-08-08）：
    #   - 原 C2 用 ROC-AUC 增益替代 AUPRC（协议 §8.10 明列 "AUPRC 不劣化"）
    #   - benign FPR 原用绝对 0.10（2× 协议 §3 防御假设 ≤5%）且 None 视为通过
    #   - ECE 原用绝对 0.15（协议无此阈值），"不劣化"相对语义未实现
    fb = ev.get("fusion_best") or {}
    sbv = ev.get("single_branch") or {}
    auc_gain = g2.get("auc_gain")
    ece = fb.get("ece_mean")
    ap = fb.get("ap_mean")
    benign_fpr = fb.get("benign_fpr")
    best_sb_ap = max(
        (v.get("ap_mean") for v in sbv.values()
         if v.get("ap_mean") is not None), default=None)
    best_sb_ece = min(
        (v.get("ece_mean") for v in sbv.values()
         if v.get("ece_mean") is not None), default=None)
    _MAX_BFPR = cfg.get("p2", {}).get("eval", {}).get(
        "max_benign_fpr", 0.05)  # 协议 §3：benign FPR ≤5%
    ap_ok = (ap is not None and best_sb_ap is not None
             and ap >= best_sb_ap - 0.005)
    # v6.5.28-fix：ECE 缺失时不得静默通过（§8.10 "ECE 不劣化"为硬判据，与
    # benign FPR 缺失 fail-closed 对称）——无 ECE 数据 → ece_ok=False + 警告披露。
    ece_ok = False
    if ece is None:
        log.warning("G2 C2: ECE 缺失（未测量）→ 按不通过披露（fail-closed）")
    elif best_sb_ece is not None:
        ece_ok = ece <= best_sb_ece + 0.01
    else:
        ece_ok = ece <= 0.15  # 兜底绝对阈值（如实披露）
        log.warning("G2 C2: 单分支 ECE 缺失，融合 ECE %.4f 用绝对阈值 0.15 兜底", ece)
    # benign FPR：绝对 ≤5%（协议防御假设）；缺失 → fail-closed 披露
    benign_ok = benign_fpr is not None and benign_fpr <= _MAX_BFPR
    if benign_fpr is None:
        log.warning("G2 C2: benign FPR 缺失（无良性样本评估）→ 按不通过披露")
    c2 = (ap_ok and benign_ok and ece_ok)

    # C3: ≥3 分支独立贡献（消融任一分支 AUC 下降）
    abl = ev.get("ablation", {}) or {}
    fusion_auc = (ev.get("fusion_best") or {}).get("auc_mean")
    contrib = 0
    for b, v in abl.items():
        if v.get("auc_mean") is not None and fusion_auc is not None \
                and v["auc_mean"] < fusion_auc - 0.001:
            contrib += 1
    c3 = contrib >= 3

    # C4: 三口径一致（P1-FULL robust_conclusion 或 P1-PILOT 方向一致）
    # v6.7-r4-fix 2026-08-07：协议 §8 G2 判据明确含"三口径一致"。
    # 原实现把 c4 标为"软性"排除在 passed 之外——与 P1-1 修复前的 G1 同类缺陷
    # （正式判据被静默降级）。修复：
    #   - P1-FULL 已完成（p1_full_stats.json 存在）→ c4 为硬性判据
    #   - P1-FULL 未完成（数据缺失）→ 降级披露（G2 note 标注"三口径未核验"），
    #     不判死（对齐 G1"不可算则降级"逻辑，防止数据缺失误杀）
    three_ok = crosslingual.get("three_way_consistent_up", False)
    p1f_available = p1f_stats.exists() and bool(p1f)
    if p1f_available:
        c4 = bool(three_ok)
        c4_note = (f"P1-FULL 三口径一致={three_ok}"
                   if p1f_available else None)
    else:
        # v6.5.18-fix（问题 56）：原 `c4=True` 将"数据缺失降级"直接视为通过
        # ——§8 G2 判据"三口径一致"是硬性判据，数据缺失时必须如实披露降级，
        # 而非静默通过（与 gate_g1 的 dispute_ok=None 降级披露风格对齐）。
        # 改为 c4=None（不否决，但显式披露"三口径未核验"，且写入 criteria）。
        c4 = None
        c4_note = "P1-FULL 未完成（p1_full_stats.json 缺失）→ 三口径一致判据降级披露（非否决，但 G2 结论须标注"

    # C5: 5 种子稳定（均值 - 2σ 仍 ≥ 增益阈值）
    # v6.6.1-fix 2026-08-08（问题 52）：原实现检查 `(auc_mean - 2*auc_std) > 0`
    # ——对象错（AUC 而非 TPR 增益）、阈值错（0 而非 min_gain_pp），与注释
    # 宣称的"均值-2σ 仍 ≥ 增益阈值"及协议 §8.10"5 种子稳定"语义不符。
    # 修复：从 seed_detail 逐种子计算 TPR 增益（融合 best_lr TPR − 最佳单分支
    # TPR），求均值-2σ 与 min_gain_pp 比较（对齐 §8.6 均值±标准差）。
    # 数据不足时如实降级（c5=None 披露），不静默通过。
    # C5: 5 种子稳定（均值 - 2σ 仍 ≥ 增益阈值）——使用上方提前计算的种子级增益
    if _g_mean is not None and _g_std is not None:
        c5 = (_gain_lb >= min_gain)
        c5_note = (f"种子级 TPR 增益 {len(seed_gains)} 个: 均值 "
                   f"{_g_mean:.2f}pp ± {_g_std:.2f}pp（2σ 下界 "
                   f"{_gain_lb:.2f}pp ≥ 阈值 {min_gain}pp）")
    elif _g_mean is not None:
        c5 = None  # 单种子无法判稳定 → 如实降级披露，不判死
        c5_note = (f"仅 1 个种子可算 TPR 增益（{_g_mean:.2f}pp）→ "
                   "种子稳定性无法核验，降级披露（非否决项）")
    else:
        c5 = None
        c5_note = "seed_detail 无可用 TPR 数据 → 种子稳定性无法核验，降级披露（非否决项）"

    fb = ev.get("fusion_best") or {}
    auc_mean = fb.get("auc_mean")
    auc_std = fb.get("auc_std") or 0
    criteria = {
        "tpr_gain_ge_3pp": bool(c1),
        "ap_not_worse_benign_fpr_ok_ece_ok": bool(c2),
        "benign_fpr_ok": bool(benign_ok),
        "three_plus_branches_contribute": bool(c3),
        "three_way_consistent": (bool(c4) if c4 is not None else None),
        "seed_stable": bool(c5) if c5 is not None else None,
        "details": {
            "tpr_gain_pp": tpr_gain_pp, "min_gain_pp": min_gain,
            "c1_significant": (_signif_ok if _signif_ok is not None else None),
            "auc_gain": auc_gain, "ece_mean": ece,
            "ap_mean": ap, "best_single_ap_mean": best_sb_ap,
            "ap_not_worse": bool(ap_ok),
            "benign_fpr": benign_fpr,
            "max_benign_fpr": _MAX_BFPR,
            "contributing_branches": contrib,
            "fusion_auc_mean": auc_mean, "fusion_auc_std": auc_std,
            "seed_tpr_gain_mean_pp": (round(_g_mean, 2)
                                      if _g_mean is not None else None),
            "crosslingual": crosslingual,
        },
    }
    # v6.7-r4-fix: 纳入 c4（协议 §8 判据）
    # v6.6.1-fix（问题 52）：c5=None（无法核验）→ 降级为不否决并披露，
    # 与 C4 的"数据缺失降级"逻辑对齐；仅 c5=False（核验了但不足）才否决。
    # v6.5.18-fix（问题 56）：c4=None（三口径未核验）同样降级为不否决，
    # 但必须披露（criteria.details.three_way_note）；仅 c4=False（核验了
    # 但不一致）才否决。c4 为 None 时 passed 不因此失败，报告必须看到降级。
    # v6.5.29-fix（第十轮审查 🟡，§8.2）：训练规模 ≥4000 为协议强制——不足时
    # G2 判不通过（fail-closed），避免"规模不足仍 mark done"静默通过。
    _n_train_ok = ev.get("n_train_ok")
    _n_train = ev.get("n_train")
    _min_train = ev.get("min_train_req", 4000)
    if _n_train_ok is None:
        # 旧 G2_input 无规模字段 → 依据 n_train 与 min_train_req 推算
        _n_train_ok = (_n_train is not None and _n_train >= _min_train)
    c6_ok = bool(_n_train_ok)
    if not c6_ok:
        log.warning("G2 训练规模 %s < 要求 %s（§8.2 规模≥4000）→ 判不通过",
                    _n_train, _min_train)
    # v6.5.29-fix（自主裁决 #5，§8.10）：C4/C5 改 fail-closed——预注册硬判据在
    # 数据缺失（None）时按"未核验"判不通过（原 None 放行 → 审稿人见"三口径未核验
    # 也通过 G2"必质疑）。核验了但通过（True）才放行；None/False 均否决。
    c4_blocks = (c4 is not True)   # fail-closed：None/False 均否决
    c5_blocks = (c5 is not True)   # fail-closed：None/False 均否决
    passed = c1 and c2 and c3 and not c4_blocks and not c5_blocks and c6_ok
    # 记录 c4/c5 核验状态供报告使用
    criteria["details"]["three_way_note"] = c4_note
    criteria["details"]["seed_stable_note"] = c5_note
    criteria["details"]["n_train"] = _n_train
    criteria["details"]["min_train_req"] = _min_train
    criteria["details"]["n_train_ok"] = c6_ok
    criteria["n_train_ge_4000"] = c6_ok

    out = {
        "gate": "G2",
        "passed": passed,
        "criteria": criteria,
        "evidence": [str(g2_input), str(p1f_stats) if p1f_stats.exists() else None],
        "verdict": "kbs_green_light" if passed else "mechanism_dominant",
        "note": ("通过 → KBS 绿灯，进入 P2-B → F → R；"
                 "不通过 → 论文收缩为机制主导（RQ1 为主），防御降级为应用验证，"
                 "仍投 KBS 但需重写贡献排序（报告中给出改写建议）"),
    }
    out["evidence"] = [e for e in out["evidence"] if e]
    gates = root / "gates"
    gates.mkdir(parents=True, exist_ok=True)
    (gates / "G2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    log.info("G2 判定: passed=%s criteria=%s", passed, criteria)
    jlog.event(stage=STAGE, event="decided", passed=passed, criteria=criteria)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
