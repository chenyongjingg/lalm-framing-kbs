# -*- coding: utf-8 -*-
"""
stage_f_figures.py — 阶段 F：出版级图表（v6.2）

依据 v6.2 提示词 / STAGE_CONTRACTS §F / config.figures。

图表（config.figures.required）：
  1. overview_diagram — 研究设计总览
  2. factorial_forest — 析因森林图（P1-PILOT/FULL 效应量 + CI）
  3. pcsd_heatmap — PCSD 配对分歧热力图（v6.4）
  4. msrf_roc_pr — MSRF ROC/PR 曲线（5 种子）
  5. ablation_bars — 消融柱状图
  6. adaptive_decay — 自适应攻击衰减图
  7. hyperparam_curves — 超参敏感性曲线

要求：dpi=300、PDF+PNG、色盲友好、字体嵌入。
输出：figures/*.pdf/png + report/figure_index.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common_utils import Checkpoint, JsonlLogger, load_config, setup_logging

STAGE = "f"


def _setup_matplotlib(cfg: dict, log):
    """matplotlib 配置：色盲友好 + 字体嵌入。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # v6.5.28-fix（F-CJK，审查发现 2026-08-09）：图含中文标签（Fig.1 框标签、
    # Fig.2 ✅/❌ 徽章、Fig.5-7 中文标题），DejaVu Sans 无 CJK 字形 → 出版级图
    # 出现豆腐块（§11 出版级图表不达标）。检测系统 CJK 字体并加入回退链。
    import matplotlib.font_manager as _fm  # noqa: PLC0415
    _cjk = sorted({f.name for f in _fm.fontManager.ttflist
                   if any(k in f.name for k in (
                       "Noto Sans CJK", "WenQuanYi", "SimHei",
                       "Microsoft YaHei", "Source Han",
                       "Droid Sans Fallback", "Noto Sans SC"))})
    # 2026-08-24（F-CJK2）：末尾追加 Symbola 作为 ✅(U+2705) 单色字形回退
    _fam = (_cjk + ["DejaVu Sans"] if _cjk else ["DejaVu Sans"]) + ["Symbola"]
    if not _cjk:
        import logging as _lg  # noqa: PLC0415
        _lg.getLogger("figures").warning(
            "未检测到 CJK 字体（Noto Sans CJK/WenQuanYi 等）→ 图中文字符可能"
            "渲染为豆腐块；建议安装 fonts-noto-cjk")
    plt.rcParams.update({
        "figure.dpi": cfg["figures"].get("dpi", 300),
        "savefig.dpi": cfg["figures"].get("dpi", 300),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": _fam,
        "axes.prop_cycle": plt.cycler(color=["#0072B2", "#D55E00", "#009E73",
                                             "#CC79A7", "#F0E442", "#56B4E9"]),
    })
    return plt


def fig_overview(plt, out_dir: Path):
    """研究设计总览图（v6.5.17 修复：v6.5 评分器口径 + 框距/标题防溢出）。

    问题 32：原"6评分器"为 v6.4 旧口径，v6.5 为 4 评分器 + 1 异构交叉验证；
    问题 36：原框宽 1.35 + 字号 7.5 时多行文字溢出框边界，标题贴近 ylim 顶边。
    """
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    W = 1.55  # 框宽（v6.5.17：1.35→1.55，容纳多行标签）
    stages = [
        ("L 新颖性核验", 0.3, 4.6, "#0072B2"),
        ("D 数据构建\n(中300+英300\nAdv200+良300)", 2.2, 4.6, "#009E73"),
        ("P0 评分体系\n(4评分器+异构交叉验证)", 4.1, 4.6, "#D55E00"),
        ("P1-PILOT\n析因预实验", 0.3, 2.6, "#56B4E9"),
        ("G1 闸门", 2.2, 2.6, "#CC79A7"),
        ("P1-FULL ∥ P0-C\n跨语言+LALM矩阵", 4.1, 2.6, "#56B4E9"),
        ("P2 MSRF\n5种子融合防御", 6.6, 2.6, "#D55E00"),
        ("P2-C 自适应攻击", 8.5, 2.6, "#CC79A7"),
        ("G2 闸门", 6.6, 0.6, "#F0E442"),
        ("P2-B 降级+基线\n→ F → R", 8.5, 0.6, "#009E73"),
    ]
    for label, x, y, c in stages:
        ax.add_patch(plt.Rectangle((x, y - 0.35), W, 0.7,
                                   facecolor=c, edgecolor="black",
                                   alpha=0.8))
        ax.text(x + W / 2, y, label, ha="center", va="center",
                fontsize=7, color="white", wrap=True)
    # 上排：L → D → P0
    ax.annotate("", xy=(2.2, 4.6), xytext=(1.85, 4.6),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.annotate("", xy=(4.1, 4.6), xytext=(3.75, 4.6),
                arrowprops=dict(arrowstyle="->", color="black"))
    # 中排：P1-PILOT → G1 → P1-FULL → P2 → P2-C
    ax.annotate("", xy=(2.2, 2.6), xytext=(1.85, 2.6),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.annotate("", xy=(4.1, 2.6), xytext=(3.75, 2.6),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.annotate("", xy=(6.6, 2.6), xytext=(5.65, 2.6),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.annotate("", xy=(8.5, 2.6), xytext=(8.15, 2.6),
                arrowprops=dict(arrowstyle="->", color="black"))
    # P0 → P2：归因→检测逻辑链（弧线绕开 P1-FULL 框）
    ax.annotate("", xy=(6.775, 2.95), xytext=(4.875, 4.25),
                arrowprops=dict(arrowstyle="->", color="black",
                                connectionstyle="arc3,rad=0.12"))
    # P2 → G2（垂直下行）→ P2-B（下排）
    ax.annotate("", xy=(6.775, 0.95), xytext=(6.775, 2.25),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.annotate("", xy=(8.5, 0.6), xytext=(8.15, 0.6),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.text(5.25, 5.7, "LALM Framing v6.5 Pipeline DAG",
            ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_factorial_forest(plt, out_dir: Path, effects: dict):
    """析因森林图：P1-PILOT/FULL N_main + N×A_s 效应量（三口径 + CI）。

    v6.5.17 修复（问题 38）：§11 ②要求"析因效应森林图"含主效应与交互效应，
    并标注 G1 (a) 双模型一致判据。原实现只画 N_main、且不标注模型一致性。
    现补充：①N_x_A_s 交互效应条目（若存在）；②"双模型一致"徽章（读
    effects[].both_models.consistent）。
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    labels = []
    effs = []
    cis = []
    metrics = []
    # 语义色映射（与正文 TikZ 架构图一致：Narrative=橙 #D55E00 系）：
    # 森林图所有条目均为叙事因子（N_main / N×A_s），故全部落在 narrative 族，
    # 三种口径（primary / dual_judge / majority）用形状 + 深浅双编码区分。
    color_map = {"primary": "#D55E00", "dual_judge": "#E8814A", "majority": "#F2B073"}
    marker_map = {"primary": "o", "dual_judge": "s", "majority": "^"}
    # 从 P1-PILOT/FULL 结果提取（N_main + N_x_A_s）
    for src_name, src_key in [("P1-PILOT", "N_main"), ("P1-FULL", "N_main"),
                              ("P1-PILOT", "N_x_A_s"), ("P1-FULL", "N_x_A_s")]:
        src = effects.get(src_name, {}) if src_name in effects else {}
        if not src:
            continue
        v = src.get(src_key, {})
        for metric in ["primary", "dual_judge", "majority"]:
            mv = v.get(metric)
            if isinstance(mv, dict) and mv.get("effect_pp") is not None:
                tag = "N×A_s" if src_key == "N_x_A_s" else "N_main"
                labels.append(f"{src_name} {tag} {metric}")
                effs.append(mv["effect_pp"])
                # v6.5.29-fix（第十轮审查 🟡，§11②）：N×A_s 条目优先用
                # ci_interaction（stage_p1_pilot 写入的交互效应 CI）——原用
                # `ci`（=ci_text，文本条件 N 效应 CI）画 N×A_s 误差棒，CI 错配。
                ci = mv.get("ci_interaction") if src_key == "N_x_A_s" \
                    else mv.get("ci")
                if ci and len(ci) == 2:
                    cis.append((ci[0] - mv["effect_pp"],
                                ci[1] - mv["effect_pp"]))
                else:
                    cis.append((None, None))
                metrics.append(metric)
    if not effs:
        ax.text(0.5, 0.5, "P1 effects missing", ha="center", fontsize=12)
    else:
        y_pos = np.arange(len(effs))
        for i, (e, (lo, hi)) in enumerate(zip(effs, cis)):
            # v6.7-r4-fix 2026-08-07：errorbar 的 xerr 必须是标量/数组，
            # 原 [[abs(lo)],[abs(hi)]] 是嵌套列表（与标量 e 不匹配，会报错）。
            # 对非对称 CI 使用 [[lo_err], [hi_err]] 的 2×N 形式（None→0）。
            lo_err = abs(lo) if lo is not None else 0.0
            hi_err = abs(hi) if hi is not None else 0.0
            ax.errorbar(e, i, xerr=[[lo_err], [hi_err]],
                        fmt=marker_map[metrics[i]], ms=5,
                        color=color_map[metrics[i]], ecolor="gray", capsize=3)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.set_yticks(y_pos)
        # 因子名用语义色（Narrative=橙，与 Figure 1 一致）
        ax.set_yticklabels(labels, fontsize=8, color="#D55E00")
        ax.set_xlabel("Effect size (pp)")
        # v6.6.0-fix: axvspan 前先确定 xlim（原 ax.get_xlim() 在未绘图时可能
        # 返回自动边界导致绿带位置错误）
        _xmin = min([e for e in effs] + [0]) - 5
        _xmax = max([e for e in effs] + [0]) + 5
        ax.set_xlim(_xmin, _xmax)
        ax.axvspan(max(10, _xmin), _xmax, alpha=0.1, color="green")
        # 问题 38：G1 (a) 双模型一致标注（读 both_models.consistent）
        cons = []
        for src_name in ["P1-PILOT", "P1-FULL"]:
            src = effects.get(src_name, {})
            bm = src.get("both_models") if isinstance(src, dict) else None
            if isinstance(bm, dict) and "consistent" in bm:
                cons.append((src_name, bm["consistent"]))
        if cons:
            txt = "Both-model agreement: " + " / ".join(
                f"{n}={'✓' if c else '✗'}" for n, c in cons)
            ax.text(0.02, 0.02, txt, transform=ax.transAxes,
                    fontsize=8, color="dimgray", va="bottom")
        # 三种口径图例（形状 + 颜色双编码，色盲友好）
        from matplotlib.lines import Line2D  # noqa: PLC0415
        lg = [Line2D([0], [0], marker=marker_map[m], color=color_map[m],
                     linestyle="", markersize=6, label=m.replace("_", " "))
              for m in ["primary", "dual_judge", "majority"]]
        ax.legend(handles=lg, fontsize=7.5, loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def fig_cmsc_heatmap(plt, out_dir: Path, cmsc_csv: Path):
    """PCSD 配对分歧热力图（v6.4：原 cmsc_heatmap 改名）。"""
    import pandas as pd
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not cmsc_csv.exists():
        ax.text(0.5, 0.5, "PCSD 数据缺失", ha="center", fontsize=12)
    else:
        df = pd.read_csv(cmsc_csv)
        if "model" in df.columns and "condition" in df.columns \
                and "cmsc" in df.columns:
            pivot = df.pivot_table(index="model", columns="condition",
                                   values="cmsc", aggfunc="mean")
            im = ax.imshow(pivot, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    v = pivot.iloc[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=8)
            fig.colorbar(im, ax=ax, label="PCSD agreement rate")
        else:
            ax.text(0.5, 0.5, "PCSD columns missing", ha="center", fontsize=12)
    fig.tight_layout()
    return fig


def fig_msrf_roc(plt, out_dir: Path, msrf_json: Path):
    """MSRF ROC/PR 曲线（5 种子均值 ± 包络带，v6.5 真实曲线）。

    v6.5 修正：原实现用 fpr^(1/auc) 解析式模拟曲线（非真实数据），
    出版图不可用。现读取 P2 阶段保存的真实 (fpr, tpr) / (precision, recall)
    曲线点，绘制均值曲线 + 5 种子包络带。
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    if not msrf_json.exists():
        for ax in axes:
            ax.text(0.5, 0.5, "MSRF 数据缺失", ha="center", fontsize=12)
        fig.tight_layout()
        return fig
    ev = json.loads(msrf_json.read_text(encoding="utf-8"))
    seeds_roc = []   # 每 seed: (fpr[], tpr[])
    seeds_pr = []    # 每 seed: (recall[], precision[])
    for sd in ev.get("seed_detail", [])[:5]:
        rp = sd.get("roc_pr")
        if not rp:
            continue
        fpr_c, tpr_c = rp.get("roc_fpr", []), rp.get("roc_tpr", [])
        prec_c, rec_c = rp.get("pr_precision", []), rp.get("pr_recall", [])
        if fpr_c and tpr_c:
            seeds_roc.append((np.array(fpr_c), np.array(tpr_c)))
        if prec_c and rec_c:
            seeds_pr.append((np.array(rec_c), np.array(prec_c)))
    if not seeds_roc:
        for ax in axes:
            ax.text(0.5, 0.5, "MSRF 曲线点缺失（需 P2 跑真实数据）",
                    ha="center", fontsize=12)
        fig.tight_layout()
        return fig
    # 统一插值到公共网格（均值曲线 + 10-90 百分位包络带）
    grid = np.linspace(0, 1, 100)
    # ROC: x=fpr, y=tpr
    roc_tprs = []
    for fpr_c, tpr_c in seeds_roc:
        roc_tprs.append(np.interp(grid, fpr_c, tpr_c))
    roc_mean = np.mean(roc_tprs, axis=0)
    roc_lo = np.percentile(roc_tprs, 10, axis=0)
    roc_hi = np.percentile(roc_tprs, 90, axis=0)
    ax = axes[0]
    ax.fill_between(grid, roc_lo, roc_hi, alpha=0.15, color="#0072B2")
    ax.plot(grid, roc_mean, color="#0072B2", lw=2,
            label=f"MSRF mean (n={len(seeds_roc)})")
    for fpr_c, tpr_c in seeds_roc:
        ax.plot(fpr_c, tpr_c, alpha=0.25, lw=0.8, color="#85B7EB")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (real curves)")
    ax.legend(fontsize=8, loc="lower right")
    # PR: x=recall, y=precision（v6.5 修正坐标轴）
    # v6.5.9-fix：sklearn precision_recall_curve 的 recall 从 1→0 递减，
    # np.interp 要求 xp 单调递增 → 原代码直接插值产生错位曲线。
    # 现逆序为 0→1 后再插值（等价曲线，无信息损失）。
    pr_precs = []
    for rec_c, prec_c in seeds_pr:
        if len(rec_c) > 1 and rec_c[0] > rec_c[-1]:
            pr_precs.append(np.interp(grid, rec_c[::-1], prec_c[::-1]))
        else:
            pr_precs.append(np.interp(grid, rec_c, prec_c))
    pr_mean = np.mean(pr_precs, axis=0)
    pr_lo = np.percentile(pr_precs, 10, axis=0)
    pr_hi = np.percentile(pr_precs, 90, axis=0)
    ax = axes[1]
    ax.fill_between(grid, pr_lo, pr_hi, alpha=0.15, color="#D55E00")
    ax.plot(grid, pr_mean, color="#D55E00", lw=2,
            label=f"MSRF mean (n={len(seeds_pr)})")
    for rec_c, prec_c in seeds_pr:
        ax.plot(rec_c, prec_c, alpha=0.25, lw=0.8, color="#F0997B")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curve (real)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def fig_ablation_bars(plt, out_dir: Path, msrf_json: Path):
    """消融柱状图（v6.5.17 修复：问题 35——补 5 种子误差棒）。

    §11 要求"互补性消融（5 种子误差棒）"。原实现只画 auc_mean 均值、
    caption 却宣称"5 种子误差棒"——图文不符。现从 seed_detail[].ablation
    聚合每个去一分支的 5 种子 AUC，画 mean±std 误差棒；Fusion 用
    seed_detail[].fusion[best_lr].auc 聚合。
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not msrf_json.exists():
        ax.text(0.5, 0.5, "MSRF data missing", ha="center", fontsize=12)
    else:
        ev = json.loads(msrf_json.read_text(encoding="utf-8"))
        seeds = ev.get("seed_detail", [])
        # 各分支 5 种子 AUC 列表
        abl_aucs = {}
        for sr_ in seeds:
            abl = sr_.get("ablation", {})
            for b, v in abl.items():
                a = v.get("auc") if isinstance(v, dict) else None
                if a is not None:
                    abl_aucs.setdefault(b, []).append(a)
        # Fusion 5 种子 AUC（每种子 best_lr 下的 auc）
        fus_aucs = []
        for sr_ in seeds:
            bl = sr_.get("best_lr")
            if bl and sr_.get("fusion", {}).get(bl, {}).get("auc") is not None:
                fus_aucs.append(sr_["fusion"][bl]["auc"])
        names = list(abl_aucs.keys())
        means = [float(np.mean(abl_aucs[b])) for b in names]
        stds = [float(np.std(abl_aucs[b])) for b in names]
        if fus_aucs:
            names.append("Fusion")
            means.append(float(np.mean(fus_aucs)))
            stds.append(float(np.std(fus_aucs)))
        if not names:
            # v6.5.28-fix（第三轮审查）：空数据（无 ablation/fusion AUC）时
            # 落"数据缺失"占位而非崩溃（原 ax.bar([]) 后 bars[-1] IndexError，
            # F 阶段反复重试）。
            ax.text(0.5, 0.5, "Missing ablation / fusion AUC",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("MSRF ablation: leave-one-branch-out vs fusion (5 seeds)")
            return
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds, capsize=3,
                      color=["#0072B2"] * (len(names) - 1) + ["#D55E00"])
        bars[-1].set_color("#D55E00")
        ax.set_ylabel("AUC (mean±std, 5 seeds)")
        ax.set_title("MSRF ablation: leave-one-branch-out vs fusion (5 seeds)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, fontsize=8)
        for xi, m, s in zip(x, means, stds):
            ax.text(xi, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=8)
        ax.set_ylim(0, min(1.0, max(means) + max(stds) + 0.12))
    fig.tight_layout()
    return fig


def fig_adaptive_decay(plt, out_dir: Path, p2c_csv: Path):
    """自适应攻击衰减图。"""
    import pandas as pd
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not p2c_csv.exists():
        ax.text(0.5, 0.5, "P2-C data missing", ha="center", fontsize=12)
    else:
        df = pd.read_csv(p2c_csv)
        # v6.5：新 P2-C 输出列 msrf_detect（旧版 detected 兼容）
        det_col = "msrf_detect" if "msrf_detect" in df.columns else "detected"
        det = df.groupby("attack")[det_col].mean() * 100

        def _wrap(name: str) -> str:
            """Wrap overly long attack names onto two lines for readability."""
            if len(name) > 22:
                mid = name.rfind("_", 0, len(name) // 2 + 8)
                if mid > 0:
                    return name[:mid] + "\n" + name[mid + 1:]
            return name

        labels = [_wrap(n) for n in det.index]
        ax.bar(det.index, det.values, color="#D55E00")
        ax.axhline(det.get("baseline", 0), color="black", linestyle="--",
                   label=f"baseline {det.get('baseline', 0):.1f}%")
        ax.set_ylabel("MSRF detection rate (%)")
        ax.set_title("Detection rate under adaptive attacks (real fused detector)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, fontsize=8, ha="right")
    fig.tight_layout()
    return fig


def fig_hyperparam(plt, out_dir: Path, msrf_json: Path):
    """超参敏感性曲线（fusion lr × thresholds）。

    v6.5.3-r7 修复：读 config 的 5 档 lr（原硬编码 3 档 "0.0001/0.0003/0.001"，
    caption 却宣称 5 档——图文不符）。key 兼容字符串与数值两种形态。
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not msrf_json.exists():
        ax.text(0.5, 0.5, "MSRF data missing", ha="center", fontsize=12)
    else:
        ev = json.loads(msrf_json.read_text(encoding="utf-8"))
        # 从 config 读 5 档（stage_f 同目录 pipeline_config.yaml）
        lrs_cfg = []
        try:
            cfg = load_config(str(out_dir.parent / "pipeline_config.yaml"))
            lrs_cfg = list(cfg.get("p2", {}).get("fusion", {}).get("lr", []))
        except Exception:  # noqa: BLE001
            pass
        if not lrs_cfg:
            lrs_cfg = ["0.00005", "0.0001", "0.0003", "0.001", "0.003"]
        # seed_detail 中 fusion 键可能是字符串或数值
        lrs = [str(x) for x in lrs_cfg]
        aucs = []
        for lr in lrs:
            vals = []
            for sr_ in ev.get("seed_detail", []):
                fus = sr_.get("fusion", {})
                v = fus.get(lr)
                if v is None:
                    try:
                        v = fus.get(float(lr))
                    except (TypeError, ValueError):  # noqa: BLE001
                        v = None
                if v and v.get("auc"):
                    vals.append(v["auc"])
            aucs.append(np.mean(vals) if vals else None)
        x = np.arange(len(lrs))
        # 无数据的档位用空圆点标注（不参与连线）
        plot_vals = [a if a is not None else float("nan") for a in aucs]
        ax.plot(x, plot_vals, "o-", color="#009E73")
        for xi, a in zip(x, aucs):
            if a is None:
                ax.plot(xi, 0.4, "o", color="#999999", markersize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(lrs, rotation=20, fontsize=8)
        ax.set_xlabel("Fusion LR")
        ax.set_ylabel("AUC")
        ax.set_title(f"Hyperparameter sensitivity: fusion learning-rate sweep ({len(lrs)} levels)")
        ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["workdir"]).expanduser()
    log = setup_logging(str(root / "logs" / f"{STAGE}.log"), STAGE)
    jlog = JsonlLogger(str(root / "logs" / f"{STAGE}.jsonl"))
    ckpt = Checkpoint(str(root / "checkpoints" / f"{STAGE}.json"))

    log.info("=== 阶段 F（出版级图表）启动 ===")
    if ckpt.is_done("done"):
        log.info("F 已完成，跳过")
        return 0

    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib(cfg, log)

    # 收集输入数据
    effects = {}
    p1p = root / "results" / "p1_pilot_effects.json"
    if p1p.exists():
        effects["P1-PILOT"] = json.loads(p1p.read_text(encoding="utf-8"))
    p1f = root / "results" / "p1_full_stats.json"
    if p1f.exists():
        effects["P1-FULL"] = json.loads(p1f.read_text(encoding="utf-8"))
    msrf_json = root / "results" / "msrf_evaluation.json"
    cmsc_csv = root / "report" / "crossmodal_pcsd.csv"   # v6.4：PCSD 输出
    p2c_csv = root / "report" / "adaptive_attack_results.csv"

    figs = {
        "overview_diagram": fig_overview(plt, out_dir),
        "factorial_forest": fig_factorial_forest(plt, out_dir, effects),
        "pcsd_heatmap": fig_cmsc_heatmap(plt, out_dir, cmsc_csv),  # v6.4 改名
        "msrf_roc_pr": fig_msrf_roc(plt, out_dir, msrf_json),
        "ablation_bars": fig_ablation_bars(plt, out_dir, msrf_json),
        "adaptive_decay": fig_adaptive_decay(plt, out_dir, p2c_csv),
        "hyperparam_curves": fig_hyperparam(plt, out_dir, msrf_json),
    }

    # v6.5.3：双语 caption（F-2）——caption 自含 + 中英双语，写入索引
    captions = {
        "overview_diagram": (
            "Fig.1 归因→检测逻辑链总览：从 E_t×N×R×A_s 全析因归因到 MSRF 四分支检测框架。",
            "Fig.1 Overview of the attribution-to-detection logic chain: from the "
            "full-factorial attribution of E_t×N×R×A_s to the four-branch MSRF detector."),
        "factorial_forest": (
            "Fig.2 析因效应森林图：各语义成分对 ASR 的主效应与交互效应（bootstrap 95% CI）。",
            "Fig.2 Factorial forest plot: main and interaction effects of framing "
            "components on ASR (bootstrap 95% CI)."),
        "pcsd_heatmap": (
            "Fig.3 PCSD 配对分歧热力图：攻击条件下响应级安全判定一致率与分歧方向。",
            "Fig.3 PCSD paired-divergence heatmap: response-level agreement and "
            "divergence asymmetry under attack conditions."),
        "msrf_roc_pr": (
            "Fig.4 MSRF 检测 ROC/PR 曲线（真实融合器，5 种子均值±10-90 百分位包络）。",
            "Fig.4 MSRF ROC/PR curves from the real fused detector "
            "(5-seed mean with 10-90 percentile envelope)."),
        "ablation_bars": (
            "Fig.5 互补性消融：去一分支后的融合器性能（5 种子误差棒）。",
            "Fig.5 Complementarity ablation: fusion performance after removing "
            "each branch (5-seed error bars)."),
        "adaptive_decay": (
            "Fig.6 自适应攻击下的检测率衰减（灰盒/白盒，真实融合器）。",
            "Fig.6 Detection-rate decay under adaptive attacks (gray-box/white-box, "
            "real fused detector)."),
        "hyperparam_curves": (
            "Fig.7 超参敏感性：融合层学习率扫描曲线（5 档）。",
            "Fig.7 Hyperparameter sensitivity: fusion learning-rate sweep (5 levels)."),
    }

    made = []
    for name, fig in figs.items():
        for fmt in cfg["figures"].get("formats", ["pdf", "png"]):
            p = out_dir / f"{name}.{fmt}"
            fig.savefig(p, format=fmt, bbox_inches="tight")
            made.append(str(p))
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    log.info("图表输出: %d 个文件", len(made))

    # figure_index.md（v6.5.3：含双语 caption，caption 自含）
    idx = ["# 图表索引（v6.4）\n",
           f"- 生成时间: {time_str()}\n",
           f"- 输出: figures/（dpi={cfg['figures'].get('dpi', 300)}，"
           f"格式={'/'.join(cfg['figures'].get('formats', ['pdf', 'png']))}，"
           f"色盲友好={cfg['figures'].get('colorblind_safe', True)}）\n",
           "- v6.5.3: 每图配中英双语 caption（caption 自含，不依赖正文）\n\n"]
    for name in cfg["figures"].get("required", []):
        cc = captions.get(name, ("", ""))
        idx.append(f"### {name}\n- **EN**: {cc[1]}\n- **中文**: {cc[0]}\n"
                   f"- 文件: figures/{name}.pdf / .png\n\n")
    (root / "report" / "figure_index.md").write_text("".join(idx),
                                                     encoding="utf-8")

    jlog.event(stage=STAGE, event="done", n_files=len(made))
    ckpt.mark_done("done")
    log.info("=== F 完成 ===")
    return 0


def time_str():
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    sys.exit(main())
