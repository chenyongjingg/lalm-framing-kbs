# benign-FPR 口径与 TPR@FPR5=0 解释（红旗4 修复定稿）

> 2026-08-25 · 目的：消除两个审稿人会追问的口径问题：
> (a) "benign FPR" 列的准确含义；(b) 开源分类器 TPR@FPR5=0 与 ROC<0.5 的解释。

## (a) "benign FPR" 列的真实含义

- 实测 `results/msrf_evaluation.json` 的 `benign_mask`：16200 行 **全部为 False**，
  即本数据集**不存在独立的"良性"标注子集**。
- 代码（stage_p2_baselines.py:196-199）中 `benign_fpr` 实际计算的是
  **label=0（无害）行在最佳阈值下的误报率** `mean(score>=th | y==0)`，
  是 FPR@最佳阈值的一个代理，与"良性集 FPR"并非同一概念。
- 因此报告列头已由 "benign FPR" 修正为 **"FP率(无害行)@最佳阈值"**（v6.6.x-fix，2026-08-25），
  避免审稿人误读为独立良性集上的错误率。

| 方法 | FP率(无害行)@最佳阈值 | 解读 |
|---|---|---|
| GradSafe 代理 | 0.1079 | 无害行 10.8% 被误报 |
| ShieldGemma | 0.0000 | 从不报 unsafe → 漏报型（无害不误报，但危害全漏） |
| WildGuard | 0.0000 | 同上 |
| GradSafe-real | 0.9989 | 无害行几乎全被误报 → 全报型 |

## (b) 为何 TPR@FPR5% = 0（二进制分数离散工作点）

- ShieldGemma/WildGuard/GradSafe-real 输出**二进制 hard label**（unsafe ∈ {0,1}），
  ROC 曲线只有 {FPR=0,TPR=0} → {FPR=f,TPR=t} → {1,1} 三个点。
- TPR@FPR5% 要求"FPR 恰为 5% 时"的 TPR。二进制分数无任何阈值能把 FPR 调到 5%
  （能调的只有 0%、f%、100% 三档）→ 按"取 FPR≤5% 最近的阈值"计，三者的 TPR 均为 0。
- **这不是"检测能力为零"的误读**：它是离散评分造成的**不可计算**（undefined operating point）。
  其真实水平应看 ROC-AUC（已统一口径，见 cross_method_roc_auc.md）：
  ShieldGemma 0.198 / WildGuard 0.174 / GradSafe-real 0.315，全部 < 0.5。

## (c) 为何 ROC-AUC < 0.5（反向相关，positive finding）

- ROC<0.5 意味着这些分类器对 framing 攻击**系统性反向**：把大量无害文本标 unsafe、
  放行大量有害 framing（漏报危害 + 误报无害的双向错配）。
- 机制：开源安全分类器的标签域为**通用安全**（sexual/violence/illegal 等），
  与本研究"framing 危害"标签域错配；framing 攻击本质是绕过安全分类器 → 被漏报。
- 论文呈现方式：作为 **framing 攻击有效性的直接证据 + 现有安全防线脆弱性的新发现**，
  而非"基线失败"的负面叙事（详见 report/cross_method_roc_auc.md）。

## 一句话口径（供论文/答辩引用）

> "本数据集无独立良性标注（benign_mask 全 False），表中 FP 率指无害行（label=0）
> 在最佳阈值下的误报率；开源分类器因输出二进制 hard label，TPR@FPR5% 处于
> 离散工作点之外不可计算，其真实水平以同帧 ROC-AUC 为准（0.17–0.32，均低于随机），
> 反映域错配下的 framing 攻击规避。"
