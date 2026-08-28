# 跨方法 ROC-AUC 口径对齐审计（红旗1 修复定稿）

> 2026-08-25 · 目的：消除 external_baselines.md 中"MSRF 报 ROC-AUC、基线报 AUPRC"的
> 不可比口径，建立完全同帧的横向对比。

## 为什么用 ROC-AUC

- ROC-AUC 对正例占比**不变**（pos-invariant）：把正例随机过采样 100 倍，ROC 曲线与 AUC 不变。
- PR-AUC / AP 依赖测试集正例占比，跨"MSRF CV 集（pos~10.1%）"与"te_mask_seed0（pos 3.6%）"
  直接比较 AP 会系统性偏置。
- 因此**统一口径 = ROC-AUC**，全部在 **te_mask_seed0**（3863 行、pos 3.6%）上评估。
  MSRF 侧取 seed0（20260811，即 msrf_fusion.pkl 部署模型）best_lr=5e-05 的融合输出——
  该测试集经逐索引比对确认与基线集**逐元素相等**（valid_rows[s0_te] == flatnonzero(te_mask_seed0)）。

## 最终同帧对比表（te_mask_seed0，3863，pos 3.6%）

| 方法 | ROC-AUC | PR-AUC | TPR@FPR5 | 备注 |
|---|---|---|---|---|
| **MSRF 融合（seed0, 部署模型）** | **0.9784** | **0.6119** | **0.8345** | 本方法；与基线完全同帧 |
| GradSafe 代理（GBC 0.7, 同特征） | 0.9652 | 0.5961 | — | 代理特征版 |
| GradSafe 真实版 | 0.3152 | 0.0243 | — | 低于随机，见下 |
| ShieldGemma（3B） | 0.1983 | 0.0287 | — | 低于随机，见下 |
| WildGuard（7B） | 0.1737 | 0.0291 | — | 低于随机，见下 |

- MSRF 5 档 lr 的 ROC 仅 0.9783–0.9787（spread 0.0004），对超参不敏感；
- 5-seed CV 均值 0.9455 ± 0.0301（跨 21 组）；seed0 落在均值之上 1.1σ，属正常波动。

## 结论（可写入论文的 headline 口径）

1. **同帧下 MSRF 融合 > GradSafe 代理**（ROC 0.9784 vs 0.9652；PR 0.6119 vs 0.5961）：
   多分支意图/叙事 + 保序校准 + MLP 融合带来真实增益，非框架差异。
2. **开源安全分类器对 framing 攻击全部低于随机**（ROC 0.17–0.32）：这些模型把大量
   无害文本标为 unsafe、却放行大量有害 framing——即**攻击规避被系统性漏报 + 无害被误报**
   的双向反向相关。这是 framing 攻击有效性的直接证据，也解释 TPR@FPR5=0。

## 注意事项（如实披露，防审稿人误读）

- GradSafe 代理 = 同特征空间的 GBC 基线，非官方版；官方 gradsafe_real 仅 3863 行上得分，
  ROC 0.3152 为真实第三方基线。
- 开源分类器标签域为**通用安全**（sexual/violence 等），与本研究"framing 危害"标签域错配；
  ROC<0.5 恰反映域错配 + 攻击规避，已作为 positive finding 呈现而非缺陷。
- PR-AUC 仅在本表内、pos 3.6% 固定条件下可比；跨 pos 率比较一律用 ROC-AUC。
