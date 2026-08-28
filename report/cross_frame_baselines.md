# 跨 5 帧外部基线评估（P2-CROSSFRAME, P2-CROSSFRAME，纯 CPU）
- 样本: 16200 | 帧数: 5
- 校验: seed0 帧复算与 cross_method_roc_auc.json 对齐 ✅ diff={'shieldgemma': 0.0, 'wildguard': 0.0, 'gradsafe_proxy': 0.0, 'gradsafe_real': 0.0}

## 各帧 ROC-AUC（融合 vs 基线）
| 帧 | 测试数 | pos率 | MSRF融合 | ShieldGemma | WildGuard | GradSafe代理 | GradSafe真实 |
|---|---|---|---|---|---|---|---|
| 20260811 | 3863 | 3.6% | 0.9784 | — | — | — | — |
| 20260812 | 2198 | 22.6% | 0.8909 | — | — | — | — |
| 20260813 | 3323 | 5.9% | 0.9447 | — | — | — | — |
| 20260814 | 4242 | 6.5% | 0.9667 | — | — | — | — |
| 20260815 | 3149 | 8.5% | 0.9470 | — | — | — | — |

## 每帧胜者与边际
- seed 20260811: 融合 0.9784 vs 最强基线 0.9652（gradsafe_proxy），边际 0.0132
- seed 20260812: 融合 0.8909 vs 最强基线 0.8582（gradsafe_proxy），边际 0.0327
- seed 20260813: 融合 0.9447 vs 最强基线 0.9076（gradsafe_proxy），边际 0.0371
- seed 20260814: 融合 0.9667 vs 最强基线 0.9508（gradsafe_proxy），边际 0.0159
- seed 20260815: 融合 0.947 vs 最强基线 0.9128（gradsafe_proxy），边际 0.0342

## 结论
- 融合在全部 5 帧胜过最强基线: ✅ 是

## 诚实披露
- PR-AUC 依赖各帧正例率（pos_rate 逐帧披露）；跨帧对比以 ROC-AUC 与 TPR@FPR5 为主。
- GradSafe 代理与 MSRF 共用分支特征空间，非独立开源分类器，论文须与真实推理基线区分标注。
- GradSafe 真实仅 seed0 帧有产物（0.3152）；跨帧未测，论文不得做真实 GradSafe 的跨帧断言。
- ShieldGemma/WildGuard 为二元 unsafe 打分，ROC 为二分类分数阶梯，跨帧可比。
