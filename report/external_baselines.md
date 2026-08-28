# P2 外部安全基线评估（v6.5.2 真实化）
- 样本: 16200 | 来源: P2-10 外部基线真实化：GradSafe 梯度代理（零额外模型）+ ShieldGemma(Llama-Guard 类,4bit) + WildGuard(NeurIPS 2024, Mistral-7B, Apache 2.0) 真实推理；JailGuard/Cross-modal/SALMONN-Guard 因依赖未公开/授权受限如实披露为未复现，不虚构数字。v6.6.1（问题 42）：GradSafe/ShieldGemma 仅在与 MSRF 相同的 te_mask_seed0 测试集上评估（GradSafe 训练子集拟合），杜绝口径漂移。v6.6.2（D2-7，2026-08-11）：WildGuard 作为第二开源安全分类器基线，与 ShieldGemma 训练家族/数据/风险分类独立互补。

## 指标对比（口径说明：见下表注）
| 方法 | AUPRC | TPR@FPR5% | FP率(无害行)@最佳阈值 | ECE | 说明 |
|---|---|---|---|---|---|
| MSRF 融合 | 0.9455* | 0.6491 | — | 0.0258 | 本方法（P2 真实融合器） |
| GradSafe（梯度代理） | 0.5998 | 0.6403 | 0.1079 | 0.0857 | 零额外模型 |
| google/shieldgemma-9b | 0.1733 | 0.0 | 0.0 | 0.8799 | 真实推理（3863 条） |

## 可部署性（KBS 卖点，§8.9）
- ShieldGemma: 单次延迟 mean=0.198s p95=0.208s | 峰值显存 6.7 GB （n=16200）
| WildGuard(NeurIPS2024) | 0.1552 | 0.0 | 0.0 | 0.8926 | 开源安全分类器（3863 条） |

## 可部署性 — WildGuard（KBS 卖点，§8.11）
- WildGuard: 单次延迟 mean=1.04s p95=1.099s | 峰值显存 4.84 GB （n=16150）
| GradSafe-real(ACL2024) | 0.0239 | 0.0 | 0.9989 | 0.1388 | 梯度分析复现（3863 条） |
> * 主表内 MSRF 行报告 ROC-AUC、外部基线行为 PR-AUC（AUPRC），两口径并置仅供纵向参照。
跨方法横向比较请用下表同帧 ROC-AUC（pos 不变，可跨 pos 率；红旗1 修复，2026-08-25）。

## 同帧 ROC-AUC 对齐对比（红旗1 修复，te_mask_seed0，3863 条，pos 3.6%）
| 方法 | ROC-AUC | PR-AUC | TPR@FPR5% | 备注 |
|---|---|---|---|---|
| MSRF 融合（seed0, 部署模型） | 0.9784 | 0.6119 | 0.8345 | 本方法；与基线同帧 |
| GradSafe 代理（GBC 0.7） | 0.9652 | 0.5961 | — | 同特征空间代理 |
| google/shieldgemma-9b | 0.1983 | 0.0287 | — | 真实推理 |
| WildGuard(Mistral-7B) | 0.1737 | 0.0291 | — | 真实推理 |
| GradSafe-real(ACL2024) | 0.3152 | 0.0243 | — | 梯度分析复现 |
- MSRF 5 档 lr ROC 仅 0.9783–0.9787（超参不敏感）；5-seed CV 0.9455±0.0301。
- 开源分类器 ROC<0.5 = 对 framing 攻击反向相关（漏报危害、误报无害），系域错配 + 攻击规避的 positive finding（详见 report/cross_method_roc_auc.md）。

## 未复现项（如实披露，不虚构）
- **JailGuard**: 多步 RAG 检索管线未公开，无法高保真复现
- **Cross-modal Information Check**: 跨模态对齐头未发布权重，仅论文报告值可参考
- **SALMONN-Guard**: SALMONN-7B 权重未开放，AudioGuard 依赖受限

## 科学一致性声明
- 本阶段删除旧版"基线清单"（仅文字列表），改为真实计算/推理；
- 无法复现的基线明确披露原因，数字一律不编造；
- G2 判据"较最佳单分支 ≥3pp"仍以 MSRF 融合 vs 单分支为主，外部基线作为绝对水平的参照系。
