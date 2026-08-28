# MSRF 融合防御评估（v6.4）
- 训练样本: 10732（银标签=三方一致）- 种子数: 5 | FPR 目标: 0.05- 银标签数: 10732
- **组分割（v6.5.3，防泄漏）**: ✅ GroupShuffleSplit 组级划分（模板族+攻击族整组进测试集）
- **银标签质量估计（v6.5.3）**: 三方一致率 66.2%，样本间评分器两两一致率均值 81.2%（训练前估计；真实错误率由 R 阶段 gold/post_validation.py 事后核定）

## 基线清单（v6.5 §8 评估基线·三级分离）
- **实测基线（P2-B 承接）**: ShieldGemma-9b、WildGuard、GradSafe（ACL 2024，安全关键梯度分析，无需额外模型）、≥4 种 prompt 级防御- **引用值（文献公开结果）**: 提示级防御（PJ-Break 等）、风格级攻击（StyleBreak 等）、未见攻击族（NYHM 等）- **未复现（如实披露）**: JailGuard、Cross-modal Information Check（音频域适配）、SALMONN-Guard- 指标：分层拦截率、AUPRC、固定 FPR 下 TPR、benign FPR、ECE/Brier、未见攻击家族泛化、缺失模态鲁棒性、单次延迟与显存

## 泛化与鲁棒性（§8.9 指标）
- **未见攻击家族泛化**: GroupShuffleSplit 组级划分（组键 ['template_family', 'attack_family']）→ 测试族样本不在训练集，OOF 测试集指标即代表未见攻击族泛化（✅ 组级保证）
- **缺失模态鲁棒性**: §8.4 mean_impute + mask 指示列（缺失音频不置 0；mask 承载缺失信息）→ 缺音频文本攻击仍可判定

## 单分支
| 分支 | AUC(mean±std) | TPR@FPR5% |
|---|---|---|
| intent | 0.9325±0.033 | 0.601 |
| narrative | 0.9136±0.0484 | 0.529 |
| acoustic | 0.5±0.0 | 0.0 |
| uncertainty | None±None | None |

## 融合（5 种子最优）
- AUC: 0.9455±0.0301  TPR@FPR5%: 0.6491  ECE: 0.0258

## 输入过滤级检测器（§8 双阶段·请求级，与输出审核分开报告）
- 数据/标签不足，未评估（如实披露，不虚构）

## 超参选择与测试真值披露（AUDIT #172 C3/C4 修复）
- **lr 超参选择**: 独立验证集 AUC（嵌套 GroupShuffleSplit：外层 25% 测试行恒不参与选择，内层再切 20% 验证集）；测试 AUC 仅报告、不参与选择（test_auc_reported_only=True）
- 各种子嵌套切分行数 train/val/test: 5816/1053/3863, 6474/2060/2198, 6092/1317/3323, 5630/860/4242, 6409/1174/3149
- **测试真值来源**: silver_label——报告 AUC 为与银标签的一致度，须如实解读

## 消融（去一分支后 AUC）
| 移除 | AUC | TPR@FPR5% |
|---|---|---|
| intent | 0.9136 | 0.529 |
| narrative | 0.9325 | 0.601 |
| acoustic | 0.9449 | 0.6435 |
| uncertainty | 0.9448 | 0.6431 |

## G2 判据
- TPR 增益: 4.81pp (要求 ≥3pp) → ✅ 通过

## 分层拦截率（§8.9，按困难样本类型分层的 TPR@FPR5% / AUPRC）
| 分层 | TPR@FPR5% (mean) | AUPRC (mean) | n (mean) |
|---|---|---|---|
| normal | None | None | 0.0 |
| disputed | None | None | 0.0 |
| extreme_asr | 0.0132 | 0.375 | 2273.8 |
| triple_mismatch | None | None | 0.0 |
| cross_family_boundary | 0.3464 | 0.6161 | 1081.2 |
（n<5 或阈值不可得的分层如实记 None，不虚构）

## 标签质量与银标签披露（§8）
- 方法: proxy_agreement_rates + R 阶段人工事后核定
- 银标签率: 0.6625（三方一致）
- 样本间评分器两两一致率均值: 0.8122
- 公开基准校验: not_performed（本文响应池无金标准，禁止虚构公开基准数字）
- 最终真值: gold/post_validation.py（阶段 R 自动核定）

## Uncertainty 信号披露（§8 目标模型置信信号）
- 置信信号: hb_prob（主评分器置信代理；目标模型生成置信信号不可得时部署可用的替代）
- 分歧信号: 2·min(p,1-p)（多评分器有效票分歧，0=全一致/1=最大分歧）
- 说明: 规范§8'目标模型置信信号'以主评分器置信代理并如实披露；区别于 Chen et al.（EMNLP 2025）单一首 token 信号
