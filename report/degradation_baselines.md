# 降级音频 + 基线对比（v6.5，真实化 + 报告值分离）

## 降级音频
| 模型 | 降级 | 可用样本 | 重推理 |
|---|---|---|---|
| gemma_4_e4b | opus_16kbps | 200 | 是 |
| gemma_4_e2b | opus_16kbps | 200 | 是 |
| gemma_4_e4b | snr_20db | 200 | 是 |
| gemma_4_e2b | snr_20db | 200 | 是 |
| gemma_4_e4b | rir_reverb | 200 | 是 |
| gemma_4_e2b | rir_reverb | 200 | 是 |

## 基线方法对比（本文实测）
| 方法 | 模型 | 条件 | ASR(%) | source | 备注 |
|---|---|---|---|---|---|
| no_defense | gemma_4_e4b | baseline | 5.78 | measured |  |
| no_defense | gemma_4_e4b | storytelling | 27.5 | measured |  |
| prefix_injection | gemma_4_e4b | storytelling | 2.5 | measured | HarmBench 实测（--score） |
| defensive_reframing | gemma_4_e4b | storytelling | 1.5 | measured | HarmBench 实测（--score） |
| self_reminder | gemma_4_e4b | storytelling | 0.5 | measured | HarmBench 实测（--score） |
| best_of_n | gemma_4_e4b | storytelling | 8.0 | measured | any_jailbreak 聚合（200 查询 × 10 采样，T=1.0，HarmBench 实测） |
| no_defense | gemma_4_e2b | baseline | 6.56 | measured |  |
| no_defense | gemma_4_e2b | storytelling | 35.78 | measured |  |
| prefix_injection | gemma_4_e2b | storytelling | 1.0 | measured | HarmBench 实测（--score） |
| defensive_reframing | gemma_4_e2b | storytelling | 2.0 | measured | HarmBench 实测（--score） |
| self_reminder | gemma_4_e2b | storytelling | 0.0 | measured | HarmBench 实测（--score） |
| best_of_n | gemma_4_e2b | storytelling | 8.5 | measured | any_jailbreak 聚合（200 查询 × 10 采样，T=1.0，HarmBench 实测） |

## 前沿基线：文献引用值（**未在本文复现**，不作本文实验证据）
> 以下 ASR 引自原文报告（stage_l citation_verification 核验元数据）；PJ-Break/StyleBreak/NYHM 的复现受实现复杂度/授权限制未完成，论文相关工作中须如实表述为文献引用值，不得宣称「复现」。
> **M9 核验状态（AUDIT #172）**：所有引用值 `verified=False`——数值引自原文报告，须人工逐字比对原文 PDF（citation_verification.md 未含数值核验列）。投稿前未核验的行必须删除。
| 方法 | 模型 | 条件 | 报告ASR(%) | source | 备注 |
|---|---|---|---|---|---|
| now_you_hear_me | gemma_4_e4b | tone_only_change | 8.0 | reported_value | EACL 2026 报告值。数值引自原文报告（stage_l citation_verification 核验 arXiv id）；未复现，不作本文实验证据 |
| now_you_hear_me | gemma_4_e2b | tone_only_change | 8.0 | reported_value | EACL 2026 报告值。数值引自原文报告（stage_l citation_verification 核验 arXiv id）；未复现，不作本文实验证据 |
| pj_break | gemma_4_e4b | delivery_preset | 35.0 | reported_value | PJ-Break(arXiv:2607.26541) 报告值。数值引自原文报告（stage_l citation_verification 核验 arXiv id）；未复现，不作本文实验证据; 本文 E_t/A_s 分离设计对照 |
| pj_break | gemma_4_e2b | delivery_preset | 35.0 | reported_value | PJ-Break(arXiv:2607.26541) 报告值。数值引自原文报告（stage_l citation_verification 核验 arXiv id）；未复现，不作本文实验证据; 本文 E_t/A_s 分离设计对照 |
| stylebreak | gemma_4_e4b | style_attack | 42.0 | reported_value | StyleBreak(AAAI 2026) 报告值。数值引自原文报告（stage_l citation_verification 核验 arXiv id）；未复现，不作本文实验证据; 攻击方法导向对照 |
| stylebreak | gemma_4_e2b | style_attack | 42.0 | reported_value | StyleBreak(AAAI 2026) 报告值。数值引自原文报告（stage_l citation_verification 核验 arXiv id）；未复现，不作本文实验证据; 攻击方法导向对照 |

> 数据口径说明：
> - `measured` = 本文实测（同查询集、同 LALM）
> - `measured_pending_score` = 真实推理完成待评分
> - `reported_value` = 文献引用值，**未复现**，独立小节呈现，绝混入实测

## 前沿基线未复现依据（KBS 可辩护降级）
- **PJ-Break（arXiv:2607.26541）**：delivery preset 依赖专有 TTS 参数（情感化音频投递管线未公开），无官方实现可复现；本文以其**方法学对照（E_t/A_s 分离设计，§3.4）**定位，非实测对比。
- **StyleBreak（AAAI 2026）**：攻击方法实现细节未公开（style attack 管线），无法高保真复现；本文以攻击方法导向对照定位。
- **Now You Hear Me（EACL 2026）**：音频叙事攻击实现未公开；本文以其核心主张（裸请求仅改语气 ASR<10%）作证据引用，未复现。
- **降级替代**：本文提供可复现的近似基线——降级三档（16kbps/SNR20/RIR）× 2 LALM × 200 条实测（§10.1），供跨方法对比。

## 声学防 TTS 捷径（P2-4）
- 说话人分离: storytelling `zh-CN-XiaoxiaoNeural` vs 防御 `zh-CN-YunxiNeural` → ✅ 分离
- TTS 引擎: {'storytelling': 'edge-tts (Microsoft)', 'defense': 'edge-tts (Microsoft)', 'engine_disjoint': False}
- 校验方式披露: speaker 身份级核验（如 Resemblyzer 嵌入距离）未纳入本文；以 voice 配置分离 + 实验声学条件对比替代，报告如实披露
- EBU R128 响度归一: 降级管线 loudnorm(I=-23 LUFS) 已接入 degrade_audio（v6.5.24-fix：对每档降级产物归一，失败时如实回退未归一并披露）

## 延迟/显存实测（P2-11，读 P0C 推理日志）
- 未实测（logs/p0c.jsonl 不存在）→ 论文如实披露为 not_measured

## 降级影响分析（P2B-2：PCSD / 分支信号 / 融合决策）
- **PCSD（降级 vs 原始）**: 降级 ASR 对比见下；配对一致率需 P0C 原始与降级双通道同 query 配对评分后重算（当前如实披露）
  - P0C 原始 PCSD 报告: report/pcsd_analysis.md（1732 字节）
- **降级后 ASR（HarmBench 实测，--score）**：
  - degraded_opus_16kbps_gemma_4_e2b.jsonl: ASR=59.5% （原始 audio ASR 66.5% → 变化 -7.0pp）
  - degraded_opus_16kbps_gemma_4_e4b.jsonl: ASR=54.0% （原始 audio ASR 42.67% → 变化 +11.3pp）
  - degraded_rir_reverb_gemma_4_e2b.jsonl: ASR=56.0% （原始 audio ASR 66.5% → 变化 -10.5pp）
  - degraded_rir_reverb_gemma_4_e4b.jsonl: ASR=45.5% （原始 audio ASR 42.67% → 变化 +2.8pp）
  - degraded_snr_20db_gemma_4_e2b.jsonl: ASR=57.0% （原始 audio ASR 66.5% → 变化 -9.5pp）
  - degraded_snr_20db_gemma_4_e4b.jsonl: ASR=45.5% （原始 audio ASR 42.67% → 变化 +2.8pp）
- **融合决策**: 降级产物 6 个已评分但无可用响应/真实特征行 → 翻转率未算出，如实披露（1342 条特征缺失 not_measured）
- **分支信号**: 各分支（Narrative/Acoustic）在降级音频上的特征分布对比待 P2 evaluate 补充（同上条件）

## GradSafe 外部参照（v6.5.2）
- GradSafe（梯度代理，零额外模型）: AUPRC=0.5998 TPR@FPR5%=0.6403 benign FPR=0.1079
- 完整对比见 report/external_baselines.md
