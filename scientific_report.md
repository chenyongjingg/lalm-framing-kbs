# LALM Framing 科学报告（Knowledge-Based Systems 目标）

> 自动生成：2026-08-10 13:46 UTC（第 10 轮审查后）。本文件为「睡醒可读」的可发表素材汇编。
> 关联：audit_log.md（第 75 轮）、report/ 各阶段报告、results/、gates/。

## 1. 实验配置摘要（第 10 轮快照）

| 项 | 值 |
|:---|:---|
| 基准文件 | prompt.md v6.5（SHA256 `447f9431…`，24700B，**自第 1 轮未变**） |
| 协议 | RESEARCH_PROTOCOL.md（SHA256 `57c47423…`，含 D1 决策：中文配额 300→400） |
| 阶段契约 | STAGE_CONTRACTS.md（SHA256 `cf58c727…`） |
| 配置 | pipeline_config.yaml（SHA256 `7be4160c…`，zh_n=400 / en_n=300 / G1 min_effect_pp=10 / 5 训练种子） |
| 环境 | RTX 4090 24GB · torch 2.13.0+cu130 · cudnn 92000 · 内核 5.15.0-113 · 磁盘 209G 空闲 |
| 种子 | PILOT=20260803 · FULL-zh=20260804 · FULL-en=20260805 · AdvBench=20260806 · 训练 [20260811-15] |
| 确定性 | manual_seed + cudnn.deterministic=True + CUBLAS_WORKSPACE_CONFIG=":4096:8"（common_utils:506-510） |
| 在途进程 | P1-PILOT 推理 PID 198729（--resume），checkpoint 账本 8519/10800（78.9%），E4B |

## 2. 当前阶段状态

- 已完成：阶段 L（novelty_audit / kbs_scope_papers / citation_verification / related_work_positioning）、阶段 D（400 中文+400 英文平行有害查询，DONE_0）。
- 进行中：阶段 P1-PILOT 推理（E4B 8519/10800；E2B text 段待 E4B 后）。
- 待启动：G1 闸门（P1-PILOT 完成后）→ P1-FULL ∥ P0-C → P2 → …

## 3. 主结果表（当前可得，LaTeX）

P1-PILOT 推理未完成，评分未开始，主效应表待 G1 后填。已产出数据集清单：

```latex
% 数据集（阶段 D）
\begin{table}[ht]
\centering
\caption{有害查询集（D1 决策后实际）}
\begin{tabular}{lcc}
\toprule
集合 & 数量 & 说明 \
\midrule
中文（平行英文） & 400 & 6 威胁类别，中英语义对齐（en 经全量平行翻译，超协议 300 下限） \
英文独立生成 & 0 & 平行翻译覆盖（协议 en_n=300 为下限，实际 400） \
良性对照 & 300 & 含叙事结构但完全良性 \
AdvBench 锚定 & 200 & 种子 20260806 \
\bottomrule
\end{tabular}
\end{table}
```

## 4. 与 prompt.md 预期对比（符合度评分）

| prompt.md 条款 | 状态 | 符合度 | 说明 |
|:---|:---|:---|:---|
| §2 阶段 L | ✅ 完成 | 100% | novelty_audit/kbs_scope/citation_verification/positioning 全产出 |
| §3 阶段 D 查询集 | ✅ 完成 | 95% | 中文 400（D1 批准偏离）、英文实际 400（平行翻译，待阶段 R 更新数据集描述） |
| §5 P1-PILOT 设计 | ✅ 在途 | 符合 | 150×8×3×3=10800 单元、账本格式 [qid,factors,A_s,模板idx] 与设计吻合 |
| §4.2 评分器中文适用性 | ⏳ 待办 | 0% | **已知缺口**：中文 400 已就绪，待 GPU 窗口验证（见 §7） |
| §5.3 G1 判据 | 待判定 | — | gate_g1.py C1-C5 与协议逐项对应（第 9 轮已全文核验） |

## 5. 发现的问题与修正（第 10 轮）

1. **posthoc 补算脚本未接线（已修复）**：P1-PILOT 在途进程装载 pre-#8 旧代码，完成后不产出评分器一致性 JSON。已在 pipeline.sh 的 G1 前插入幂等调用。备份 `pipeline.sh.bak_r75_20260810_133315`，新 sha256 `0a1a4055…`。科学依据：三口径一致性（κ/Spearman）是 §4.3 判据的量化证据，KBS 主表必需。
2. **协议英文配额 vs 实际数据（文档性）**：协议 L302 写「英文 300」，实际英文池 400（平行翻译）。建议阶段 R 数据集描述写「中文 400 + 英文 400 中英平行」。
3. **run/d.complete 缺失（无科学影响）**：D 已成功但手动启动未建 marker，pipeline.sh 下次幂等重跑秒级跳过。

## 6. KBS 补全模块进展

- 统计效力：5 种子（训练组件）已配置；bootstrap CI 已实现（stage_p1_pilot:_bootstrap_ci）；TOST/序贯方案为 P3 内容。
- 消融：config:263 ablation 开关 + stage_p2_msrf 互补性消融（去一分支，5 种子）。
- 误差分析/可解释性：混淆矩阵、特征重要性样例 10 条已内建 P2 要求。
- 计算预算透明化：心跳/monitor 记录 GPU 峰值显存（当前 E4B 推理 15.8GB）。
- 鲁棒性/敏感度：融合层超参 5 档扫描、拒绝阈值双策略（P2）。
- 过拟合监控：训练/验证 gap 监控（P2 训练期）。

## 7. 已知缺口与风险登记

- **§4.2 中文适用性验证未评估**（第 9 轮已披露）。方案：P0 已有完整验证基建（scorer_validation_on_public_benchmarks.md 602 行真实数据），中文 400 查询已就绪 → 在 GPU 空闲窗口运行中文评分器验证，更新该报告。
- 前沿基线（PJ-Break/StyleBreak/Now-You-Hear-Me）真实复现未实现（errors.jsonl 有披露），以 reported_value 呈现。
- G1 判定后需执行可复现性二次运行（阶段 4.3）。

## 8. 下一步建议（自我进化计划）

1. 监控 P1-PILOT 完成 → 验证 posthoc 接线自动产出 agreement JSON（幂等）。
2. GPU 空闲窗口运行 §4.2 中文适用性验证。
3. G1 判定 → 视结果走 P1-FULL ∥ P0-C 或交互/机制主导叙事。
4. 阶段 R 数据集描述更新（en 400 中英平行）。
