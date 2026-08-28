# S35：judge_small 强制解码验证 + null 格打标

- 日期：2026-08-15
- 方法：首 token argmax(logits('0'), logits('1'))，与自由生成同模板同模型

## A. E2B 主链（3600 格）

| 对比 | n | 一致率 | κ | Spearman |
|---|---|---|---|---|
| forced_vs_freegen | 3573 | 0.995 | 0.992 | 0.970 |
| forced_vs_judge_big | 3600 | 0.810 | 0.726 | -0.078 |
| forced_vs_qwen32 | 3600 | 0.813 | 0.722 | -0.031 |
| forced_vs_cross_check | 3598 | 0.534 | -0.181 | 0.206 |

分层 E_t=0：1789 | 0.993 | 0.988 | 0.975
分层 E_t=1：1784 | 0.998 | 0.996 | 0.989
分层 N=0：1777 | 0.992 | 0.988 | 0.955
分层 N=1：1796 | 0.998 | 0.997 | 0.987

|margin| 一致格 mean=8.8250 median=8.0000（n=3556）；分歧格 mean=5.1027 median=5.6250（n=17）

null 格 27 条：强制标签 0/1 = 1/26；N=0/N=1 = 23/4；judge_big 对 null 有害率 0.333
null 格 forced vs judge_big：8/27（0.296）
null 格 forced vs qwen32：10/27（0.370）
null 格 forced vs cross_check_e2b：24/27（0.889）

### N/E_t 效应敏感性（E2B 主链，dual_judge 权威口径，null 赋值场景）

| 场景 | N_effect(E_t=0) | Et_effect(N=0) | Et_effect(N=1) | Et_effect(pooled) |
|---|---|---|---|---|
| exclude（当前排除 27 格）| 0.0568 [0.0323,0.0822] ✓ | 0.0285 [0.0013,0.0538] ✓ | 0.0273 [-0.0005,0.0558] ✗ | 0.0273 [0.0034,0.0518] ✓ |
| forced_fill（null 以强制标签填充）| 0.0555 [0.0314,0.0814] ✓ | 0.0347 [0.008,0.0605] ✓ | 0.0285 [0.0004,0.0567] ✓ | 0.0308 [0.0073,0.0551] ✓ |

## B. S28 异族音频（1200 格）

| 对比 | n | 一致率 | κ | Spearman |
|---|---|---|---|---|
| forced_vs_freegen | 1189 | 0.999 | 0.999 | 0.996 |
| forced_vs_judge_big | 1200 | 0.787 | 0.634 | 0.499 |
| forced_vs_strongreject | 1200 | 0.685 | 0.354 | 0.201 |
| forced_vs_harmbench | 1200 | 0.794 | 0.628 | 0.235 |

|margin| 一致格 mean=9.0269（n=1188）；分歧格 mean=6.2246（n=1）

null 格 11 条：强制标签 0/1 = 0/11
null 格 forced vs judge_big：5/11（0.455）
null 格 forced vs strongreject：8/11（0.727）
null 格 forced vs harmbench：5/11（0.455）

- 标签文件：`results/gpu1_pipeline/s35_forced_verdict_e2b_main_labels.jsonl` + `s35_forced_verdict_s28_labels.jsonl`（独立新文件，未改写生产缓存）
- 详情：`results/gpu1_pipeline/s35_forced_verdict.json`
## 关键解读（margin 置信度诊断）

- **协议稳定性已验证**：强制解码 vs 自由生成 一致率 E2B 主链 **0.9952**（κ=0.9923）/ S28 **0.9992**（κ=0.9985）。两种评分模式互证，强制解码可作为测量缺口的标准补充。
- **null 格是低置信格，而非评分事故**：E2B 主链 null 格强制 |margin| 中位 4.06 vs 全量中位 7.88；18/27 条 margin<5（最低 0.06≈抛硬币）。S28 同理：中位 5.09 vs 7.75，5/11 条 <5。
- **与 S34 交叉验证一致**：E2B 自由生成失败的格 = E2B 自身 0/1 置信低的格。这些格上 judge_big 也分歧（9/27 有害），双评分器在同一批格上同时不确定 → 排除它们不构成系统性偏差（敏感性表佐证）。
- **强制标签方向（弱证据）**：null 格强制判 26/27（E2B 主链）、11/11（S28）有害，但均为低置信（见上）→ 只能作为「E2B 在这些格上偏有害倾向」的弱证据，不能作为强真值。与 judge_big 在这些格上仅 8/27、5/11 一致，恰反映这些格在评分器间真实分歧。
- **敏感性第 6 场景（dual_judge 权威口径，2026-08-16 修正）**：null 以强制标签填充后 N_effect 0.0568→0.0555（仍显著 ✓）、Et_effect(N=0) 0.0285→0.0347（仍显著 ✓）——与 S34 5 场景同向，论文 N/E_t 主结论完全稳健。初始脚本误用 judge_small 单评分器口径（E_t 近零），已弃用。

- 标签文件独立存放，未改写任何生产缓存。
- 详情：`results/gpu1_pipeline/s35_forced_verdict.json`
