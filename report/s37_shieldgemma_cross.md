# S37：ShieldGemma-9B 独立评分家族（交叉一致 + 效应复现）

- 日期：2026-08-16；方法：ShieldGemmaScorer（4bit bnb，yes=1/no=0）
- 披露：同 Google 家族——异构训练成立、跨家族不成立；绝对基率可能与
  Gemma-4 评分器不同（见条件效应而非绝对率）。
- 结果 json：`results/gpu1_pipeline/s37_shieldgemma_cross.json`

## A. E2B 主链（3600 格）

ShieldGemma：解析 3600（None 0），有害率 0.1658

### A.1 交叉一致矩阵（一致率 | κ | Spearman）

| 对比 | n | 一致率 | κ | Spearman |
|---|---|---|---|---|
| shieldgemma vs judge_big | 3600 | 0.828 | 0.777 | 0.601 |
| shieldgemma vs judge_small | 3573 | 0.681 | 0.521 | 0.236 |
| shieldgemma vs qwen32 | 3600 | 0.804 | 0.737 | 0.597 |
| shieldgemma vs cross_check | 3598 | 0.260 | -1.113 | -0.195 |
| shieldgemma vs strongreject | 3600 | 0.467 | -0.024 | -0.118 |
| shieldgemma vs harmbench | 3600 | 0.618 | 0.389 | 0.127 |
| shieldgemma vs forced | 3600 | 0.675 | 0.507 | -0.183 |

### A.2 效应复现（query 聚类 bootstrap；✓=95%CI 排除 0）

| 效应 | ShieldGemma | dual_judge（论文权威） |
|---|---|---|
| N 效应（E_t=0, N0 vs N1） | -0.0361 [-0.0566,-0.0156] ✓ | 0.0584 [0.0348,0.0832] ✓ |
| E_t 效应（N=0, E0 vs E1） | -0.0038 [-0.0354,0.0211] ✗ | 0.0298 [0.005,0.0567] ✓ |
| E_t 效应（N=1） | 0.0253 [0.0067,0.0444] ✓ | 0.0277 [0.0094,0.0478] ✓ |
| E_t 效应（pooled） | 0.0122 [-0.0033,0.0326] ✗ | 0.0261 [0.0004,0.0473] ✓ |

### A.3 ShieldGemma 有害率按条件

| 条件 | n | 有害率 |
|---|---|---|
| E0_N0 | 900 | 0.1778 |
| E0_N1 | 900 | 0.1422 |
| E1_N0 | 900 | 0.1756 |
| E1_N1 | 900 | 0.1678 |

## B. S28 异族音频（1200 格）

ShieldGemma：解析 1200（None 0），有害率 0.3175

| 对比 | n | 一致率 | κ | Spearman |
|---|---|---|---|---|
| shieldgemma vs judge_small | 1189 | 0.689 | 0.440 | 0.087 |
| shieldgemma vs judge_big | 1200 | 0.790 | 0.651 | 0.613 |
| shieldgemma vs strongreject | 1200 | 0.466 | -0.102 | -0.092 |
| shieldgemma vs harmbench | 1200 | 0.718 | 0.503 | 0.063 |
| shieldgemma vs forced | 1200 | 0.686 | 0.432 | 0.346 |

## 解读

- 若 ShieldGemma 的 N/E_t 效应与 dual_judge 同方向且显著：效应跨评分器家族复现，测量稳健性强证据（KBS 加分）。
- 若不一致：如实披露为评分器敏感性（论文结论以权威 dual_judge 口径为准），并给出基率差异解释。

- 标签文件：`results/gpu1_pipeline/s37_shieldgemma_e2b_main_labels.jsonl`
  + `s37_shieldgemma_s28_labels.jsonl`（独立新文件，未改写生产缓存）