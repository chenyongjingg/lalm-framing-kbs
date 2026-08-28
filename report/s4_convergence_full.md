# S4 全语言评分器收敛验证（GPU1 空闲期测量验证 Wave 2 · 2026-08-12）

- 单元: zh 额外 120（与 S1 去重，zh 合计 180） + en 60 + adv 60 = 240
- 响应: 480（E4B/E2B 各 240，顺序逐条 greedy）
- 评分器合奏: harmbench, strongreject, judge_big, judge_small, cross_check, shieldgemma
  （论文 4 票 = strongreject + gemma(==judge_small) + judge_big + judge_small；gemma 与 judge_small 合并）

## zh

- 响应数: 240
- 双 judge 一致率: 0.8067 （争议率 0.1933，n=238）
- dual_judge 共识 vs 异构一致率: 0.4844（n=192）
- **判定: 评分器敏感**

| 评分器对 | n | 一致率 | κ |
|---|---|---|---|
| harmbench ↔ strongreject | 240 | 0.8708 | 0.741 |
| harmbench ↔ judge_big | 240 | 0.8208 | 0.612 |
| harmbench ↔ judge_small | 238 | 0.8277 | 0.645 |
| harmbench ↔ cross_check | 237 | 0.6414 | 0.346 |
| harmbench ↔ shieldgemma | 240 | 0.5333 | -0.053 |
| strongreject ↔ judge_big | 240 | 0.7667 | 0.528 |
| strongreject ↔ judge_small | 238 | 0.8151 | 0.629 |
| strongreject ↔ cross_check | 237 | 0.6878 | 0.388 |
| strongreject ↔ shieldgemma | 240 | 0.4625 | -0.094 |
| judge_big ↔ judge_small | 238 | 0.8067 | 0.579 |
| judge_big ↔ cross_check | 237 | 0.4979 | 0.197 |
| judge_big ↔ shieldgemma | 240 | 0.6875 | 0.090 |
| judge_small ↔ cross_check | 236 | 0.6356 | 0.334 |
| judge_small ↔ shieldgemma | 238 | 0.5294 | -0.070 |
| cross_check ↔ shieldgemma | 237 | 0.2321 | -0.126 |

Dawid-Skene 潜类别误差率：
| 评分器 | 敏感度 | 特异度 | 加权误分类率 |
|---|---|---|---|
| harmbench | 0.9061 | 0.959 | 0.0675 |
| strongreject | 0.9583 | 0.8738 | 0.0839 |
| judge_big | 0.6237 | 1.0 | 0.1882 |
| judge_small | 0.8391 | 0.9155 | 0.1227 |
| cross_check | 1.0 | 0.4054 | 0.2973 |
| shieldgemma | 0.1343 | 0.8233 | 0.5212 |
- 潜类别 P(有害) = 0.4375

## en

- 响应数: 120
- 双 judge 一致率: 0.6949 （争议率 0.3051，n=118）
- dual_judge 共识 vs 异构一致率: 0.5244（n=82）
- **判定: 评分器敏感**

| 评分器对 | n | 一致率 | κ |
|---|---|---|---|
| harmbench ↔ strongreject | 120 | 0.8583 | 0.718 |
| harmbench ↔ judge_big | 120 | 0.6417 | 0.277 |
| harmbench ↔ judge_small | 118 | 0.8644 | 0.729 |
| harmbench ↔ cross_check | 118 | 0.7119 | 0.429 |
| harmbench ↔ shieldgemma | 120 | 0.4417 | -0.130 |
| strongreject ↔ judge_big | 120 | 0.6000 | 0.300 |
| strongreject ↔ judge_small | 118 | 0.8644 | 0.728 |
| strongreject ↔ cross_check | 118 | 0.8136 | 0.565 |
| strongreject ↔ shieldgemma | 120 | 0.3167 | -0.143 |
| judge_big ↔ judge_small | 118 | 0.6949 | 0.395 |
| judge_big ↔ cross_check | 118 | 0.4576 | 0.162 |
| judge_big ↔ shieldgemma | 120 | 0.6833 | -0.040 |
| judge_small ↔ cross_check | 117 | 0.7350 | 0.462 |
| judge_small ↔ shieldgemma | 118 | 0.4322 | -0.121 |
| cross_check ↔ shieldgemma | 118 | 0.1864 | -0.156 |

Dawid-Skene 潜类别误差率：
| 评分器 | 敏感度 | 特异度 | 加权误分类率 |
|---|---|---|---|
| harmbench | 0.8494 | 1.0 | 0.0753 |
| strongreject | 1.0 | 0.8707 | 0.0647 |
| judge_big | 0.4031 | 1.0 | 0.2985 |
| judge_small | 0.8658 | 0.9794 | 0.0774 |
| cross_check | 1.0 | 0.5234 | 0.2383 |
| shieldgemma | 0.0627 | 0.7695 | 0.5839 |
- 潜类别 P(有害) = 0.5667

## adv

- 响应数: 120
- 双 judge 一致率: 0.8750 （争议率 0.1250，n=120）
- dual_judge 共识 vs 异构一致率: 0.5333（n=105）
- **判定: 评分器敏感**

| 评分器对 | n | 一致率 | κ |
|---|---|---|---|
| harmbench ↔ strongreject | 120 | 0.9000 | 0.690 |
| harmbench ↔ judge_big | 120 | 0.9000 | 0.625 |
| harmbench ↔ judge_small | 120 | 0.8750 | 0.619 |
| harmbench ↔ cross_check | 120 | 0.6167 | 0.281 |
| harmbench ↔ shieldgemma | 120 | 0.6917 | -0.047 |
| strongreject ↔ judge_big | 120 | 0.9000 | 0.681 |
| strongreject ↔ judge_small | 120 | 0.9417 | 0.839 |
| strongreject ↔ cross_check | 120 | 0.6833 | 0.399 |
| strongreject ↔ shieldgemma | 120 | 0.6417 | -0.068 |
| judge_big ↔ judge_small | 120 | 0.8750 | 0.608 |
| judge_big ↔ cross_check | 120 | 0.6000 | 0.252 |
| judge_big ↔ shieldgemma | 120 | 0.7083 | -0.026 |
| judge_small ↔ cross_check | 120 | 0.6917 | 0.414 |
| judge_small ↔ shieldgemma | 120 | 0.6333 | -0.076 |
| cross_check ↔ shieldgemma | 120 | 0.4083 | -0.115 |

Dawid-Skene 潜类别误差率：
| 评分器 | 敏感度 | 特异度 | 加权误分类率 |
|---|---|---|---|
| harmbench | 0.5968 | 0.9928 | 0.2052 |
| strongreject | 0.8628 | 1.0 | 0.0686 |
| judge_big | 0.5547 | 1.0 | 0.2227 |
| judge_small | 0.8935 | 1.0 | 0.0533 |
| cross_check | 1.0 | 0.6168 | 0.1916 |
| shieldgemma | 0.1235 | 0.7831 | 0.5467 |
- 潜类别 P(有害) = 0.2667

## 说明
> 无金标准：以多评分器共识强度度量各语言响应测量的稳健性。en 为论文主语言，adv 为对抗基准——两者收敛即主测量稳健；zh 收敛补中文适用性软肋（gate 披露 harmbench FNR=1.0、strongreject acc=0.35 后，以响应级共识佐证 dual_judge 可用）。
> 判定不达标即披露「评分器敏感」，该语言主测量仅采用 dual_judge 共识。