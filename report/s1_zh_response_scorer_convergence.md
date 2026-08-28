# S1 中文响应级评分器收敛验证（GPU1 空闲期测量验证 · 2026-08-12）

- zh 单元: 60（stratifed 抽样，seed=20260812）
- 响应: 120（E4B/E2B 各 60，顺序逐条 greedy 生成）
- 评分器合奏: harmbench, strongreject, judge_big, judge_small, cross_check, shieldgemma
  （论文 4 票 = strongreject + gemma(==judge_small 同模型同 rubric) + judge_big + judge_small；gemma 与 judge_small 合并）
- 零人工标注，无金标准（Dawid-Skene 共识替代真值）

## 双 judge 一致性
- 一致率: 0.7083（n=120）
- 争议率: 0.2917

## 两两一致率（含 Cohen κ）

| 评分器对 | n | 一致率 | κ |
|---|---|---|---|
| harmbench ↔ strongreject | 120 | 0.8167 | 0.633 |
| harmbench ↔ judge_big | 120 | 0.7250 | 0.450 |
| harmbench ↔ judge_small | 118 | 0.7881 | 0.576 |
| harmbench ↔ cross_check | 119 | 0.6723 | 0.348 |
| harmbench ↔ shieldgemma | 120 | 0.4417 | -0.117 |
| strongreject ↔ judge_big | 120 | 0.6417 | 0.351 |
| strongreject ↔ judge_small | 118 | 0.8475 | 0.698 |
| strongreject ↔ cross_check | 119 | 0.7899 | 0.508 |
| strongreject ↔ shieldgemma | 120 | 0.3417 | -0.092 |
| judge_big ↔ judge_small | 118 | 0.7203 | 0.428 |
| judge_big ↔ cross_check | 119 | 0.4454 | 0.146 |
| judge_big ↔ shieldgemma | 120 | 0.6833 | -0.051 |
| judge_small ↔ cross_check | 117 | 0.6581 | 0.334 |
| judge_small ↔ shieldgemma | 118 | 0.4661 | -0.118 |
| cross_check ↔ shieldgemma | 119 | 0.1345 | -0.102 |

## Dawid-Skene 潜类别误差率（无金标准）

| 评分器 | 敏感度 | 特异度 | 加权误分类率 |
|---|---|---|---|
| harmbench | 0.7642 | 0.9694 | 0.1332 |
| strongreject | 0.9637 | 1.0 | 0.0181 |
| judge_big | 0.4298 | 1.0 | 0.2851 |
| judge_small | 0.7488 | 1.0 | 0.1256 |
| cross_check | 1.0 | 0.4859 | 0.257 |
| shieldgemma | 0.013 | 0.8612 | 0.5629 |

- 潜类别 P(有害) = 0.6333

## 关键判据（预登记）

- dual_judge 共识子集: n=85
- dual_judge 共识 vs 异构（cross_check+shieldgemma）一致率: 0.5059
- **判定: 评分器敏感——zh 仅用 dual_judge**

## 说明
> 无金标准：以多评分器共识强度度量 zh 响应测量的稳健性。局限：评分器同族偏差（judge_big/judge_small/strongreject 同属 Gemma-4；shieldgemma 为 Gemma-2 底座安全专用分类器，训练目标异构），Dawid-Skene 假设条件独立。
> 判定不达标即如实披露「评分器敏感」，zh 主测量仅采用 dual_judge 共识。