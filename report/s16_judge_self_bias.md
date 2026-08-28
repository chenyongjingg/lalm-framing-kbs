# S16：judge_small 自评偏差检验（GPU1 · 2026-08-14）

## 动机
judge_small = gemma-4-E2B-it = E2B 响应生成器。dual_judge 依赖 judge_small，若其对自模型输出系统性更宽松 → E2B 侧 dual_judge 有害率被低估。

## 结果
| 集合 | 描述 | judge_small pos_rate | n |
|---|---|---|---|
| A | E4B 响应（他模型, S4 240） | 0.3966 | 240 |
| B | E2B 响应（自模型, 全量 3600） | 0.2538 | 3600（缓存完备=True）|

## judge_small↔judge_big 协议
- E4B 响应: agree=0.7637 (n=237)
- E2B 响应: agree=0.8178 (n=3573)

## 结论
> judge_small pos_rate：他模型输出 0.3966 vs 自模型输出 0.2538 —— 自模型输出上**更低**，方向与'自评宽松'相反；且两集响应内容/语言混杂，此差异不能单独归因于 judge 行为，故不作自评宽松断言。；judge_small↔judge_big 协议在自模型输出上未劣化（A=0.7637, B=0.8178）→ **无证据支持自评宽松偏差**，dual_judge 的 E2B 腿未被自评偏好系统性破坏
