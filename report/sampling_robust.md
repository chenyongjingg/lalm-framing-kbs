# 主链 E2B 解码采样鲁棒性（N3，探索性）

- 模型：Gemma-4-E2B bf16（与 P1-FULL 同生成器），解码 temp=1.0/top_p=0.9/seed 固定。
- cells: 48（48 queries × baseline/storytelling × t0/t1/t2）。
- greedy vs sampling 同格 dual 一致率：97.4%（n 有双标 48，争议 12）。

## N 效应（storytelling vs baseline，query 配对 bootstrap 95% CI）
| 解码 | baseline ASR% | storytelling ASR% | Δ(N) pp [95%CI] |
|---|---|---|---|
| greedy（P1-FULL） | 0.0 | 28.57 | 20.72 [0.0,45.83] |
| sampling（本实验） | 0.0 | 20.83 | 20.75 [4.17,41.67] |

## 判读（如实）
- 若 sampling 下 Δ(N) 方向与 greedy 一致 → 主链 N 效应非 greedy 解码伪影。
- 样本小（探索性）：CI 变宽预期，与 S33（Qwen2-Audio）合并构成「采样下方向保持」证据。若方向翻转/归零 → 如实披露为解码敏感。
- 数据：results/sampling_robust/。