# S3 异构安全分类器公开基准验证（GPU1 空闲期测量验证 · 2026-08-12）

- 基准: 同 gate 的 602 行人工标注公开基准（jalmbench）
- 可标注行: 602
- 成员: shieldgemma（Llama-Guard-3-8B / WildGuard gated 不可用）
  ShieldGemma 披露：基于 Gemma-2-9B 安全专用训练，与论文 Gemma-4 评分器不同训练目标，但同 Google 家族。

| 评分器 | acc | recall | FPR | FNR | n | n_dropped |
|---|---|---|---|---|---|---|
| shieldgemma | 0.7857 | 0.8388 | 0.2584 | 0.1612 | 602 | 0 |

## 与既有 cross_validation 对照（gate P0_scorers.json）
> judge_big acc=0.8555, harmbench acc=0.8239, strongreject acc=0.7342, cross_check acc=0.6312（602 行）。ShieldGemma 数字即上述，直接可比，扩展现有 cross_validation 块；不修改 gates/（单权威纪律）。

## 说明
> 复用 validate_single_scorer 含分布塌缩检测：单类 >95% 即判定解析塌缩，评分器标记不可用而非静默产出——防幻觉纪律 #1。