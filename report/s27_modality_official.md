# S27：模态效应 × 官方基准评分器稳健性（GPU1 · 2026-08-15）

**状态**：权威（E4B 全量 + S20d/S25 官方腿全覆盖）｜ 口径：strongreject / harmbench 二分类

## 1. 单模态有害率（官方评分器）
| 评分器 | text | neutral_audio | styled_audio |
|---|---|---|---|
| strongreject | 0.4683 | 0.5525 | 0.5456 |
| harmbench | 0.2806 | 0.3469 | 0.3417 |

## 2. 模态效应（Δ = pos(au) − pos(text)，query 聚类 95%CI）
| 评分器 | 对比 | n0 | n1 | pos0 | pos1 | Δ | 95%CI | Fisher OR(p) |
|---|---|---|---|---|---|---|---|---|
| strongreject | text_vs_neutral_audio | 3600 | 3600 | 0.4683 | 0.5525 | 0.0842 | [0.0536, 0.1158] | 1.4016(0.0) |
| strongreject | text_vs_styled_audio | 3600 | 3600 | 0.4683 | 0.5456 | 0.0772 | [0.0467, 0.1108] | 1.3628(0.0) |
| strongreject | neutral_audio_vs_styled_audio | 3600 | 3600 | 0.5525 | 0.5456 | -0.0069 | [-0.0195, 0.005] | 0.9723(0.56975) |
| strongreject | text_vs_all_audio | 3600 | 7200 | 0.4683 | 0.549 | 0.0807 | [0.0512, 0.1131] | 1.3821(0.0) |
| harmbench | text_vs_neutral_audio | 3600 | 3600 | 0.2806 | 0.3469 | 0.0664 | [0.0397, 0.0942] | 1.3623(0.0) |
| harmbench | text_vs_styled_audio | 3600 | 3600 | 0.2806 | 0.3417 | 0.0611 | [0.0319, 0.0878] | 1.3309(0.0) |
| harmbench | neutral_audio_vs_styled_audio | 3600 | 3600 | 0.3469 | 0.3417 | -0.0053 | [-0.0181, 0.0069] | 0.9769(0.655269) |
| harmbench | text_vs_all_audio | 3600 | 7200 | 0.2806 | 0.3443 | 0.0637 | [0.0378, 0.0896] | 1.3465(0.0) |

## 3. 覆盖率
| 评分器 | text n | text cov | audio n | audio cov |
|---|---|---|---|---|
| strongreject | 3600 | 1.0 | 7200 | 1.0 |
| harmbench | 3600 | 1.0 | 7200 | 1.0 |

## 判读
> S24 模态效应（dual/qwen32/judge）在官方基准评分器下是否稳健（符号、量级、显著性）；若 strongreject/harmbench 亦显示 audio>>text，则为多模态放大声明提供跨评分器证据。
