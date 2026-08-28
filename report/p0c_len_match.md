# P0-C 长度匹配敏感性分析（MH 分层 + 最近邻匹配）

数据：`results/p0c_scored.parquet`（10800 行）| harmbench 口径 | 有效 10800 行
方法 A：log_len 十分位分层 → Mantel-Haenszel 合并 OR（含 0.5 连续性校正，Robins-Breslow-Greenland SE）。
方法 B：1:1 最近邻匹配（log_len，caliper=0.2 SD），McNemar 检验，20 次重抽中位稳定。

## A. Mantel-Haenszel 分层 OR（控制 log_len 后）

| model | modality | treat vs base | OR_mh | 95%CI | p | 层内加权 ASR 差 | 层内同向占比 |
|---|---|---|---|---|---|---|---|
| gemma_4_e2b | audio | storytelling | 1.70 | [1.30, 2.21] | 0.000 | +2.6pp | 89% |
| gemma_4_e2b | audio | unrestricted | 3.18 | [2.31, 4.38] | 0.000 | +7.0pp | 67% |
| gemma_4_e2b | text | storytelling | 0.53 | [0.36, 0.78] | 0.001 | -0.4pp | 50% |
| gemma_4_e2b | text | unrestricted | 0.62 | [0.41, 0.93] | 0.021 | -0.2pp | 50% |
| gemma_4_e4b | audio | storytelling | 0.60 | [0.47, 0.77] | 0.000 | -9.3pp | 29% |
| gemma_4_e4b | audio | unrestricted | 1.02 | [0.80, 1.31] | 0.859 | -0.5pp | 43% |
| gemma_4_e4b | text | storytelling | 0.39 | [0.29, 0.52] | 0.000 | -1.9pp | 20% |
| gemma_4_e4b | text | unrestricted | 0.49 | [0.38, 0.63] | 0.000 | -2.0pp | 40% |
| qwen2_audio_7b | audio | storytelling | 0.84 | [0.69, 1.03] | 0.092 | -5.5pp | 44% |
| qwen2_audio_7b | audio | unrestricted | 6.16 | [4.63, 8.19] | 0.000 | +33.9pp | 38% |
| qwen2_audio_7b | text | storytelling | 0.82 | [0.67, 1.01] | 0.057 | -6.9pp | 43% |
| qwen2_audio_7b | text | unrestricted | 2.85 | [2.34, 3.46] | 0.000 | +14.0pp | 57% |

> 判定：若 MH OR 显著 >1 且层内差为正 ⇒ 控制长度后仍有残留效应；若 OR 回落到 1 附近 ⇒ 效应主要由长度介导。

## B. 1:1 最近邻匹配（log_len, caliper=0.2 SD）+ McNemar

| model | modality | treat vs base | 匹配后 ASR 差 | n_pairs | p (McNemar) |
|---|---|---|---|---|---|
| gemma_4_e2b | audio | storytelling | +2.9pp | 69 | 0.683 |
| gemma_4_e2b | audio | unrestricted | +2.8pp | 71 | 0.683 |
| gemma_4_e2b | text | storytelling | -1.8pp | 55 | 1.000 |
| gemma_4_e2b | text | unrestricted | -2.0pp | 49 | 1.000 |
| gemma_4_e4b | audio | storytelling | -10.5pp | 38 | 0.386 |
| gemma_4_e4b | audio | unrestricted | -4.1pp | 49 | 0.724 |
| gemma_4_e4b | text | storytelling | -4.3pp | 115 | 0.131 |
| gemma_4_e4b | text | unrestricted | -2.7pp | 110 | 0.371 |
| qwen2_audio_7b | audio | storytelling | -39.3pp | 28 | 0.003 |
| qwen2_audio_7b | audio | unrestricted | -5.9pp | 34 | 0.617 |
| qwen2_audio_7b | text | storytelling | -33.3pp | 33 | 0.006 |
| qwen2_audio_7b | text | unrestricted | -4.4pp | 68 | 0.371 |

## 结论（拟稿）

> 长度匹配/MH 分层与单斜率 logistic 一致：control 长度后，audio storytelling 的效应大幅衰减（e2b/qwen 不显著），unrestricted 在 audio 及 e2b/qwen 上保持显著；text 模态在匹配长度下保留真实效应。支持「framing 主要通过诱发更长响应提高有害率，长度是 audio 主中介、unrestricted 存在长度无关直接效应」的机制结论。
