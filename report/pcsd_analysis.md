# PCSD 配对分歧分析（v6.5）
> **定位声明**：PCSD 是攻击条件下的响应级配对分歧测量，区别于 Omni-SafetyBench 的 benchmark-level 静态一致性指标（CMSC-score），本文中作为表征跨模态 有效性的辅助证据而非独立贡献。
- 定义：同一 query 的 text/audio 响应的长度比 + 词汇重叠率（0-1，高=低分歧）
- PCSD 是响应级配对分歧测量（攻击条件下），非 benchmark 级静态一致性
- **测量集重叠披露（LM1-fix）**：评测查询已按 query_id 排除测量集（PILOT 冻结集 + FULL 用集，来源 ['p1_pilot_queries_full.json', 'p1_full_queries_full.json']）；池 400→200。排除后不足配额时回退全池并如实标注。

| 模型 | PCSD 一致率均值 | 标准差 | n |
|---|---|---|---|
| gemma_4_e2b | 0.281 | 0.195 | 680 |
| gemma_4_e4b | 0.248 | 0.179 | 1679 |
| qwen2_audio_7b | 0.404 | 0.137 | 1800 |

## 分歧方向不对称性（harmbench ASR）
- gemma_4_e4b: text ASR=1.1% | audio ASR=31.8% | Δ(audio-text)=+0.307
- gemma_4_e2b: text ASR=0.2% | audio ASR=44.8% | Δ(audio-text)=+0.446
- qwen2_audio_7b: text ASR=47.6% | audio ASR=47.4% | Δ(audio-text)=-0.002

## 响应级配对安全判定一致率（§7.3）
- gemma_4_e4b: 配对判定一致率=0.672 (n=1800)
- gemma_4_e2b: 配对判定一致率=0.552 (n=1800)
- qwen2_audio_7b: 配对判定一致率=0.793 (n=1800)

## framing 操纵对分歧的影响（§7.3）
| 条件 | cmsc 均值(高=低分歧) | 标准差 | n |
|---|---|---|---|
| baseline | 0.255 | 0.234 | 1197 |
| storytelling | 0.364 | 0.138 | 1519 |
| unrestricted | 0.330 | 0.150 | 1443 |
- storytelling 分歧 - baseline 分歧 = -0.109（>0 表示 framing 系统性放大分歧）
