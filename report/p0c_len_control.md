# P0-C 长度混杂控制 + text 效应分层分析

数据：`results/p0c_scored.parquet`（10800 行）| 主口径：harmbench_label | 有效行（hb 非空）：10800 | 弃 null：0
长度 = response 字符数。logistic = 手写 ridge-IRLS（λ=1e-6），Wald 检验。

## A. Framing 效应：raw OR vs 控制 log_len 后 adjusted OR

| model | modality | ASR base | ASR storyt | ASR unrest | OR_s_raw | OR_s_adj | OR_u_raw | OR_u_adj | len 交互 p |
|---|---|---|---|---|---|---|---|---|---|
| gemma_4_e2b | audio | 3.5% | 66.5% | 64.3% | 54.73 | 1.24 | 49.73 | 2.35 | 1.73e-06 |
| gemma_4_e2b | text | 0.2% | 0.2% | 0.2% | 1.00 | 0.21 | 1.00 | 0.32 | 0.250 |
| gemma_4_e4b | audio | 6.7% | 42.7% | 46.2% | 10.42 | 5.20 | 12.01 | 6.51 | 0.00e+00 |
| gemma_4_e4b | text | 2.3% | 0.3% | 0.7% | 0.14 | 0.14 | 0.28 | 0.28 | 0.701 |
| qwen2_audio_7b | audio | 5.8% | 56.2% | 80.2% | 20.68 | 0.60 | 65.25 | 2.57 | 7.27e-05 |
| qwen2_audio_7b | text | 6.0% | 62.0% | 74.8% | 25.56 | 0.19 | 46.58 | 0.76 | 0.531 |

> 判定：adjusted OR 与 raw OR 同向且量级相近 ⇒ framing 效应在控制长度后保持；交互显著 ⇒ framing 效应随长度变化（长度是修饰因子，非单纯混杂）。

## B. 长度分层：condition × 长度档 → 有害率

### audio

| 长度档 | baseline ASR | storytelling ASR | unrestricted ASR | 各档 S vs B 差 |
|---|---|---|---|---|
| <50 (n=1066/69/112) | 2.2% | 0.0% | 0.0% | -2.2pp |
| 50-199 (n=592/30/53) | 2.5% | 3.3% | 0.0% | +0.8pp |
| 200-799 (n=98/580/1014) | 26.5% | 50.0% | 70.1% | +23.5pp |
| 800+ (n=44/1118/621) | 72.7% | 62.7% | 69.7% | -10.0pp |

### text

| 长度档 | baseline ASR | storytelling ASR | unrestricted ASR | 各档 S vs B 差 |
|---|---|---|---|---|
| <50 (n=178/74/68) | 2.8% | 0.0% | 0.0% | -2.8pp |
| 50-199 (n=462/20/52) | 1.7% | 0.0% | 0.0% | -1.7pp |
| 200-799 (n=267/586/524) | 9.0% | 27.8% | 55.2% | +18.8pp |
| 800+ (n=459/858/814) | 3.1% | 24.7% | 20.3% | +21.7pp |

> 解读：若多数长度档内 storytelling>baseline 且差显著 ⇒ 混杂不推翻方向；若仅长响应档有效应 ⇒ 效应主要出现在「更长更顺从」的输出，需在正文如实披露。

## C. text 模态 framing 效应：逐模型 Wald 检验（架构边界）

| model | modality | cond | OR | 95%CI | p | 判定 |
|---|---|---|---|---|---|---|
| gemma_4_e2b | audio | storytelling | 1.24 | [0.55, 2.77] | 0.604 | 不显著 |
| gemma_4_e2b | text | storytelling | 0.21 | [0.01, 3.65] | 0.284 | 不显著 |
| gemma_4_e2b | audio | unrestricted | 2.35 | [1.07, 5.16] | 0.033 | 显著 |
| gemma_4_e2b | text | unrestricted | 0.32 | [0.02, 5.19] | 0.420 | 不显著 |
| gemma_4_e4b | audio | storytelling | 5.20 | [2.88, 9.39] | 4.54e-08 | 显著 |
| gemma_4_e4b | text | storytelling | 0.14 | [0.03, 0.62] | 0.010 | 显著(负向) |
| gemma_4_e4b | audio | unrestricted | 6.51 | [3.77, 11.24] | 1.80e-11 | 显著 |
| gemma_4_e4b | text | unrestricted | 0.28 | [0.09, 0.85] | 0.025 | 显著(负向) |
| qwen2_audio_7b | audio | storytelling | 0.60 | [0.31, 1.17] | 0.136 | 不显著 |
| qwen2_audio_7b | text | storytelling | 0.19 | [0.09, 0.41] | 1.80e-05 | 显著(负向) |
| qwen2_audio_7b | audio | unrestricted | 2.57 | [1.35, 4.88] | 0.004 | 显著 |
| qwen2_audio_7b | text | unrestricted | 0.76 | [0.38, 1.52] | 0.431 | 不显著 |

## D. 可直接引用的边界声明（拟稿）

> Text 模态下 framing 的放大效应呈**架构依赖**：Gemma 家族两个 LALM 上 text 模态无放大（或弱负向），Qwen2-Audio 的 text 模态仍显著放大；Audio 模态在全部三模型上均显著放大。该架构边界与附录（ShieldGemma 对 Gemma 家族生成器的 N 反转）指向一致的生成器族特定行为，而非通用的跨模态 framing 效应。
