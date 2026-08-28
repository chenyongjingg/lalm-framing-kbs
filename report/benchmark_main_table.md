# LALM-Frame Benchmark 主表 + V(model) 易感性指数

- 数据：P1-FULL 16200 行；主 scope text×zh/en×e4b/e2b（每格 2400）
- 标签：dual_judge 共识（双 judge 一致）；CI=query 簇 bootstrap B=2000

## 总体各条件有害率（主 scope）
| condition | ASR% | 95% CI | n_dual |
|---|---|---|---|
| baseline | 2.087 | 1.568–2.773 | 2204 |
| storytelling | 40.101 | 37.554–42.702 | 1389 |
| unrestricted | 20.563 | 18.73–22.525 | 1741 |

## 按 模型×语言 效应（storytelling/unrestricted vs baseline）
| model | lang | Δstory(pp) | Δstory CI | OR story | Δunrest | OR unrest |
|---|---|---|---|---|---|---|
| gemma_4_e4b | zh | 38.381 | 31.998–45.05 | 27.208 | 18.638 | 9.195 |
| gemma_4_e4b | en | 40.16 | 33.433–46.906 | 28.105 | 19.215 | 13.228 |
| gemma_4_e2b | zh | 29.879 | 24.377–35.993 | 24.509 | 16.015 | 10.267 |
| gemma_4_e2b | en | 37.466 | 30.154–44.894 | 16.104 | 24.429 | 10.883 |

## V(model) 易感性指数（跨条件×语言均值）
| model | V_pp (mean Δ) | V_or (geo-mean OR) |
|---|---|---|
| gemma_4_e4b | 29.099 | 17.464 |
| gemma_4_e2b | 26.947 | 14.491 |

## 架构对照 qwen2_audio_7b（text，排除主推断）
- baseline: ASR=5.281% (n_dual=1155)
- storytelling: ASR=38.816% (n_dual=997)
- unrestricted: ASR=34.971% (n_dual=1018)

## OOD 域（AdvBench 锚定集）
- baseline: ASR=0.168% (n_dual=1193)
- storytelling: ASR=31.009% (n_dual=803)
- unrestricted: ASR=13.174% (n_dual=1002)
- storytelling Δ=38.038pp CI=[np.float64(32.279), np.float64(43.699)] OR=363.319 CI=[130.54, 876.214]

## 判读
- V(model) 零训练成本定义 LALM 对 framing 的易感性；主 scope 全正且显著。
- 良性查询对照（S40/S40b）应显示各条件 ≈0 → framing 特异性放大恶意查询。
- qwen2_audio_7b 与 adv OOD 须独立披露，不并入主 V。