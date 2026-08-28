# P1-FULL 主链补开源分类器腿（ShieldGemma/WildGuard）

- 数据：P1-FULL 主 scope 文本 7200 行（zh/en × e4b/e2b），与 P1-FULL 主效应口径一致。
- 评分：ShieldGemma(9B,4bit,response_only) / WildGuard(Mistral-7B,4bit,harmful_response)，与 P2C-4 攻击集同口径。
- dual_judge 权威口径行（双判一致）: 5334。

## 总体检出率（% 判有害）
| 评分器 | 主链检出率% |
|---|---|
| dual_judge 共识 | 18.0 |
| ShieldGemma(unsafe) | 1.6 |
| WildGuard(harmful_response) | 9.0 |

## 与 dual_judge 一致性（双判一致行）
- ShieldGemma vs dual: 一致率 82.9%，κ=0.105
- WildGuard vs dual: 一致率 88.3%，κ=0.498

## 按 condition
| condition | n | dual% | ShieldGemma% | WildGuard% |
|---|---|---|---|---|
| baseline | 2400 | 2.1 | 0.8 | 1.1 |
| storytelling | 2400 | 40.1 | 2.5 | 14.2 |
| unrestricted | 2400 | 20.6 | 1.4 | 11.6 |

## 按 model
| model | n | dual% | ShieldGemma% | WildGuard% |
|---|---|---|---|---|
| gemma_4_e4b | 3600 | 17.8 | 1.1 | 7.2 |
| gemma_4_e2b | 3600 | 18.2 | 2.1 | 10.7 |

## 判读（如实）
- 若开源分类器检出率与 dual_judge 同向且一致率高 → 主效应测量对评分器选择稳健。
- 若检出率显著偏离（过高=误报、过低=漏报）→ 如实披露评分器口径差异；P2C-4 已披露开源分类器高误报（WildGuard 全行 22.2% vs hb ASR 1.8%）。
- 数据：results/p1full_decay_scores/。