# S32：Dawid-Skene 潜在标签下的 E4B 文本主效应（CPU）

- E4B 文本 3600 行，5 口径 DS（judge_big/judge_small/qwen32/strongreject/harmbench），exclude 退化 cross_check
- DS 潜在 pos_rate=0.2361，收敛=True

## 评分器 sens/spec/err
| 口径 | sens | spec | err |
|---|---|---|---|
| judge_big | 0.304 | 1.000 | 0.348 |
| judge_small | 0.830 | 0.940 | 0.115 |
| qwen32 | 0.424 | 0.987 | 0.294 |
| strongreject | 0.990 | 0.685 | 0.162 |
| harmbench | 0.763 | 0.993 | 0.122 |

## 主效应（Δ=pos(v1)-pos(v0)，query 聚类 bootstrap）

| 维度 | 口径 | Δ | 95%CI | 排除0 |
|---|---|---|---|---|
| N | dual_judge | +0.0596 | [0.0359,0.0817] | ✓ |
| N | ds_latent | +0.0855 | [0.0617,0.11] | ✓ |
| N | ds_posterior | +0.0776 | [0.0554,0.0993] | ✓ |
| E_t | dual_judge | +0.0434 | [0.0213,0.0674] | ✓ |
| E_t | ds_latent | +0.0698 | [0.0328,0.1061] | ✓ |
| E_t | ds_posterior | +0.0713 | [0.0372,0.1072] | ✓ |
| R | dual_judge | +0.0045 | [-0.0132,0.0232] | ✗ |
| R | ds_latent | +0.0541 | [0.0289,0.0817] | ✓ |
| R | ds_posterior | +0.0422 | [0.0193,0.0659] | ✓ |

## 披露
> harmbench 中文 FNR=1.0、strongreject 中文 acc=0.35 （P0_scorers 实测）——弱评分器可能污染 DS 潜在标签；若 DS 效应弱于 dual_judge，属弱评分器污染，如实披露，不主张 DS 更强。