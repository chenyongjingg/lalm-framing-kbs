# S15：A_s acoustic 变体（neutral vs styled）对比（GPU1 · 2026-08-14）

## 数据
- neutral_audio=1965, styled_audio=1963（音频仍在生成，本表为当前快照（E4B 完成后重跑获权威））
- 配对结构：key（query,template,N,R）984 个，双变体齐备 984，仅 neutral 0，仅 styled 0

## 按 E_t 分层属性
| E_t | 变体 | n | 均长 | 中位长 | p90 | plot_stall | data_clarify | refusal | other |
|---|---|---|---|---|---|---|---|---|---|
| E_t=0 | neutral | 984 | 336 | 76 | 790 | 4 | 121 | 492 | 367 |
| E_t=0 | styled | 984 | 335 | 68 | 800 | 8 | 117 | 503 | 356 |
| E_t=1 | neutral | 981 | 430 | 574 | 780 | 101 | 28 | 438 | 414 |
| E_t=1 | styled | 979 | 413 | 448 | 779 | 121 | 30 | 404 | 424 |

## 配对长度检验（同 key neutral−styled）
- n_pairs=984, mean_diff=1.18, p=0.235428

## 判读
> A_s 两变体属性差异如上；若配对长度/类别显著差异则 A_s 为系统主效应，主效应模型须含 A_s（设计已含）。
