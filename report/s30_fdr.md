# S30：FDR 多重比较校正（CPU）

- 收集 **84** 个显式 p 值（来源 12 个结果 JSON）
- 原始显著（p<0.05）：**65**
- **q=0.05** BH 校正后存活：**65**
- **q=0.10** BH 校正后存活：**65**

## 校正后丢失的显著结果（诚实披露）

无（原始显著全部通过 q=0.05）

## 覆盖率

| 文件 | 提取 p 值 |
|---|---|
| s10_audio_stalling.json | 2 |
| s10b_clarified_prompt.json | 1 |
| s11_e2b_cross_family.json | 0 |
| s12_text_clarified.json | 1 |
| s12b_modality_stall.json | 0 |
| s13_test_retest.json | 0 |
| s14_len_harm.json | 4 |
| s15_as_variants.json | 1 |
| s16_judge_self_bias.json | 0 |
| s18_dispute_stall.json | 1 |
| s19_e2b_sweep.json | 8 |
| s1_zh_convergence.json | 0 |
| s20_e2b_r_n_interaction.json | 18 |
| s20c_e4b_dual_crossgen.json | 0 |
| s20e_template_strat.json | 0 |
| s22_attack_family.json | 0 |
| s23_e4b_template_strat.json | 0 |
| s24_e4b_attack_family.json | 0 |
| s24_modality_effect.json | 26 |
| s25_hate_interaction.json | 4 |
| s26_modality_family.json | 10 |
| s2_determinism.json | 0 |
| s3_shieldgemma_bench.json | 0 |
| s4_convergence_full.json | 0 |
| s5_zh_audio.json | 0 |
| s7_ds_simulation.json | 8 |
| s9_cross_family.json | 0 |

## 披露
> BH 假定独立检验；本结果存在天然相关（同一实验多维度），相关结构未调整，属保守-激进边界问题，如实披露。仅含 CI 无显式 p 值的结果未纳入校正。