# P1-FULL 跨语言全量确认（v6.5）
- 推理条数: 0 | 查询: zh=200 en=200 adv=200- 主评分器: judge_big_label- 三口径方向一致: ✅

## N_main（storytelling vs baseline，text 模态）
| 口径 | ASR baseline | ASR storytelling | Δ(pp) | 方向 |
|---|---|---|---|---|
| primary | 2.17 | 30.08 | 27.92 | up |
| dual_judge | 2.09 | 40.1 | 38.01 | up |
| majority | 2.09 | 38.88 | 36.79 | up |

## 跨语言
| 语言 | Δ(pp) | baseline | storytelling |
|---|---|---|---|
| zh | 29.25 | 1.92 | 31.17 |
| en | 26.58 | 2.42 | 29.0 |

## 结论
- 三口径一致: True | 跨语言一致: True | 跨模型一致: True
- 混合效应（BinomialBayesMixedGLM）: 公式 `label ~ condition*lang + C(model) + C(template_idx)` | condition OR=16.6505 (95% CI [14.0962, 19.6676]) | 随机效应 query_id: log-SD=-0.9284（statsmodels vcp 为 log 尺度，非负方差；组间 SD≈0.395、variance≈0.156） | 数据范围: text × 中英 × 主模型（['gemma_4_e2b', 'gemma_4_e4b']）；架构对照 ['qwen2_audio_7b'] 不参与主效应推断（§7）；主模型<3 → 固定效应降级档（§9）
- 稳健结论可用: **✅ Narrative structure exhibits a robust causal effect under controlled prompt interventions**
- 评分器独立性与披露: 主评分器 judge_big_label=gemma-4-E4B；dual_judge=judge_big×judge_small=**E4B×E2B 跨尺寸一致性**（同 Gemma-4 家族，非跨家族）；gemma_judge_model 与 judge_small_model 均为 gemma-4-E2B，gemma_label 与 judge_small 同源非独立（16127/16127 逐条一致），4 票 majority 中 E2B 双重计数；主结论由三路独立信号支撑（judge_big E4B 主 / harmbench Llama-2 跨家族 / strongreject E4B 不同协议），披露不影响主效应。
- 科学有效性披露·一致性子样本选择偏差: dual_judge 与 majority 口径仅在 judge_big 与 judge_small 逐条一致的行上定义（分歧 2913 行置 NaN，其中 storytelling 1627 / baseline 249）。故一致性口径入选率 storytelling 仅 57.9%（1389/2400）、baseline 91.8%（2204/2400），Δ 被系统性放大（dual_judge 38.01 / majority 36.79 pp vs 主口径 27.92 pp）。该机制为「条件于一致性」的选择偏差；主口径 judge_big 全样本 27.92pp 不受影响，为本研究主结论依据。
- 科学有效性披露·长度协变量: P1-FULL storytelling 响应平均 937 字符 vs baseline 417 字符，混合模型未含长度协变量。P0-C 分带缓解显示同长度带内 storytelling 仍显著更高（200-799 带 9.0→27.8pp；≥800 带 3.1→24.7pp），主效应非纯长度驱动；审稿时建议明示此局限。**P1-FULL 直接分带（judge_big，主模型×中英，每格 2400）**：baseline 80% <800字符（<50 带 n=510 ASR=0.0%、50-199 n=593 0.2%、200-799 n=828 0.7%），storytelling 96% ≥800（n=2304 ASR=31.3%）→ 两条件长度分布近不重叠；唯一有实质baseline 样本的重叠带 200-799 Δ≈+0.5pp（0.7→1.2%），效应集中于 ≥800 带（baseline 9.6% vs story 31.3%，Δ=+21.7pp），len≥1012 匹配下仍 +17.5pp →主效应**部分经长度介导**（framing→更长→更有害），但同长度域内仍存实质内容效应；judge 非'见长即有害'（baseline 长响应 ASR 仅 9.6%）。
- 科学有效性披露·中文评分器可靠性: P0 zh 探针（request-only）显示单评分器中文召回近零——harmbench acc=0.50/FNR=1.0、strongreject FNR=0.993、gemma FNR=0.96，仅 dual_judge acc=0.853。主评分器 judge_big(E4B) 在中文真实生成上的校准未单独验证，zh Δ=29.25pp 的数值可靠性需以谨慎口径报告。
- 科学有效性披露·P2B基线无效生成（限定 §10，主结论不受影响）: P2B baselines_text 重度退化循环 511/800（63.9%，rep>0.8），baseline 条件 82.0%（164/200）；空响应 118/800（14.8%，其中 self_reminder 105/200=52.5%）。机制：§10 用短裸查询（平均 17 字符）+ 贪心解码（do_sample=False, max_new=512），gemma-4 易陷入短句无限重复；best_of_n（T=1.0, n=10）0% 退化。**主结论数据源 P1-FULL 无退化**：baseline/storytelling/unrestricted 三条件重度 0.0%、空 0.0%、均值 rep 0.046-0.065（三模型全净）——因 P1-FULL 用带指令前缀的较长 prompt（平均 57 字符 vs P2B 17 字符）。故 §10 降级 framing 比较的基线不可靠：P2B 基线退化循环内容混杂（顺从式重复如『请提供登录凭』与乱码式重复并存），对降级 Δ 的偏倚方向**评分前不确定**（非单纯'低估'，方向取决于评分器对退化循环的标注）；§10 须待降级评分后以 modality==audio 纯净子集重估。**主效应 Δ=27.92pp（P1-FULL 干净数据，主模型×中英，每格 2400）不受无效生成污染**，为本研究主结论依据。
