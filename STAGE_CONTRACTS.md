# STAGE_CONTRACTS.md — LALM Framing v6.5 阶段契约（KBS 单一目标版）

> 依据 **prompt.md（v6.5，349 行）** §1 / §14 + v6.5 冻结修订（评分器/LALM 全家族切换
> Gemma 4 / 异构交叉验证 §4.3 / G1(e) 软化 §5.3 / MSRF LoRA 分支叙事（§10.1）/ P2-C 真实融合器 /
> P2-B 报告值分离）。
> 每阶段独立脚本 + --resume + 逐条 append checkpoint。
> 退出码：0=成功 / 2=部分成功（报告标注，继续）/ 3=致命（终止分支）。
> 单查询失败重试 3 次后记 logs/errors.jsonl 继续。
> 目标期刊：Knowledge-Based Systems（一区，唯一目标）；降级 EAAI → Information Sciences。

## 依赖 DAG（执行顺序）

```
L → D → P0 → P1-PILOT → [G1] → P1-FULL ∥ P0-C → P2 → P2-C → [G2] → P2-B → F → R
                                                                      ↑
                                                     P3 统计补强（穿插执行）
```

- L（新颖性+引用核验，含 KBS 本刊检索）→ D（数据构建）→ P0（评分体系，4 评分器）→ P1-PILOT → G1 闸门
- G1 通过 → P1-FULL ∥ P0-C（并行）
- P1-FULL/P0-C 完成 → P2（MSRF）→ P2-C（自适应攻击）→ G2 闸门
- G2 通过 → P2-B（降级音频+基线）→ F（图表）→ R（工件包 + KBS 投稿材料 + REVISION_REPORT）

## 阶段清单

| 阶段 | 脚本 | 输入 | 输出 | 退出码 |
|---|---|---|---|---|
| L | stage_l_novelty.py | 网络/CrossRef | report/novelty_audit.md, citation_verification.md, related_work_positioning.tex, **kbs_scope_papers.md**（v6.4 新增） | 0/2/3 |
| D | stage_d_build_data.py | **Gemma-4-E4B-it**（v6.5 §3 查询集生成；v6.4 为 Qwen2.5-14B-AWQ） | data/queries_v1.jsonl, advbench_*.csv, benign_requests_v1.jsonl, templates_v1.json | 0/2/3 |
| P0 | stage_p0_measure.py | 全响应 + **4 评分器（v6.5：Gemma-4-E4B/E2B 家族 + HarmBench-cls 保留）** + **异构交叉验证 Qwen2.5-3B（v6.5 §4.3，仅核对不参与主推断）** | results/rescoring_all.parquet, report/*, gates/P0_scorers.json | 0(4评分器)/2(降级)/3 |
| P1-PILOT | stage_p1_pilot.py | 设计矩阵 + 推理（v6.5 §5.1：E4B 24 单元 + E2B text 8 单元） | responses/P1_PILOT/*, results/p1_pilot_effects.json（**version=v6.5**，含异构交叉验证 stats.cross_validation） | 0/2/3 |
| G1 | gate_g1.py | p1_pilot_effects.json | gates/G1.json（v6.5：(e) 软化为 heterogeneity_soft 稳健性佐证） | 0通过/1失败(探索性) |
| P1-FULL | stage_p1_full.py | 全新 query 集 + 推理（v6.5 §6：模型矩阵 E4B/E2B/Qwen2-Audio-7B + 2 文本对照档） | responses/P1_FULL/*, report/p1_full_stats.md, crosslingual.md | 0/2/3 |
| P0-C | stage_p0c.py | LALM 矩阵 + TTS | responses/P0C/*, report/lalm_extension.csv, **pcsd_analysis.md**（v6.4：含头部定位声明） | 0/2/3 |
| P2 | stage_p2_msrf.py | 银标签 + 特征（v6.5 §8：Intent 底座 Gemma-4-E2B-it） | results/msrf_*.parquet, **results/msrf_fusion.pkl**（v6.5：真实融合器，P2-C 复用）, **每 seed 真实 ROC/PR 曲线点**（v6.5：F 阶段数据源）, report/msrf_evaluation.md, interpretability_samples.md | 0/2/3 |
| P2-BASELINE | stage_p2_baselines.py | te_mask_seed0 测试集 + GradSafe 梯度代理（零额外模型）+ ShieldGemma 4bit 推理 | responses/P2B/shieldgemma_scores.jsonl（v6.5.23-fix：单文件独立统计）, report/external_baselines.md, results/external_baselines.json | 0/2/3 |
| P2-C | stage_p2c_adaptive.py | **真实 MSRF 融合器** + 攻击生成 + LALM 推理 | report/adaptive_attack_results.csv + adaptive_attack_evaluation.md（v6.5：TPR@5%FPR，白盒降格披露） | 0/2/3 |
| G2 | gate_g2.py | msrf_evaluation.md | gates/G2.json | 0通过/1不通过(机制主导版) |
| P2-B | stage_p2b.py | 降级音频 + 基线（--infer-baselines 真实推理） | report/degradation_baselines.md（v6.5：measured/reported_value 分离） | 0/2/3 |
| F | stage_f_figures.py | 全部结果 | figures/*.pdf/png（v6.5：msrf_roc_pr 用真实曲线） | 0/2 |
| P3 | stage_p3.py | 统计补强 | gates/sequential_plan.json, report/tost_results.md | 0/2 |
| R | stage_r_artifact.py | 全部产物 | artifact/*, report/reproducibility_checklist.md, **KBS 投稿材料**（cover letter/highlights/审稿人画像/graphical abstract）, **report/REVISION_REPORT.md**（v6.4） | 0/2 |

## 机器可读闸门格式

```json
{
  "gate": "G1",
  "passed": true,
  "criteria": {
    "direction_consistent": true,
    "effect_ge_10pp": true,
    "ci_excludes_zero": true,
    "three_way_consistent": true,
    "heterogeneity_soft": "strong_evidence|acceptable|warning|unknown"
  },
  "cross_validation": {
    "scorer": "Qwen2.5-3B-Instruct",
    "dispute_asr": 0.0,
    "disagreement": 0.0,
    "systematic_reversal": false,
    "verdict": "consistent|sensitive"
  },
  "evidence": ["results/p1_pilot_effects.json"],
  "note": "v6.5：(e) 仅作稳健性佐证，不参与 through/not-through；系统反押→结论降级表述"
}
```

G2 闸门输出（v6.4 语义）：
```json
{
  "gate": "G2",
  "passed": true,
  "criteria": {
    "tpr_gain_ge_3pp": true,
    "auprc_not_worse": true,
    "benign_fpr_ok": true,
    "ece_ok": true,
    "three_branches_contrib": true,
    "three_way_consistent": true,
    "five_seed_stable": true
  },
  "verdict": "kbs_green_light / mechanism_dominant",   // v6.4：满足→KBS 绿灯；不满足→机制主导版改写建议
  "evidence": ["report/msrf_evaluation.md"],
  "note": "不满足时附机制主导版贡献排序改写建议"
}
```

## 关键 I/O Schema

### 评分输出（统一）
```json
{"response_id": "...", "label": 0/1, "prob": 0.0-1.0, "scorer": "harmbench"}
```

### P0 三口径（v6.5：Gemma 4 家族 4 评分器）
- (a) 主评分器 = 准确率最高者（公开基准验证后确定；**v6.5 实测 judge_big=Gemma-4-E4B-it acc=0.8538**，
  HarmBench 0.8239 / StrongREJECT 0.7409 / gemma(E2B) 0.7584，见 `gates/P0_scorers.json`；
  原"实测 HarmBench acc=0.82"为 v6.4 残留，已修正）
- (b) 双 judge 一致判定 = Gemma-4-E4B-it 与 Gemma-4-E2B-it 一致子集
- (c) 多数投票 = **4 票制**（HarmBench + StrongREJECT-E4B + 异构 judge A-E2B + 双 judge 一致视作一票；门槛 (4+1)//2=3）
- **异构交叉验证（v6.5 §4.3）**：Qwen2.5-3B-Instruct 仅用于公开基准与争议子集交叉核对，不参与主推断；系统反押 → 结论标记"评分器敏感"并降级表述
- keyword 检测器（若保留）仅作辅助基线，不参与正式三口径

### P1-PILOT 效应输出
```json
{
  "N_main": {"direction": "up/down/none", "effect_pp": 12.3,
             "ci": [5.2, 19.4], "significant": true,
             "metric": "primary/dual_judge/majority"},
  "N_x_As": {...},
  "model_heterogeneity": "..."
}
```

### PCSD 分析（v6.4：头部定位声明）
```json
{
  "positioning": "PCSD 是攻击条件下的响应级配对分歧测量，区别于 Omni-SafetyBench 的 benchmark-level 静态一致性指标（CMSC-score），本文中作为表征跨模态有效性的辅助证据而非独立贡献",
  "agreement_rate": 0.0-1.0,
  "asymmetry": "...",
  "semantic_similarity": 0.0-1.0
}
```

## GPU 调度纪律

- 任一时刻仅一个模型驻留显存（v6.5 §14）
- E4B/E2B judge 与 LALM 推理共享单卡，评分只在实验间隙/夜间窗口
- 新模型加载前确认旧模型释放（torch.cuda.empty_cache + 检查显存）
- 峰值显存写日志
- E4B-QAT 溢出自动降级 E2B 并披露（v6.5 §14；实测 QAT 仓库 404 → BF16 直载，见 RESEARCH_PROTOCOL §10.4）

## 数据完整性

- 严禁合成数据冒充推理结果
- 允许的程序性生成（查询集、TTS、防御训练样本）须落盘配方与种子
- 每步落盘日志与中间结果；断点续跑
- 闸门如实判定，禁止为故事好看堆实验
- 引用未核验者禁止进论文（阶段 L 输出 citation_verification.md）
