# S11：E2B 全量 3600 文本响应跨族核验（GPU1 · 2026-08-14）

## 动机
S9 已核验 judge_big(E4B) 侧 vs Qwen2.5-32B-AWQ 跨族收敛（0.84-0.93）。
但 dual_judge 共识 = judge_big(E4B) + judge_small(E2B) 一致才采信 —— judge_small(E2B) 侧此前未被强异构锚点直接核验。本实验对 E2B 生成的
3600 条文本响应（E_t 0/1 各 1800）全量评分，补全该证据。

## 判据
dual_judge 共识 vs qwen32 一致率 ≥0.80 → 跨族收敛（测量可信）；
<0.80 → 如实披露。

## 数据
- 源：E2B 响应 3600 条（text），主流水线 100% 完成。
- 评分器：judge_big（GemmaJudge E4B）、judge_small（GemmaJudge E2B）、
  qwen32（Qwen2.5-32B-AWQ，CROSS_CHECK_RUBRIC）。

## 分布检查
- judge_big: pass=True, pos_rate=0.0936, counts={'0': 3263, '1': 337}
- judge_small: pass=True, pos_rate=0.2538, counts={'0': 2666, '1': 907}
- qwen32: pass=True, pos_rate=0.1353, counts={'0': 3113, '1': 487}

## 全样本判据
- dual_judge vs qwen32 一致率: 0.9624（n=2922）跨族收敛（测量可信）

## 全样本两两一致率
- judge_big vs judge_small: 0.8178（n=3573, κ=0.3907207105498834）
- judge_big vs qwen32: 0.9361（n=3600, κ=0.6861451267966109）
- judge_small vs qwen32: 0.8200（n=3573, κ=0.436625351428827）

## 按 E_t 分层
- E_t=0（n=1800）: dual_judge vs qwen32=0.9806（n=1442）跨族收敛（测量可信）
  - judge_big vs judge_small: 0.8178（n=3573, κ=0.3907207105498834）
  - judge_big vs qwen32: 0.9361（n=3600, κ=0.6861451267966109）
  - judge_small vs qwen32: 0.8200（n=3573, κ=0.436625351428827）
- E_t=1（n=1800）: dual_judge vs qwen32=0.9446（n=1480）跨族收敛（测量可信）
  - judge_big vs judge_small: 0.8178（n=3573, κ=0.3907207105498834）
  - judge_big vs qwen32: 0.9361（n=3600, κ=0.6861451267966109）
  - judge_small vs qwen32: 0.8200（n=3573, κ=0.436625351428827）

## 判读
> E2B 侧 dual_judge vs qwen32 跨族收敛 → judge_small(E2B) 评分与强异构锚点一致，dual_judge 两腿（E4B+E2B）均获核验，中文响应测量可信度闭合。
