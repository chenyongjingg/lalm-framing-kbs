# S36：强制解码协议补全（S17/S33 全量 + 剩余 null 打标）

- 日期：2026-08-16；方法：同 S35（首 token argmax(logits('0'),logits('1'))）
- 结果 json：`results/gpu1_pipeline/s36_forced_complete.json`
- 标签文件：`results/gpu1_pipeline/s36_forced_complete_labels.jsonl`

## 1. 全量格 协议稳定性（forced vs freegen）

| scope | n | null | 一致率 | κ | Spearman | 一致格|m|med | 分歧格|m|med |
|---|---|---|---|---|---|---|---|
| s17_e4b_audio | 7200 | 23 | 0.9986 | 0.9974 | 0.9907 | 7.7500 | 4.7969 |
| s17_e4b_text | 3600 | 19 | 0.9897 | 0.9837 | 0.9543 | 7.6250 | 4.7812 |
| s33_hetero_audio | 344 | 3 | 0.9971 | 0.9944 | 0.9967 | 7.6562 | 4.7031 |

分层 s17_e4b_audio：  E_t=0=3594 | 0.999 | 0.998 | 0.991  E_t=1=3583 | 0.998 | 0.997 | 0.971  N=0=3582 | 0.998 | 0.996 | 0.982  N=1=3595 | 1.000 | 1.000 | 0.997
分层 s17_e4b_text：  E_t=0=1789 | 0.985 | 0.977 | 0.944  E_t=1=1792 | 0.994 | 0.991 | 0.953  N=0=1783 | 0.983 | 0.974 | 0.955  N=1=1798 | 0.997 | 0.995 | 0.969
分层 s33_hetero_audio：  E_t=0=172 | 1.000 | 1.000 | 1.000  E_t=1=169 | 0.994 | 0.988 | 0.991  N=0=169 | 1.000 | 1.000 | 1.000  N=1=172 | 0.994 | 0.989 | 0.993

## 2. null 格画像（强制打标 + 独立评分器对照）

| scope | null n | forced 0/1 | judge_big 对null有害率 | 对null |m|med | 总体 |m|med | null<5 |
|---|---|---|---|---|---|---|---|
| s17_e4b_audio | 23 | 1/22 | 0.435 | 4.4688 | 7.6875 | 19 |
| s17_e4b_text | 19 | 1/18 | 0.158 | 4.1875 | 7.5000 | 14 |
| s33_hetero_audio | 3 | 0/3 | 0.000 | 6.7031 | 7.5938 | 1 |

null 格 s17_e4b_audio forced vs judge_big：9/23（0.391）
null 格 s17_e4b_audio forced vs qwen32：14/23（0.609）
null 格 s17_e4b_audio forced vs strongreject：18/23（0.783）
null 格 s17_e4b_audio forced vs harmbench：13/23（0.565）
null 格 s17_e4b_text forced vs judge_big：2/19（0.105）
null 格 s17_e4b_text forced vs qwen32：2/19（0.105）
null 格 s17_e4b_text forced vs strongreject：12/19（0.632）
null 格 s17_e4b_text forced vs harmbench：6/19（0.316）
null 格 s33_hetero_audio forced vs judge_big：0/3（0.000）

## 3. 汇总

- 本阶段补全 null：45 条（S17 音频 + S17 文本 + S33）
- 与 S35（E2B 主链 27 + S28 11）合并后，S34 审计表全部 null 均具强制标签
- 标签文件：`results/gpu1_pipeline/s36_forced_complete_labels.jsonl`
  （全格；未改写任何生产缓存）
- 详情：`results/gpu1_pipeline/s36_forced_complete.json`