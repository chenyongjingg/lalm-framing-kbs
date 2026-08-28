# -*- coding: utf-8 -*-
"""
gemma4_batch_text.py — Gemma-4 文本批量推理模块（选项 B，v6.5 宪法下）

背景（证据链：audit_log.md AUDIT #165 / report/acceleration_plan.md v2）：
- stage_p1_full.py L371-416 / stage_p0c.py L216-251 的 Gemma-4 文本路径为逐条
  贪心推理（do_sample=False，无采样 RNG，确定性）。
- Gemma4Processor 无 padding_side、批路径不稳（逐条"稳妥"），故原为逐条。
- 本模块提供与逐条路径**逐位同构**的批量推理，唯一变量是 batch 维度：
    1) 逐单元用与生产完全相同的 `proc(text=...)` tokenize
       → 每序列 input_ids/attention_mask 逐位一致（构造上保证）；
    2) 手工左 padding 拼成 batch，attention_mask 因果隔离；
    3) generate 贪心；decode 用与生产相同的 `proc.batch_decode`。
- 残余风险 = 内核因 batch 维度选不同算法的浮点微差 → 由 validate_batch_text.py
  验证协议（E4B 200 + E2B 100 单元 A/B，100% 逐字节一致才启用）把关。
  任一不一致 → 弃 B（逐条路径保留，无损失）。

启用前提：validate_batch_text.py 全绿。
"""
from __future__ import annotations


def encode_texts_left_padded(proc, texts, max_len=4096):
    """逐单元 tokenize（与生产 proc(text=...) 完全相同），左 padding 组 batch。

    返回 {"input_ids": Tensor[B,L], "attention_mask": Tensor[B,L]}（CPU）。
    pad 位于左侧，attention_mask 将 pad 位置置 0 → 因果注意力隔离，逐序列独立。
    """
    import torch

    encs = []
    for t in texts:
        e = proc(text=t, return_tensors="pt", truncation=True, max_length=max_len)
        encs.append(e)
    if not encs:
        raise ValueError("texts 为空")
    Lmax = max(e["input_ids"].shape[1] for e in encs)
    pad_id = proc.tokenizer.pad_token_id
    n = len(encs)
    input_ids = torch.full((n, Lmax), pad_id, dtype=torch.long)
    attn = torch.zeros((n, Lmax), dtype=torch.long)
    for i, e in enumerate(encs):
        ids = e["input_ids"][0]
        L = ids.shape[0]
        am = e.get("attention_mask")
        if am is None:
            am = torch.ones_like(ids)
        else:
            am = am[0]
        input_ids[i, Lmax - L:] = ids
        attn[i, Lmax - L:] = am
    return {"input_ids": input_ids, "attention_mask": attn}


def infer_text_batch(model, proc, texts, max_new=512, max_len=4096, bs=8):
    """Gemma-4 文本批量贪心推理。

    texts: list[str]，已应用 chat template（system+user）。
    返回 list[str]，长度 = len(texts)，顺序与输入一致。
    失败抛异常（由调用方逐条降级）。
    """
    import torch

    results = [None] * len(texts)
    for s in range(0, len(texts), bs):
        chunk = texts[s:s + bs]
        inputs = encode_texts_left_padded(proc, chunk, max_len)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=proc.tokenizer.pad_token_id)
        resp = proc.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        resp = list(resp)
        if len(resp) != len(chunk):
            raise RuntimeError(
                f"batch_decode 返回 {len(resp)} 行，期望 {len(chunk)}")
        results[s:s + bs] = resp
    return results
