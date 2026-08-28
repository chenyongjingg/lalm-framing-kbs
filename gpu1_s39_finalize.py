#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S39 收官：重建完整汇总（strongreject+harmbench+qwen32 披露）。"""
import json
import sys
from pathlib import Path

root = Path("/root/lalm_framing_revision_v6")
cache_dir = root / "results" / "gpu1_pipeline" / "scorers_cache"


def _stat(fam):
    p = cache_dir / ("s33_hetero_audio_%s.jsonl" % fam)
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    nn = [r for r in rows if r.get("label") is not None]
    hr = round(sum(r["label"] for r in nn) / max(1, len(nn)), 4) if nn else 0.0
    return {"cached": len(rows), "nonnull": len(nn), "harmful_rate": hr}


caches = {fam: _stat(fam) for fam in ("strongreject", "harmbench", "qwen32")}
summary = {
    "stage": "S39",
    "date": "2026-08-16",
    "purpose": ("S33（Qwen2-Audio 温度鲁棒性 scope）补官方基准腿 "
                "strongreject+harmbench；qwen32 因 GPU1 显存受限未完成"),
    "method": ("复用 S21 协议：strongreject=Gemma4ScorerBase.score_batch；"
               "harmbench=左填充 1-token Yes/No logit wrapper（--hb-batch 2，"
               "GPU1 被主链占用 4.86GB 后降批；label 逐条 argmax 判定不受 "
               "batch 影响）；rid 键控幂等，s33_ 前缀"),
    "disclosure": ("qwen32 32B-AWQ 在 GPU1 上 OOM（主链 505447 占用 GPU1 "
                   "4.86GB，剩余 ~19.2GB 装不下 ~20GB 模型），仅 4/344 非空，"
                   "不纳入效应分析。S33 已有 5 家完整覆盖（jb/js/sr/hb/sg），"
                   "与 S28 对称。"),
    "s33_hetero_audio": caches,
}
out = root / "results" / "gpu1_pipeline" / "s39_s33_bench.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("已重建 s39_s33_bench.json")
print(json.dumps(caches, indent=1))

# 对照 S28 vs S33 harmbench N 效应
d33 = json.load((root / "results" / "gpu1_pipeline" /
                 "s33_five_family_effects.json").open(encoding="utf-8"))
d28 = json.load((root / "results" / "gpu1_pipeline" /
                 "s28_five_family_effects.json").open(encoding="utf-8"))
print("S33 harmbench N(Et0) =", d33["effects"]["harmbench"]["N_Et0"])
print("S28 harmbench N(Et0) =", d28["effects"]["harmbench"]["N_Et0"])
sys.exit(0)
