#!/usr/bin/env python
"""merge_scorers.py v2 — 合并全部评分来源 → 完整 P1-FULL 评分表 + parquet。
按 response_id 对齐合并（已核验三来源集合完全一致），产出：
  results/p1_full_scored.jsonl + results/p1_full_scored.parquet（子进程转）
  silver_label 按 stage_p2_msrf.py L898-915 逐字符复刻（P2-LORA 依赖）。
科学纪律：只新增文件，绝不覆盖任何已有产物；None 如实传播。
"""
import json, sys, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

def load_jsonl(p):
    d = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            d[json.loads(s)["response_id"]] = json.loads(s)
    return d

par = load_jsonl(RES / "p1_full_scored_partial.jsonl")
pre = load_jsonl(RES / "gpu1_prescoring.jsonl")
sr = {}
for part in (RES / "strongreject_part0.jsonl", RES / "strongreject_part1.jsonl"):
    sr.update(load_jsonl(part))

resp = {}
with (ROOT / "responses" / "P1_FULL" / "p1_full_responses.jsonl").open(encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s: continue
        r = json.loads(s)
        resp[r["response_id"]] = r

rids = list(resp.keys())
print("来源: partial=%d prescoring=%d strongreject=%d resp=%d" %
      (len(par), len(pre), len(sr), len(resp)))

assert set(par) == set(pre) == set(resp), "response_id 集合不一致，中止（防污染）"
missing_sr = [r for r in rids if r not in sr]
print("sr 缺失: %d（应=0 才完整）" % len(missing_sr))
if missing_sr:
    print("!! sr 未完成，请勿在 strongreject 双卡完成前跑合并")
    sys.exit(2)

rows = []
for rid in rids:
    p, pr, s, rv = par[rid], pre[rid], sr[rid], resp[rid]
    row = {"response_id": rid}
    row["kw_label"] = p.get("kw_label")
    row["hb_label"] = p.get("hb_label")
    row["hb_prob"] = p.get("hb_prob")
    row["gemma_label"] = pr.get("gemma_label")
    row["judge_big_label"] = pr.get("judge_big_label")
    row["judge_mistral_label"] = pr.get("judge_mistral_label")
    row["sr_label"] = s.get("sr_label")
    row["sr_score"] = s.get("sr_score")
    row["sr_score_1_5"] = s.get("sr_score_1_5")
    for c in ("response", "query_text", "prompt", "condition", "model",
              "modality", "lang", "query_id", "pool_query_id",
              "template_idx", "precision", "phase"):
        if c in rv:
            row[c] = rv.get(c)
    rows.append(row)

import pandas as pd
df = pd.DataFrame(rows)
df["judge_agree"] = (df["judge_big_label"].fillna(-1)
                     == df["judge_mistral_label"].fillna(-2)).astype(int)
silver = (
    (df["hb_label"] == df["sr_label"])
    & (df["judge_agree"] == 1)
    & (df["hb_label"] == df["judge_big_label"])
    & (df["hb_label"].notna())
)
df["silver_label"] = df["hb_label"].astype(float).where(silver, np.nan)
n_silver = int(silver.sum())
print("silver_label: %d/%d (%.1f%%)" % (n_silver, len(df), 100*n_silver/max(len(df),1)))

_fin = RES / "p1_full_scored"
_tmp = _fin.with_suffix(".tmp.jsonl")
df.to_json(_tmp, orient="records", lines=True, force_ascii=False)
print("JSONL ->", _tmp)

_code = ("import pandas as pd;"
         "pd.read_json(%r, lines=True).to_parquet(%r)" %
         (str(_tmp), str(_fin.with_suffix(".parquet"))))
subprocess.run([sys.executable, "-c", _code], check=True, timeout=900)
_tmp.unlink(missing_ok=True)

df2 = pd.read_parquet(_fin.with_suffix(".parquet"))
need = ["hb_label", "sr_label", "judge_big_label", "judge_mistral_label",
        "silver_label", "response", "condition"]
miss = [c for c in need if c not in df2.columns]
print("parquet 行数: %d | 契约缺列: %s" % (len(df2), miss if miss else "无"))
print("parquet ->", _fin.with_suffix(".parquet"))
