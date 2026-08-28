#!/usr/bin/env python3
"""s_gpu1_prescoring.py：GPU1 并行预评分（GemmaJudge + DualJudge）。

背景：主链 stage_p1_full 在 GPU0 串行跑 5 评分器。StrongREJECT（128-token 生成）
实测极慢（~2.2s/条）。GemmaJudge/DualJudge 为 8-token 短输出（快），分流到空闲
GPU1 并行预评分，主链完成后交叉验证或合并加速。

纪律：只读 responses/P1_FULL/p1_full_responses.jsonl；只写 results/gpu1_prescoring*.jsonl
+ logs/gpu1_prescoring.log；绝不触碰主链文件。
"""
import json
import os
import sys
import time
import logging
from pathlib import Path

ROOT = Path("/root/lalm_framing_revision_v6")
RESP = ROOT / "responses" / "P1_FULL" / "p1_full_responses.jsonl"
OUT = ROOT / "results" / "gpu1_prescoring.jsonl"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(
    filename=str(ROOT / "logs" / "gpu1_prescoring.log"),
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gpu1_prescoring")


def load_rows():
    rows = [json.loads(l) for l in open(RESP, encoding="utf-8") if l.strip()]
    log.info("读响应 %d 条", len(rows))
    return rows


def behaviors_of(rows):
    # 与 stage_p1_full L786 同口径：query_text 回退 prompt 提取
    beh = []
    for r in rows:
        qt = str(r.get("query_text", "")).strip()
        if qt:
            beh.append(qt)
        else:
            p = str(r.get("prompt", ""))
            beh.append(p.split("：")[-1] if "：" in p else p)
    return beh


def load_results():
    results = []
    if OUT.exists():
        for l in open(OUT, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    results.append(json.loads(l))
                except Exception:
                    pass
    log.info("已有结果 %d 条", len(results))
    return results


def save(results):
    tmp = OUT.with_suffix(".tmp.jsonl")
    with open(tmp, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    tmp.replace(OUT)
    log.info("增量落盘 %d 条", len(results))


def main():
    rows = load_rows()
    beh = behaviors_of(rows)
    results = load_results()
    done = {r["response_id"] for r in results}
    pend = [(r, beh[i]) for i, r in enumerate(rows) if r["response_id"] not in done]
    log.info("待评分 %d 条", len(pend))

    from scorer_utils import GemmaJudgeScorer, DualJudgeScorer
    import yaml
    cfg = yaml.safe_load(open(ROOT / "pipeline_config.yaml", encoding="utf-8"))
    sc = cfg["scorers"]

    def upsert(rid, **kv):
        res = next((x for x in results if x["response_id"] == rid), None)
        if res is None:
            res = {"response_id": rid}
            results.append(res)
        for k, v in kv.items():
            res[k] = v

    # ---- GemmaJudge（E2B，8-token）----
    t0 = time.time()
    gemma = GemmaJudgeScorer(sc["gemma_judge_model"], load_in_4bit=False)
    gd = {r["response_id"] for r in results if r.get("gemma_label") is not None}
    todo = [p for p in pend if p[0]["response_id"] not in gd]
    log.info("Gemma 待评 %d", len(todo))
    for idx, (r, b) in enumerate(todo):
        try:
            lab = gemma.score_one(b, str(r["response"]))
            v = lab.get("label") if lab else None
            v = None if v is None else int(v)
        except Exception as e:  # noqa: BLE001
            v = None
            if idx < 10:
                log.warning("gemma fail idx=%d: %s", idx, str(e)[:100])
        upsert(r["response_id"], gemma_label=v)
        if (idx + 1) % 500 == 0:
            save(results)
            log.info("Gemma %d/%d (%.2f/s)", idx + 1, len(todo),
                     (idx + 1) / (time.time() - t0))
    gemma.close()
    save(results)
    log.info("Gemma 完成 %d 条", len(results))

    # ---- DualJudge（E4B big → E2B small，8-token）----
    t0 = time.time()
    dual = DualJudgeScorer(sc["judge_big_model"],
                           sc.get("judge_small_model",
                                  sc.get("judge_mistral_model")),
                           load_in_4bit=False)
    bd = {r["response_id"] for r in results if r.get("judge_big_label") is not None}
    todo = [p for p in pend if p[0]["response_id"] not in bd]
    log.info("DualJudge big 待评 %d", len(todo))
    for idx, (r, b) in enumerate(todo):
        try:
            lab = dual.score_one_big(b, str(r["response"]))
            v = None if lab is None else int(lab)
        except Exception as e:  # noqa: BLE001
            v = None
            if idx < 10:
                log.warning("big fail idx=%d: %s", idx, str(e)[:100])
        upsert(r["response_id"], judge_big_label=v)
        if (idx + 1) % 500 == 0:
            save(results)
            log.info("big %d/%d (%.2f/s)", idx + 1, len(todo),
                     (idx + 1) / (time.time() - t0))
    dual.unload_big()
    save(results)
    log.info("big 轮完成")

    try:
        dual.load_mistral_only()
    except Exception as e:  # noqa: BLE001
        log.warning("load_mistral_only 失败: %s", str(e)[:150])
    sd = {r["response_id"] for r in results if r.get("judge_mistral_label") is not None}
    todo = [p for p in pend if p[0]["response_id"] not in sd]
    log.info("DualJudge small 待评 %d", len(todo))
    for idx, (r, b) in enumerate(todo):
        try:
            lab = dual.score_one_mistral(b, str(r["response"]))
            v = None if lab is None else int(lab)
        except Exception as e:  # noqa: BLE001
            v = None
            if idx < 10:
                log.warning("small fail idx=%d: %s", idx, str(e)[:100])
        upsert(r["response_id"], judge_mistral_label=v)
        if (idx + 1) % 500 == 0:
            save(results)
            log.info("small %d/%d (%.2f/s)", idx + 1, len(todo),
                     (idx + 1) / (time.time() - t0))
    dual.unload_mistral()
    dual.close()
    save(results)
    log.info("全部完成，落盘 %d 条", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
