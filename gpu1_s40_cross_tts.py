#!/usr/bin/env python3
"""S40 跨 TTS 稳健性（AUDIT #180 修复窗口，2026-08-21 16:00 UTC）。

科学问题：P1-PILOT 关键发现"N 效应（framing 攻击）audio>text"（dual_judge
audio 11.91pp vs text 4.66pp）基于单一 TTS 声音（zh-CN-XiaoxiaoNeural 女声）。
审稿人必问：换成别的 TTS 声音，N 效应还在吗？

设计：取 P1-PILOT 音频配对子集（150 query 中确定性抽 60 query，
N∈{0,1} × A_s∈{neutral_audio, styled_audio} = 每 query 4 单元 → 240 单元），
用第二 TTS 声音 zh-CN-YunxiNeural（男声）重合成 WAV → gemma_4_e4b 音频推理
重生成响应 → dual_judge（E4B+E2B）评分 → 计算 N=0 vs N=1 的 dual 有害率差
（N 效应 Δpp）及 A_s 分层，与 Xiaoxiao 原声对照。

纪律：只写新文件（results/gpu1_pipeline/s40_*），零人工标注，GPU1
（CUDA_VISIBLE_DEVICES=1），不触碰主链响应/checkpoint/scored。与主链
P1-FULL 评分（GPU0）物理隔离。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

def resolve_root(cfg):
    return Path(cfg.get("paths", {}).get("root", "."))

def get_logger(name, root):
    import logging
    logging.basicConfig(
        filename=str(root / "logs" / f"{name}.log"),
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--n-query", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry", action="store_true", help="只抽样落盘设计，不跑推理")
    args = ap.parse_args()

    import random
    import yaml
    import numpy as np

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = resolve_root(cfg)
    log = get_logger("s40", root)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # ---- 配对源：P1-PILOT E4B audio 响应（含完整 N×A_s 设计）----
    resp_file = root / "responses" / "P1_PILOT" / "gemma_4_e4b_responses.jsonl"
    rows = [json.loads(l) for l in open(resp_file, encoding="utf-8") if l.strip()]
    aud = [r for r in rows if r.get("modality") == "audio"]
    by_q = {}
    for r in aud:
        by_q.setdefault(r["query_id"], []).append(r)
    qids = sorted(by_q.keys())
    # N 主效应配对：combo = (E_t, N, R)（经实证 combo[1]=N：combo[1]=1 ⟺ N=1）。
    # 固定基线档 E_t=0, R=0，取 combo=(0,0,0)→N=0 与 combo=(0,1,0)→N=1。
    # × A_s ∈ {neutral, styled} → 每 query 4 单元。
    complete = []
    for q in qids:
        grp = by_q[q]
        have = {(str(r.get("combo")), r["N"], r["A_s"]) for r in grp}
        want = {(str([0, 0, 0]), 0, "neutral_audio"),
                (str([0, 0, 0]), 0, "styled_audio"),
                (str([0, 1, 0]), 1, "neutral_audio"),
                (str([0, 1, 0]), 1, "styled_audio")}
        if want <= have:
            complete.append(q)
    log.info("完整配对 query=%d/%d", len(complete), len(qids))
    rng = random.Random(args.seed)
    sel_q = rng.sample(complete, min(args.n_query, len(complete)))
    units = []
    for q in sel_q:
        for r in by_q[q]:
            combo = str(r.get("combo"))
            if (combo in (str([0, 0, 0]), str([0, 1, 0]))
                    and r["template_idx"] == 0):
                units.append({
                    "query_id": q, "N": r["N"], "A_s": r["A_s"],
                    "prompt": r["prompt"], "template_idx": r["template_idx"],
                    "old_audio": r["audio_path"],
                    "old_response": r["response"],
                    "combo": r.get("combo"),
                })
    log.info("S40 单元=%d（query=%d × N0/N1 × neutral/styled，E_t=0,R=0，template=0）",
             len(units), len(sel_q))
    design_path = out_dir / "s40_design.json"
    design_path.write_text(json.dumps(units, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    if args.dry:
        log.info("dry 结束，design=%s", design_path)
        return 0

    # ---- TTS：第二声音重合成 ----
    from stage_p0c import _lalm_audio_one, synthesize_tts
    from gpu1_common import load_generation_model, release

    voice = "zh-CN-YunxiNeural"
    audio_dir = out_dir / "s40_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    texts = [u["prompt"] for u in units]
    wavs = synthesize_tts(texts, audio_dir, voice,
                          cfg.get("p0c", {}).get("tts", {}).get("sample_rate", 16000),
                          log, prefix="s40_")
    log.info("TTS 合成 %d 条（voice=%s）", len(wavs), voice)
    n_ok = sum(1 for w in wavs if w)
    log.info("TTS 成功 %d/%d", n_ok, len(wavs))

    # ---- E4B 音频推理重生成 ----
    mname = "gemma_4_e4b"
    mconf = cfg["models"][mname]
    log.info("加载 %s ...", mname)
    model, tok = load_generation_model(mname, mconf, cfg, log)
    max_new = cfg.get("gpu", {}).get("max_new_tokens", 512)
    new_units = []
    for u, wav in zip(units, wavs):
        rec = dict(u)
        rec["new_audio"] = wav
        if not wav:
            rec["new_response"] = None
            rec["error"] = "tts_failed"
            new_units.append(rec)
            continue
        try:
            resp = _lalm_audio_one(mname, model, tok, wav, u["prompt"], max_new)
            rec["new_response"] = resp
        except Exception as e:  # noqa: BLE001
            log.warning("推理失败 idx=%s N=%s: %s", u["query_id"], u["N"],
                        str(e)[:150])
            rec["new_response"] = None
            rec["error"] = str(e)[:150]
        new_units.append(rec)
        if len(new_units) % 25 == 0:
            log.info("E4B 重生成 %d/%d", len(new_units), len(units))
    # 中途落盘（防崩丢失）
    (out_dir / "s40_regenerated.json").write_text(
        json.dumps(new_units, ensure_ascii=False, indent=2), encoding="utf-8")
    release()
    log.info("E4B 重生成完成，成功 %d/%d",
             sum(1 for u in new_units if u.get("new_response")), len(new_units))

    # ---- dual_judge 评分（顺序加载 E4B→E2B）----
    from scorer_utils import DualJudgeScorer
    jcfg = cfg["scorers"]
    dual = DualJudgeScorer(jcfg["judge_big_model"],
                           jcfg.get("judge_small_model",
                                    jcfg.get("judge_mistral_model")),
                           load_in_4bit=False)
    scored = []
    for i, u in enumerate(new_units):
        resp = u.get("new_response")
        if not resp:
            u["dual_label"] = None
            u["dual_big"] = None
            u["dual_small"] = None
            scored.append(u)
            continue
        try:
            u["dual_big"] = dual.score_one_big(u["prompt"], resp)
        except Exception as e:  # noqa: BLE001
            log.warning("big 评分失败 idx=%d: %s", i, str(e)[:120])
            u["dual_big"] = None
        scored.append(u)
        if len(scored) % 25 == 0:
            log.info("big 评分 %d/%d", len(scored), len(new_units))
    dual.unload_big()
    # small 轮
    try:
        dual.load_mistral_only()
    except Exception as e:  # noqa: BLE001
        log.warning("load_mistral_only 失败: %s", str(e)[:150])
    for i, u in enumerate(scored):
        resp = u.get("new_response")
        if not resp:
            u["dual_small"] = None
            continue
        try:
            u["dual_small"] = dual.score_one_mistral(u["prompt"], resp)
        except Exception as e:  # noqa: BLE001
            log.warning("small 评分失败 idx=%d: %s", i, str(e)[:120])
            u["dual_small"] = None
    dual.unload_mistral()
    dual.close()

    # ---- 派生 dual 标签（一致才给标签，与主链口径一致）----
    for u in scored:
        b, s = u.get("dual_big"), u.get("dual_small")
        u["dual_label"] = (int(b) if b is not None else None) if b == s else None
    out_path = out_dir / "s40_scored.json"
    out_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # ---- 统计：N 效应（audio_stronger 跨 TTS 是否复现）----
    def dual_harm(sub):
        labs = [u["dual_label"] for u in sub
                if u.get("dual_label") is not None]
        if not labs:
            return None
        return sum(labs) / len(labs)
    stats = {"voice": voice, "n_query": len(sel_q), "n_units": len(scored),
             "n_with_label": sum(1 for u in scored if u.get("dual_label") is not None),
             "by_As": {}, "N_effect": {}}
    for As in ["neutral_audio", "styled_audio"]:
        n0 = [u for u in scored if u["A_s"] == As and u["N"] == 0]
        n1 = [u for u in scored if u["A_s"] == As and u["N"] == 1]
        h0, h1 = dual_harm(n0), dual_harm(n1)
        stats["by_As"][As] = {
            "harm_N0": h0, "harm_N1": h1,
            "N_effect_pp": None if (h0 is None or h1 is None) else round((h1 - h0) * 100, 2),
            "n_N0": len(n0), "n_N1": len(n1),
        }
    # 合并 A_s
    all0 = [u for u in scored if u["N"] == 0]
    all1 = [u for u in scored if u["N"] == 1]
    h0, h1 = dual_harm(all0), dual_harm(all1)
    stats["N_effect"]["all_audio"] = {
        "harm_N0": h0, "harm_N1": h1,
        "N_effect_pp": None if (h0 is None or h1 is None) else round((h1 - h0) * 100, 2),
        "n_N0": len(all0), "n_N1": len(all1),
    }
    (out_dir / "s40_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("S40 完成。N 效应(合并 audio): %s", stats["N_effect"]["all_audio"])
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
