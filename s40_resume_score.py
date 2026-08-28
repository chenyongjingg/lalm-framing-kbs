#!/usr/bin/env python3
"""S40 补评分（AUDIT #182）：s40 死因 = release() 未彻底释放 E4B 推理模型
→ DualJudgeScorer 再载 E4B OOM（与 AUDIT #179 unload_all 同源）。
重生成产物 s40_regenerated.json（240/240 有 new_response）完好，本脚本在
空 GPU1 上只做评分+统计，不重跑 TTS/推理。只写 s40_* 输出，纪律合规。

用法：CUDA_VISIBLE_DEVICES=1 python3 s40_resume_score.py
"""
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path("/root/lalm_framing_revision_v6")
OUT = ROOT / "results" / "gpu1_pipeline"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(
    filename=str(ROOT / "logs" / "s40_resume.log"),
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("s40_resume")


def main():
    import yaml
    cfg = yaml.safe_load(open(ROOT / "pipeline_config.yaml", encoding="utf-8"))
    regen = json.load(open(OUT / "s40_regenerated.json", encoding="utf-8"))
    n_ok = sum(1 for u in regen if u.get("new_response"))
    log.info("读回 %d 单元（有响应 %d/%d）", len(regen), n_ok, len(regen))

    from scorer_utils import DualJudgeScorer
    jcfg = cfg["scorers"]
    small = jcfg.get("judge_small_model", jcfg.get("judge_mistral_model"))
    log.info("加载 DualJudgeScorer big=%s small=%s（空 GPU1，bf16 直载）",
             jcfg["judge_big_model"], small)
    dual = DualJudgeScorer(jcfg["judge_big_model"], small, load_in_4bit=False)
    log.info("DualJudgeScorer 加载完成 → big 轮")

    # big 轮：E4B 批评全部响应
    for i, u in enumerate(regen):
        resp = u.get("new_response")
        if not resp:
            u["dual_big"] = None
            continue
        try:
            u["dual_big"] = dual.score_one_big(u["prompt"], resp)
        except Exception as e:  # noqa: BLE001
            log.warning("big 失败 idx=%d: %s", i, str(e)[:120])
            u["dual_big"] = None
        if (i + 1) % 25 == 0:
            log.info("big 评分 %d/%d", i + 1, len(regen))
    dual.unload_big()
    log.info("big 轮完成 → small 轮")

    try:
        dual.load_mistral_only()
    except Exception as e:  # noqa: BLE001
        log.warning("load_mistral_only 失败: %s", str(e)[:150])

    for i, u in enumerate(regen):
        resp = u.get("new_response")
        if not resp:
            u["dual_small"] = None
            continue
        try:
            u["dual_small"] = dual.score_one_mistral(u["prompt"], resp)
        except Exception as e:  # noqa: BLE001
            log.warning("small 失败 idx=%d: %s", i, str(e)[:120])
            u["dual_small"] = None
        if (i + 1) % 25 == 0:
            log.info("small 评分 %d/%d", i + 1, len(regen))
    dual.unload_mistral()
    dual.close()

    # 派生 dual_label（与主链口径一致：big==small 才给标签）
    for u in regen:
        b, s = u.get("dual_big"), u.get("dual_small")
        u["dual_label"] = (int(b) if b is not None else None) if b == s else None
    (OUT / "s40_scored.json").write_text(
        json.dumps(regen, ensure_ascii=False, indent=2), encoding="utf-8")
    n_lab = sum(1 for u in regen if u.get("dual_label") is not None)
    log.info("已落盘 s40_scored.json，有效标签 %d/%d", n_lab, len(regen))

    # 统计（与 s40.py 同口径）：N 效应跨 TTS 是否复现
    def dual_harm(sub):
        labs = [u["dual_label"] for u in sub if u.get("dual_label") is not None]
        if not labs:
            return None
        return sum(labs) / len(labs)

    stats = {
        "voice": "zh-CN-YunxiNeural",
        "n_query": len({u["query_id"] for u in regen}),
        "n_units": len(regen),
        "n_with_label": n_lab,
        "by_As": {}, "N_effect": {},
    }
    for As in ["neutral_audio", "styled_audio"]:
        n0 = [u for u in regen if u["A_s"] == As and u["N"] == 0]
        n1 = [u for u in regen if u["A_s"] == As and u["N"] == 1]
        h0, h1 = dual_harm(n0), dual_harm(n1)
        stats["by_As"][As] = {
            "harm_N0": h0, "harm_N1": h1,
            "N_effect_pp": None if (h0 is None or h1 is None)
            else round((h1 - h0) * 100, 2),
            "n_N0": len(n0), "n_N1": len(n1)}
    all0 = [u for u in regen if u["N"] == 0]
    all1 = [u for u in regen if u["N"] == 1]
    h0, h1 = dual_harm(all0), dual_harm(all1)
    stats["N_effect"]["all_audio"] = {
        "harm_N0": h0, "harm_N1": h1,
        "N_effect_pp": None if (h0 is None or h1 is None)
        else round((h1 - h0) * 100, 2),
        "n_N0": len(all0), "n_N1": len(all1)}
    (OUT / "s40_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("S40 补评分完成。N 效应(合并): %s", stats["N_effect"]["all_audio"])
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
