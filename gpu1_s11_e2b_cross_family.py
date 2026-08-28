#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S11：E2B 全量 3600 文本响应 跨族核验（补 dual_judge 的 E2B 侧证据）。

背景（S9 遗留空白）：
  S9 用 Qwen2.5-32B-AWQ 强异构锚点核验了 E4B 侧（S4 480 响应 + S5 音频 120），
  dual_judge vs qwen32 收敛 0.84-0.93。但 dual_judge 共识 = judge_big(E4B) +
  judge_small(E2B) 一致才采信 —— judge_small(E2B) 侧从未被强异构锚点直接核验。
  若 E2B 侧评分与 qwen32 不收敛，则 dual_judge 对 E2B 生成的响应不可信。

S11 设计：
  - 数据：E2B 响应 3600 条（text，E_t 0/1 各 1800，主流水线已 100% 完成）。
  - 评分：qwen32 全量 + judge_big(GemmaJudge E4B) 全量 + judge_small(GemmaJudge E2B)
    全量，每个评分器独立加载→评分→close→gc→empty_cache（杜绝 S10b OOM）。
  - 分析：dual_judge 共识 vs qwen32 判据（全量 + 按 E_t 分层）；judge_big vs qwen32、
    judge_small vs qwen32 直接一致率（补 E2B 侧缺口）；Dawid-Skene；分布检查。
  - 判据：>=0.80 跨族收敛，<0.80 如实披露（用户批准）。
  - 纪律：只写 results/gpu1_pipeline/+report/，不碰主账本/checkpoint/done。

用法：CUDA_VISIBLE_DEVICES=1 python gpu1_s11_e2b_cross_family.py [--smoke] [--max-rows N]
"""
import argparse
import collections
import gc
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu1_s9_cross_family as s9  # noqa: E402


def _log(m):
    print("[s11 %s] %s" % (Path(__file__).stem, m), flush=True)


def _distr(lbl, name):
    """分布塌缩检查（对齐 scorer_utils.distribution_check 语义）。"""
    vals = [int(x) for x in lbl if x is not None]
    if not vals:
        return {"pass": False, "error": "无有效标签", "counts": {},
                "pos_rate": None}
    from scorer_utils import distribution_check
    try:
        distribution_check(vals, name)
        return {"pass": True,
                "counts": {str(c): vals.count(c) for c in sorted(set(vals))},
                "pos_rate": round(vals.count(1) / len(vals), 4)}
    except RuntimeError as e:
        return {"pass": False, "error": str(e)[:200],
                "counts": {str(c): vals.count(c) for c in sorted(set(vals))},
                "pos_rate": round(vals.count(1) / len(vals), 4)
                if vals else None}


def _score_all(scorer, rows, tag="", cache_path=None):
    """逐条评分（复用 score_one）。若 cache_path 给定，增量落盘、重启可恢复。

    恢复语义：已存在于缓存的行跳过；单行失败记 None 不重试。返回与 rows
    等长的 label 列表（缺失为 None）。
    """
    done = {}
    if cache_path and cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                done[rec["i"]] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
        if done:
            _log("[%s] 从缓存恢复 %d 条已评分" % (tag, len(done)))
    out = []
    for i, r in enumerate(rows):
        if i in done:
            out.append(done[i])
            continue
        try:
            res = scorer.score_one(r["prompt"], r["response"] or "")
            label = res.get("label")
        except Exception as e:  # noqa: BLE001
            _log("[%s] idx=%d 失败: %s" % (tag, i, str(e)[:120]))
            label = None
        out.append(label)
        if cache_path:
            with cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"i": i, "label": label},
                                   ensure_ascii=False) + "\n")
        if (i + 1) % 400 == 0:
            _log("[%s] 评分 %d/%d" % (tag, i + 1, len(rows)))
    return out


def _crit_dj_vs_qw(mat, idxs):
    """dual_judge 共识（jb==js）vs qwen32 一致率。mat: n x 3 (jb, js, qw)。"""
    pairs, dispute, dj_n = [], 0, 0
    for i in idxs:
        jb, js, qw = mat[i]
        if np.isnan(jb) or np.isnan(js):
            continue
        dj_n += 1
        if jb != js:
            dispute += 1
            continue
        if np.isnan(qw):
            continue
        pairs.append((int(jb), int(qw)))
    if not pairs:
        return None, {"n_dual_judge": dj_n, "dispute_rate": None}
    agree = sum(1 for a, b in pairs if a == b)
    rate = agree / len(pairs)
    return {
        "n_dual_consensus": len(pairs),
        "agreement_dual_vs_qwen32": round(rate, 4),
        "pass_0_80": rate >= 0.80,
        "verdict": "跨族收敛（测量可信）" if rate >= 0.80 else "评分器敏感",
    }, {"n_dual_judge": dj_n,
        "dispute_rate": round(dispute / dj_n, 4) if dj_n else None,
        "n_disputed": dispute}


def _pairwise(mat, idxs):
    """mat n x 3 (jb, js, qw)：3 评分器两两一致率 + κ。"""
    from scorer_utils import cohens_kappa
    names = ["judge_big", "judge_small", "qwen32"]
    pairs = []
    for a in range(3):
        for b in range(a + 1, 3):
            va, vb = mat[:, a], mat[:, b]
            mask = ~np.isnan(va) & ~np.isnan(vb)
            if mask.sum() == 0:
                continue
            agree = float((va[mask] == vb[mask]).mean())
            try:
                k = cohens_kappa(list(va[mask].astype(int)),
                                 list(vb[mask].astype(int)),
                                 n_boot=1000)["kappa"]
            except Exception:  # noqa: BLE001
                k = None
            pairs.append({"scorer_a": names[a], "scorer_b": names[b],
                          "n_valid": int(mask.sum()), "agreement": round(agree, 4),
                          "cohens_kappa": k})
    return pairs


def _ds(mat):
    from scorer_utils import dawid_skene
    try:
        ds = dawid_skene(mat)
        return {
            "converged": bool(ds["converged"]), "n_iter": int(ds["n_iter"]),
            "per_scorer": {["judge_big", "judge_small", "qwen32"][j]: {
                "sensitivity": round(float(ds["sensitivity"][j]), 4),
                "specificity": round(float(ds["specificity"][j]), 4),
                "error_rate": round(float(ds["error_rate"][j]), 4),
            } for j in range(3)},
            "latent_pos_rate": round(float(ds["item_label"].mean()), 4),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--skip-qwen32", action="store_true",
                    help="debug: 仅跑 6 现有评分器，不加载 qwen32")
    ap.add_argument("--cache-dir", default=None,
                    help="评分缓存目录（默认 out_dir/scorers_cache）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        out_dir / "scorers_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- 数据 ----
    e2b_path = root / "responses" / "P1_PILOT" / "gemma_4_e2b_responses.jsonl"
    rows = [json.loads(l) for l in open(e2b_path, encoding="utf-8")]
    if args.max_rows:
        rows = rows[:args.max_rows]
    if args.smoke:
        rows = rows[:6]
    _log("E2B 响应=%d（E_t0=%d, E_t1=%d）" % (
        len(rows),
        sum(1 for r in rows if r["E_t"] == 0),
        sum(1 for r in rows if r["E_t"] == 1)))

    # ---- 评分（顺序加载/关闭，杜绝 OOM） ----
    lbl = {"judge_big": [None] * len(rows),
           "judge_small": [None] * len(rows),
           "qwen32": [None] * len(rows)}

    done_qw = {}
    if not args.skip_qwen32:
        qw_cache = cache_dir / "qwen32.jsonl"
        if qw_cache.exists():
            for line in qw_cache.open(encoding="utf-8"):
                try:
                    rec = json.loads(line)
                    done_qw[rec["i"]] = rec["label"]
                except Exception:  # noqa: BLE001
                    continue
            for i, v in done_qw.items():
                lbl["qwen32"][i] = v
            _log("qwen32 缓存恢复 %d/%d" % (len(done_qw), len(rows)))
        # 缺失 = 从未评分（缓存中无该索引）。已缓存（含 label=null）跳过。
        missing = [i for i in range(len(rows)) if i not in done_qw]
        if missing:
            qw = s9.Qwen32Scorer(s9._discover_awq(), batch_size=8)
            for start in range(0, len(missing), 100):
                chunk = missing[start:start + 100]
                pairs = [(rows[i]["prompt"], rows[i]["response"] or "")
                         for i in chunk]
                res = qw.score_batch(pairs)
                with qw_cache.open("a", encoding="utf-8") as f:
                    for i, x in zip(chunk, res):
                        lbl["qwen32"][i] = x.get("label")
                        f.write(json.dumps({"i": i,
                                            "label": x.get("label")},
                                           ensure_ascii=False) + "\n")
                _log("qwen32 评分 %d/%d" % (min(start + len(chunk),
                                                 len(missing)),
                                            len(missing)))
            qw.close()
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        _log("qwen32 完成: 非空=%d" % sum(1 for v in lbl["qwen32"]
                                          if v is not None))

    s9.register_scorers(cfg)
    for sn in ("judge_small", "judge_big"):
        sc = s9.SCORER_FACTORIES[sn]()
        lbl[sn] = _score_all(sc, rows, tag=sn,
                             cache_path=cache_dir / (sn + ".jsonl"))
        sc.close()
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        _log("%s 完成: 非空=%d" % (sn, sum(1 for v in lbl[sn]
                                           if v is not None)))

    # ---- 标签矩阵 n x 3 ----
    mat = np.full((len(rows), 3), np.nan)
    for j, sn in enumerate(["judge_big", "judge_small", "qwen32"]):
        for i, v in enumerate(lbl[sn]):
            if v is not None:
                mat[i, j] = int(v)

    # ---- 分析 ----
    distr = {sn: _distr(lbl[sn], sn) for sn in ["judge_big", "judge_small",
                                                "qwen32"]}
    all_idx = list(range(len(rows)))
    crit_all, dj_all = _crit_dj_vs_qw(mat, all_idx)
    by_et = {}
    for et in (0, 1):
        idxs = [i for i, r in enumerate(rows) if r["E_t"] == et]
        crit, dj = _crit_dj_vs_qw(mat, idxs)
        by_et["E_t=%d" % et] = {
            "n_responses": len(idxs),
            "criterion": crit,
            "dual_judge": dj,
            "pairwise": _pairwise(mat, np.array(idxs, dtype=int)),
            "dawid_skene": _ds(mat[np.array(idxs, dtype=int)]),
        }
        _log("E_t=%d 判据: %s" % (et, json.dumps(crit, ensure_ascii=False)))

    overview = {
        "stage": "S11",
        "model": "gemma_4_e2b_responses (text 3600)",
        "n_responses": len(rows),
        "distribution": distr,
        "overall": {"criterion": crit_all, "dual_judge": dj_all,
                    "pairwise": _pairwise(mat, np.array(all_idx, dtype=int)),
                    "dawid_skene": _ds(mat)},
        "by_et": by_et,
        "note": ("E2B 生成响应在 dual_judge（judge_big E4B + judge_small E2B 一致）"
                 "口径下 vs Qwen2.5-32B-AWQ 强异构锚点的一致率；≥0.80 跨族收敛 "
                 "补全 S9 缺失的 judge_small(E2B) 侧证据。"),
    }
    _log("overall 判据: %s" % json.dumps(crit_all, ensure_ascii=False))

    # ---- 落盘 ----
    with open(out_dir / "s11_e2b_cross_family.json", "w",
              encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    md = render_md(overview)
    (out_dir / "s11_e2b_cross_family.md").write_text(md, encoding="utf-8")
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "s11_e2b_cross_family.md").write_text(md, encoding="utf-8")
    _log("已落盘 s11_e2b_cross_family.json/.md")

    print(json.dumps({"stage": "S11", "overall": crit_all,
                      "by_et": {k: v["criterion"] for k, v in by_et.items()}},
                     ensure_ascii=False, indent=2))
    return 0


def render_md(o):
    lines = [
        "# S11：E2B 全量 3600 文本响应跨族核验（GPU1 · 2026-08-14）\n",
        "## 动机",
        "S9 已核验 judge_big(E4B) 侧 vs Qwen2.5-32B-AWQ 跨族收敛（0.84-0.93）。",
        "但 dual_judge 共识 = judge_big(E4B) + judge_small(E2B) 一致才采信 —— "
        "judge_small(E2B) 侧此前未被强异构锚点直接核验。本实验对 E2B 生成的",
        "3600 条文本响应（E_t 0/1 各 1800）全量评分，补全该证据。\n",
        "## 判据",
        "dual_judge 共识 vs qwen32 一致率 ≥0.80 → 跨族收敛（测量可信）；",
        "<0.80 → 如实披露。\n",
        "## 数据",
        "- 源：E2B 响应 %d 条（text），主流水线 100%% 完成。" % o["n_responses"],
        "- 评分器：judge_big（GemmaJudge E4B）、judge_small（GemmaJudge E2B）、",
        "  qwen32（Qwen2.5-32B-AWQ，CROSS_CHECK_RUBRIC）。\n",
        "## 分布检查",
    ]
    for sn, d in o["distribution"].items():
        lines.append("- %s: pass=%s, pos_rate=%s, counts=%s" % (
            sn, d.get("pass"), d.get("pos_rate"), d.get("counts")))
    c = o["overall"]["criterion"]
    if c:
        lines.append("\n## 全样本判据")
        lines.append("- dual_judge vs qwen32 一致率: %.4f（n=%d）%s" % (
            c["agreement_dual_vs_qwen32"], c["n_dual_consensus"], c["verdict"]))
    lines.append("\n## 全样本两两一致率")
    for p in o["overall"]["pairwise"]:
        lines.append("- %s vs %s: %.4f（n=%d, κ=%s）" % (
            p["scorer_a"], p["scorer_b"], p["agreement"], p["n_valid"],
            p["cohens_kappa"]))
    lines.append("\n## 按 E_t 分层")
    for et, blk in o["by_et"].items():
        c = blk["criterion"]
        line = "- %s（n=%d）: " % (et, blk["n_responses"])
        if c:
            line += "dual_judge vs qwen32=%.4f（n=%d）%s" % (
                c["agreement_dual_vs_qwen32"], c["n_dual_consensus"],
                c["verdict"])
        else:
            line += "无有效 dual_judge 共识对"
        lines.append(line)
        for p in blk["pairwise"]:
            lines.append("  - %s vs %s: %.4f（n=%d, κ=%s）" % (
                p["scorer_a"], p["scorer_b"], p["agreement"], p["n_valid"],
                p["cohens_kappa"]))
    lines.append("\n## 判读")
    if c and c["pass_0_80"]:
        lines.append("> E2B 侧 dual_judge vs qwen32 跨族收敛 → judge_small(E2B)"
                     " 评分与强异构锚点一致，dual_judge 两腿（E4B+E2B）均获核验，"
                     "中文响应测量可信度闭合。")
    else:
        lines.append("> E2B 侧未达 0.80 → 如实披露：dual_judge 对 E2B 生成响应的"
                     "测量存在评分器敏感，需在论文披露。")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
