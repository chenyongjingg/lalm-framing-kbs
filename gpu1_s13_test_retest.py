#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S13：评分器 test-retest 稳定性（2026-08-14，S12 后排队）。

动机：跨族一致率（S9 0.84-0.93 / S11 运行中）的上限受评分器自身噪声限制。
若评分器对同一响应重复打分都不自洽，则一致率 0.84 可能是噪声而非真实信号。
本实验对确定性抽样的 E2B 响应，每个评分器（qwen32/judge_big/judge_small）
独立重复打分 2 遍（贪心确定性），测自一致率 + κ。

设计：
  - 抽样：E2B 3600 中按 (E_t, template) 轮转抽 300 条（确定性，跨层覆盖）。
  - 评分：每评分器顺序加载→pass A(300)→pass B(300)→close；增量缓存
    scorers_cache/s13_<scorer>_<pass>.jsonl，崩溃可续。
  - 分析：自一致率、κ（boot CI）、跨 pass 决策翻转；同样本跨族一致率对照。

判据：自一致率 ≥0.95 → 评分器高度稳定（0.84 一致率主要反映评分器间差异）；
<0.95 → 披露评分器自身存在噪声，跨族一致率需结合噪声上限解读。
零人工标注；只写 results/gpu1_pipeline/s13_* + report/；不碰主账本。

用法：CUDA_VISIBLE_DEVICES=1 python gpu1_s13_test_retest.py [--smoke] [--n 300]
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
    print("[s13 %s] %s" % (Path(__file__).stem, m), flush=True)


def _round_robin(rows, k, seed=20260815):
    """按 (E_t, template_idx) 轮转确定性抽样 k 条。"""
    np.random.seed(seed)
    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[(r["E_t"], r["template_idx"])].append(r)
    keys = sorted(by_key)
    out = []
    qi = 0
    guard = 0
    while len(out) < k and guard < k * 10:
        key = keys[qi % len(keys)]
        if by_key[key]:
            out.append(by_key[key].pop(0))
        qi += 1
        guard += 1
    return out[:k]


def _score_once(scorer, rows, tag, cache_path, done=None):
    done = dict(done or {})
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                done[rec["i"]] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
    labels = []
    for i, r in enumerate(rows):
        if i in done:
            labels.append(done[i])
            continue
        try:
            res = scorer.score_one(r["prompt"], r["response"] or "")
            label = res.get("label")
        except Exception as e:  # noqa: BLE001
            _log("[%s] idx=%d 失败: %s" % (tag, i, str(e)[:120]))
            label = None
        labels.append(label)
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"i": i, "label": label},
                               ensure_ascii=False) + "\n")
        if (i + 1) % 100 == 0:
            _log("[%s] %d/%d" % (tag, i + 1, len(rows)))
    return labels


def _agreement(a, b):
    from scorer_utils import cohens_kappa
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None]
    if not pairs:
        return None
    agree = sum(1 for x, y in pairs if x == y) / len(pairs)
    try:
        kres = cohens_kappa([x for x, _ in pairs], [y for _, y in pairs])
        kappa, ci95 = kres["kappa"], kres["ci95"]
    except Exception:  # noqa: BLE001
        kappa, ci95 = None, None
    flips = sum(1 for x, y in pairs if x != y)
    return {"n": len(pairs), "self_agreement": round(agree, 4),
            "kappa": round(kappa, 4) if kappa is not None else None,
            "kappa_ci95": [round(v, 4) for v in ci95] if ci95 is not None else None,
            "n_flips": flips,
            "pass_0_95": agree >= 0.95,
            "verdict": "稳定" if agree >= 0.95 else "存在自身噪声"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        out_dir / "scorers_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    e2b_path = root / "responses" / "P1_PILOT" / "gemma_4_e2b_responses.jsonl"
    rows = [json.loads(l) for l in open(e2b_path, encoding="utf-8")]
    n = 6 if args.smoke else args.n
    sample = _round_robin(rows, n)
    _log("抽样=%d（E_t0=%d, E_t1=%d）" % (
        len(sample), sum(1 for r in sample if r["E_t"] == 0),
        sum(1 for r in sample if r["E_t"] == 1)))

    def _read_cache(p):
        out = {}
        for line in p.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                out[rec["i"]] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
        return out

    def _cache_full(p, need):
        """缓存行数覆盖所需索引（>=need）才算完备——smoke(n=6) 写的缓存对 full(n=300) 不算。"""
        if not p.exists():
            return False
        try:
            return len(_read_cache(p)) >= need
        except Exception:  # noqa: BLE001
            return False

    results = {}
    # ---- qwen32（先，最大）----
    qw_a_cache, qw_b_cache = cache_dir / "s13_qwen32_a.jsonl", \
        cache_dir / "s13_qwen32_b.jsonl"
    if not (_cache_full(qw_a_cache, len(sample))
            and _cache_full(qw_b_cache, len(sample))):
        qw = s9.Qwen32Scorer(s9._discover_awq(), batch_size=8)
    else:
        qw = None
    if qw is not None:
        _log("qwen32 pass A")
        pairs_a = [(r["prompt"], r["response"] or "") for r in sample]
        res_a = qw.score_batch(pairs_a)
        with qw_a_cache.open("w", encoding="utf-8") as f:
            for i, x in enumerate(res_a):
                f.write(json.dumps({"i": i, "label": x.get("label")},
                                   ensure_ascii=False) + "\n")
        _log("qwen32 pass B")
        res_b = qw.score_batch(pairs_a)
        with qw_b_cache.open("w", encoding="utf-8") as f:
            for i, x in enumerate(res_b):
                f.write(json.dumps({"i": i, "label": x.get("label")},
                                   ensure_ascii=False) + "\n")
        qw.close()
        qw = None
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    def _read_cache(p):
        out = {}
        for line in p.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                out[rec["i"]] = rec["label"]
            except Exception:  # noqa: BLE001
                continue
        return out
    qw_a = [_read_cache(qw_a_cache).get(i) for i in range(len(sample))]
    qw_b = [_read_cache(qw_b_cache).get(i) for i in range(len(sample))]
    results["qwen32"] = _agreement(qw_a, qw_b)
    _log("qwen32 test-retest: %s" % json.dumps(results["qwen32"],
                                               ensure_ascii=False))

    # ---- judge_small / judge_big ----
    s9.register_scorers(cfg)
    for sn in ("judge_small", "judge_big"):
        a_path = cache_dir / ("s13_%s_a.jsonl" % sn)
        b_path = cache_dir / ("s13_%s_b.jsonl" % sn)
        sc = s9.SCORER_FACTORIES[sn]()
        lab_a = _score_once(sc, sample, "%s(A)" % sn, a_path)
        lab_b = _score_once(sc, sample, "%s(B)" % sn, b_path)
        sc.close()
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        results[sn] = _agreement(lab_a, lab_b)
        _log("%s test-retest: %s" % (sn, json.dumps(results[sn],
                                                    ensure_ascii=False)))

    # ---- 同样本跨族对照（pass A）----
    from scorer_utils import cohens_kappa
    def cross(jb, js, qw):
        pairs = []
        for a, b, c in zip(jb, js, qw):
            if a is not None and b is not None and a == b and c is not None:
                pairs.append((int(a), int(c)))
        if not pairs:
            return None
        agree = sum(1 for x, y in pairs if x == y) / len(pairs)
        return {"n": len(pairs), "agreement_dual_vs_qwen32": round(agree, 4),
                "pass_0_80": agree >= 0.80}
    # 需要 judge 标签：从 cache 读 pass A
    jb_a = [_read_cache(cache_dir / "s13_judge_big_a.jsonl").get(i)
            for i in range(len(sample))]
    js_a = [_read_cache(cache_dir / "s13_judge_small_a.jsonl").get(i)
            for i in range(len(sample))]
    cross_fam = cross(jb_a, js_a, qw_a)
    results["cross_family_on_sample"] = cross_fam
    _log("跨族(抽样) : %s" % json.dumps(cross_fam, ensure_ascii=False))

    overview = {
        "stage": "S13", "date": "2026-08-14", "n_sampled": len(sample),
        "sampling": "按 (E_t, template) 轮转确定性抽样（E2B 响应）",
        "test_retest": results,
        "note": ("test-retest 自一致率锚定跨族一致率的噪声上限：评分器自身不稳定"
                 "则一致率部分来自噪声。≥0.95 表明跨族一致率主要反映评分器间"
                 "真实差异。"),
    }
    with open(out_dir / "s13_test_retest.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)
    md = render_md(overview)
    (out_dir / "s13_test_retest.md").write_text(md, encoding="utf-8")
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "s13_test_retest.md").write_text(md, encoding="utf-8")
    _log("已落盘 s13_test_retest.json/.md")

    print(json.dumps({"stage": "S13", "n_sampled": len(sample),
                      "test_retest": results}, ensure_ascii=False, indent=2))
    return 0


def render_md(o):
    lines = [
        "# S13：评分器 test-retest 稳定性（GPU1 · 2026-08-14）\n",
        "## 目的",
        "锚定跨族一致率（S9/S11）的评分器自身噪声上限：若评分器对同一响应重复打分"
        "不自洽，则一致率 0.84 可能部分来自噪声。\n",
        "## 设计",
        "- 抽样：E2B 3600 中按 (E_t, template) 轮转抽 %d 条（确定性）。" % o["n_sampled"],
        "- 每评分器独立重复打分 2 遍（贪心确定性），计算自一致率 + κ（boot 95%%CI）。\n",
        "## 结果",
        "| 评分器 | n | 自一致率 | κ | κ 95%%CI | 翻转 | 判定 |",
        "|---|---|---|---|---|---|---|",
    ]
    for sn, r in o["test_retest"].items():
        if sn == "cross_family_on_sample":
            continue
        lines.append("| %s | %d | %.4f | %s | %s | %d | %s |" % (
            sn, r["n"], r["self_agreement"],
            r["kappa"] if r["kappa"] is not None else "N/A",
            r["kappa_ci95"] if r.get("kappa_ci95") else "N/A",
            r["n_flips"], r["verdict"]))
    cf = o["test_retest"].get("cross_family_on_sample")
    if cf:
        lines.append("\n## 同样本跨族对照（pass A）")
        lines.append("- dual_judge vs qwen32: %.4f（n=%d）%s" % (
            cf["agreement_dual_vs_qwen32"], cf["n"],
            "→ 收敛" if cf["pass_0_80"] else "→ 未收敛"))
    lines.append("\n## 判读")
    lines.append("> 若各评分器自一致率 ≥0.95：评分器高度稳定，跨族一致率主要反映"
                 "评分器间真实差异（测量可信）；若有 <0.95 者，如实披露其噪声，"
                 "并在解读跨族一致率时结合该上限。")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
