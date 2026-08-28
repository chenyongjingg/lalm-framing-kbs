#!/usr/bin/env python3
"""GPU1 S7：Dawid-Skene 无金标准恢复模拟（CPU，2026-08-12 用户批准）。

目的：论文「无金标准测量」依赖 DS 潜类别 EM 的可靠性。本模拟在**实测评分器
误差率区间**下（gate P0_scorers.json：judge_big sens=0.938/spec=0.787、
harmbench sens=0.963/spec=0.708、strongreject sens=0.777/spec=0.699、
gemma sens=0.856/spec=0.686、cross_check sens=0.993/spec=0.331）测
DS 潜标签恢复率与误差率估计偏差 → 支撑方法学主张。

纯 CPU，不占 GPU1，与 GPU 流水线并行；零人工标注。
只写 results/gpu1_pipeline/s7_ds_simulation.{json,md} + report/s7_dawid_skene_simulation.md
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path

# 实测评分器区间（gate P0_scorers.json，602 行人工标注基准）
MEASURED = [
    ("judge_big", 0.9377, 0.7872),
    ("harmbench", 0.9634, 0.7082),
    ("strongreject", 0.7766, 0.6991),
    ("gemma", 0.8561, 0.6862),
    ("cross_check", 0.9927, 0.3313),
]
TYPICAL = (np.mean([m[1] for m in MEASURED]),
           np.mean([m[2] for m in MEASURED]))  # (sens, spec) 均值


def run_sim(n_items, p_pos, raters, n_rep=50, seed=42):
    """一次配置的多重复现。raters = [(sens, spec), ...]。
    返回 dict：潜标签恢复率、误差率估计偏差等。"""
    from scorer_utils import dawid_skene
    rng = np.random.RandomState(seed)
    R = len(raters)
    latent_accs, sens_errs, spec_errs = [], [], []
    for _ in range(n_rep):
        y = (rng.rand(n_items) < p_pos).astype(float)
        mat = np.full((n_items, R), np.nan)
        for j, (sens, spec) in enumerate(raters):
            for i in range(n_items):
                if y[i] == 1:
                    mat[i, j] = 1 if rng.rand() < sens else 0
                else:
                    mat[i, j] = 0 if rng.rand() < spec else 1
        res = dawid_skene(mat)
        est = res["item_label"]
        latent_accs.append((est == y).mean())
        sens_errs.append(np.mean(np.abs(res["sensitivity"]
                                        - np.array([r[0] for r in raters]))))
        spec_errs.append(np.mean(np.abs(res["specificity"]
                                        - np.array([r[1] for r in raters]))))
    return {
        "latent_label_recovery_mean": float(np.mean(latent_accs)),
        "latent_label_recovery_min": float(np.min(latent_accs)),
        "sens_est_mae": float(np.mean(sens_errs)),
        "spec_est_mae": float(np.mean(spec_errs)),
        "n_rep": n_rep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-rep", type=int, default=50)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 复现性：禁 Date/random 以外依赖，直接用 numpy RandomState(seed)
    n_items = 180
    p_pos = 0.5
    rng_base = args.seed

    configs = [
        # 实测合奏（5 评分器）
        ("measured_5", [ (s, sp) for _, s, sp in MEASURED ], n_items, p_pos),
        # 实测核心 4 票（去掉 cross_check——它 spec 极低）
        ("measured_core4",
         [(s, sp) for n_, s, sp in MEASURED if n_ != "cross_check"],
         n_items, p_pos),
        # 典型误差率（均值）下 R=2/3/4/5/6
        ("typical_r2", [TYPICAL] * 2, n_items, p_pos),
        ("typical_r3", [TYPICAL] * 3, n_items, p_pos),
        ("typical_r4", [TYPICAL] * 4, n_items, p_pos),
        ("typical_r6", [TYPICAL] * 6, n_items, p_pos),
        # 类失衡（有害占比低）——实际 jailbreak 场景
        ("typical_r4_p02", [TYPICAL] * 4, n_items, 0.2),
        # 小样本 + 实测
        ("measured_5_n60", [ (s, sp) for _, s, sp in MEASURED ], 60, p_pos),
    ]

    results = []
    for cfg_name, raters, ni, pp in configs:
        r = run_sim(ni, pp, raters, n_rep=args.n_rep,
                    seed=rng_base + len(results))
        r.update({"config": cfg_name, "n_items": ni, "p_pos": pp,
                  "n_raters": len(raters)})
        results.append(r)
        print(f"[S7] {cfg_name}: recovery={r['latent_label_recovery_mean']:.4f} "
              f"(min {r['latent_label_recovery_min']:.4f}) sens_mae="
              f"{r['sens_est_mae']:.4f} spec_mae={r['spec_est_mae']:.4f}")

    overall = {
        "stage": "S7", "n_rep": args.n_rep, "seed": args.seed,
        "measured_scorers": [n_ for n_, _, _ in MEASURED],
        "typical_sens_spec": list(TYPICAL),
        "results": results,
    }
    with open(out_dir / "s7_ds_simulation.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    lines = [
        "# S7 Dawid-Skene 无金标准恢复模拟（GPU1 空闲期测量验证 · 2026-08-12）\n",
        f"- 复现数/配置: {args.n_rep}（seed={args.seed}）",
        f"- 实测评分器区间: " +
        "; ".join(f"{n_}(sens={s},spec={sp})" for n_, s, sp in MEASURED),
        f"- 典型误差率（均值）: sens={TYPICAL[0]:.3f}, spec={TYPICAL[1]:.3f}\n",
        "| 配置 | 评分器数 | 项数 | P(有害) | 潜标签恢复率(均值) | 恢复率(min) | 敏感度MAE | 特异度MAE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']} | {r['n_raters']} | {r['n_items']} | "
            f"{r['p_pos']} | {r['latent_label_recovery_mean']:.4f} | "
            f"{r['latent_label_recovery_min']:.4f} | {r['sens_est_mae']:.4f} | "
            f"{r['spec_est_mae']:.4f} |")
    lines += [
        "\n## 判读",
        "> 恢复率 ≈ 潜标签作为「无金标准真值」代理的可靠度；误差率 MAE = "
        "DS 对各评分器能力估计的偏差。论文正文按此表支撑「以共识替代真值」",
        "> 局限：DS 假设评分器错误条件独立、两类先验均衡；真实场景偏离时恢复率"
        "下移（如 cross_check 低 spec 使 measured_5 恢复率低于核心 4 票）。",
        "> 纯 CPU 模拟，不占 GPU1；零人工标注。",
    ]
    (root / "report" / "s7_dawid_skene_simulation.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"[S7] report written: report/s7_dawid_skene_simulation.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
