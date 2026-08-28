# -*- coding: utf-8 -*-
"""stage_p2_crossframe_baselines.py — 跨 5 帧外部基线评估（纯 CPU，08-27）

论文主张"融合在每帧都胜过最强开源分类器"需要跨帧基线硬证据。
multi_frame_roc.json 已提供 5 帧融合 ROC；本脚本补齐同 5 帧的外部基线
ROC（ShieldGemma / WildGuard / GradSafe 代理），消除 multi_frame_roc.md
红旗1"基线仅 seed0 同帧，跨帧基线未测"限制。

输入（全部既有产物，零新推理、零 GPU）：
  results/msrf_evaluation.json              feats/labels/seed_detail(每帧 te_idx)
  responses/P2B/shieldgemma_scores.jsonl    全量 16200 行（idx 键控）
  responses/P2B/wildguard_scores.jsonl      全量 16200 行
  results/multi_frame_roc.json              5 帧融合指标（直接复用）
  results/cross_method_roc_auc.json         seed0 同帧对齐值（校验用）
输出：
  results/cross_frame_baselines.json
  report/cross_frame_baselines.md

验证策略：seed0 帧复算值必须与 cross_method_roc_auc.json 对齐
（shieldgemma 0.1983 / wildguard 0.1737 / proxy 0.9652 / real 0.3152），
对齐通过后跨帧数值才可信；偏差 >1e-2 则报 FAIL 不写结果。
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/lalm_framing_revision_v6")
STAGE = "P2-CROSSFRAME"

# 复用 stage_p2_baselines 的指标函数（与既有评估同口径保证）
sys.path.insert(0, str(ROOT))
from stage_p2_baselines import _pr_roc_metrics, gradsafe_scores_from_features  # noqa: E402


def _safe_label(v):
    """feats 行的 label 可能是 None/float('nan')/数字 → 归一化为 int 或 None。"""
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv):
        return None
    return int(fv)


def _load_scores(path):
    """读全量打分 jsonl → {idx: 0.0/1.0}；容错坏行（C5-fix 口径）。"""
    score_by_idx = {}
    n_corrupt = 0
    n_invalid = 0
    if not path.exists():
        return score_by_idx, n_corrupt, n_invalid
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            n_corrupt += 1
            continue
        if r.get("unsafe") is None:
            n_invalid += 1
            continue
        score_by_idx[r["idx"]] = 0.0 if not r["unsafe"] else 1.0
    return score_by_idx, n_corrupt, n_invalid


def _filtered_sy(rows):
    """与 _pr_roc_metrics 相同的 NaN 标签过滤 → (s, y)。"""
    s = np.asarray([x["gradsafe_score"] for x in rows], dtype=float)
    y = np.asarray([x["label"] for x in rows], dtype=float)
    valid = ~np.isnan(y)
    return s[valid], y[valid]


def roc_auc_with_ci(rows, seed, n_boot=1000):
    """ROC-AUC 点估计 + bootstrap 95% CI [lo, pt, hi]（对齐 cross_method 格式）。"""
    s, y = _filtered_sy(rows)
    if len(s) == 0 or len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    pt = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(float(roc_auc_score(y[idx], s[idx])))
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "roc_auc": round(pt, 4),
        "roc_ci": [round(float(lo), 4), round(pt, 4), round(float(hi), 4)],
        "n_eval": int(len(y)),
        "n_pos": int((y == 1).sum()),
    }


def _pr_metrics(rows):
    """PR-AUC / TPR@FPR5 / benign_FPR / ECE（复用 stage_p2_baselines 同款）。"""
    labels = np.asarray([x["label"] for x in rows], dtype=float)
    return _pr_roc_metrics(rows, labels, fpr_fixed=0.05)


def main() -> int:
    ev = ROOT / "results" / "msrf_evaluation.json"
    evalj = json.loads(ev.read_text(encoding="utf-8"))
    feats = evalj.get("feats") or evalj.get("features") or []
    labels = np.asarray(evalj.get("labels") or [], dtype=float)
    seed_detail = evalj.get("seed_detail") or []
    n_all = len(feats)
    if not seed_detail or not feats:
        print("FAIL: msrf_evaluation.json 缺 feats/seed_detail")
        return 3

    # 融合每帧指标（直接复用 multi_frame_roc.json，不重算）
    mfr = json.loads((ROOT / "results" / "multi_frame_roc.json").read_text(
        encoding="utf-8")) if (ROOT / "results" / "multi_frame_roc.json").exists() else {}
    fusion_by_seed = {}
    for f in mfr.get("frames", []):
        fusion_by_seed[f["seed"]] = {
            "roc_auc": f.get("roc_auc"),
            "pr_auc": f.get("pr_auc"),
            "tpr_at_fpr5": f.get("tpr_at_fpr5"),
            "ece": f.get("ece"),
            "best_lr": f.get("best_lr"),
            "test_n": f.get("test_n"),
            "deployment_frame": f.get("deployment_frame", False),
        }

    # 校验参考（seed0 同帧）
    ref = {}
    cr = ROOT / "results" / "cross_method_roc_auc.json"
    if cr.exists():
        ref = json.loads(cr.read_text(encoding="utf-8"))

    # 全量基线分数
    sg_score, sg_corrupt, sg_invalid = _load_scores(
        ROOT / "responses" / "P2B" / "shieldgemma_scores.jsonl")
    wg_score, wg_corrupt, wg_invalid = _load_scores(
        ROOT / "responses" / "P2B" / "wildguard_scores.jsonl")
    gsr = []
    gsr_f = ROOT / "results" / "gradsafe_real_scores.json"
    if gsr_f.exists():
        try:
            _g = json.loads(gsr_f.read_text(encoding="utf-8"))
            gsr = _g if isinstance(_g, list) else []
        except Exception:  # noqa: BLE001
            gsr = []

    print(f"n_all={n_all} frames={len(seed_detail)} "
          f"sg_rows={len(sg_score)} wg_rows={len(wg_score)} gsr_rows={len(gsr)} "
          f"sg_corrupt={sg_corrupt} wg_corrupt={wg_corrupt}")

    rows_out = []
    validation = {"matched": True, "seed0_ref": {}, "seed0_ours": {}, "diff": {}}

    # te_idx 是有效标签子集（labels 非 NaN 行）内的相对索引（08-27 实证：
    # seed0 经 valid 映射后与 te_mask_seed0 完全全等 3863/3863）。
    valid_rows = [i for i, v in enumerate(labels) if not np.isnan(v)]
    if not valid_rows:
        print("FAIL: 无有效标签行")
        return 3

    for ei, frame in enumerate(seed_detail):
        seed = frame["seed"]
        te_rel = np.asarray(frame.get("te_idx") or [], dtype=int)
        if len(te_rel) == 0:
            continue
        # 相对索引 -> 全量绝对索引
        te = np.asarray([valid_rows[i] for i in te_rel], dtype=int)
        tr = np.setdiff1d(np.arange(n_all), te)
        # 该帧测试行的标签（feats[i]["label"]，NaN 归一）
        f_map = feats
        # ---- ShieldGemma（真实推理，全量已有） ----
        sg_rows = []
        for i in te:
            f = f_map[int(i)]
            fidx = int(f.get("idx", int(i)))
            if fidx not in sg_score:
                continue
            lab = _safe_label(f.get("label"))
            sg_rows.append({
                "gradsafe_score": sg_score[fidx],
                "label": lab,
                "benign": bool(f.get("benign", False)),
            })
        # ---- WildGuard ----
        wg_rows = []
        for i in te:
            f = f_map[int(i)]
            fidx = int(f.get("idx", int(i)))
            if fidx not in wg_score:
                continue
            lab = _safe_label(f.get("label"))
            wg_rows.append({
                "gradsafe_score": wg_score[fidx],
                "label": lab,
                "benign": bool(f.get("benign", False)),
            })
        # ---- GradSafe 代理（每帧训练子集拟合，测试子集评分，同 v6.6.1 口径） ----
        gs_rows = gradsafe_scores_from_features(
            feats, labels, seed=42, fit_idx=tr, score_idx=te)

        # ---- GradSafe 真实（仅 seed0 帧有产物） ----
        gr_rows = []
        if ei == 0:
            for r in gsr:
                sc = r.get("gradsafe_real_score")
                lab = r.get("label")
                if sc is None or lab is None:
                    continue
                gr_rows.append({"gradsafe_score": float(sc),
                                "label": int(lab), "benign": False})

        fr = {
            "seed": seed,
            "deployment_frame": bool(frame.get("deployment_frame", False)),
            "test_n": int(len(te)),
            "te_pos_rate": round(float((np.asarray([_safe_label(f_map[int(i)].get("label")) for i in te], dtype=float) == 1).mean()), 4) if len(te) else None,
            "fusion": fusion_by_seed.get(seed, {}),
        }
        for name, rows in (("shieldgemma", sg_rows), ("wildguard", wg_rows)):
            lab = np.asarray([x["label"] for x in rows], dtype=float)
            m = _pr_metrics(rows)
            roc = roc_auc_with_ci(rows, seed=seed + (1 if name == "wildguard" else 0))
            fr[name] = {
                "n_eval": len(rows),
                "metrics": m,
                "roc_auc": roc["roc_auc"] if roc else None,
                "roc_ci": roc["roc_ci"] if roc else None,
                "n_pos": roc["n_pos"] if roc else None,
            }
        lab_gs = np.asarray([x["label"] for x in gs_rows], dtype=float)
        m_gs = _pr_metrics(gs_rows)
        roc_gs = roc_auc_with_ci(gs_rows, seed=seed + 2)
        fr["gradsafe_proxy"] = {
            "n_eval": len(gs_rows),
            "metrics": m_gs,
            "roc_auc": roc_gs["roc_auc"] if roc_gs else None,
            "roc_ci": roc_gs["roc_ci"] if roc_gs else None,
            "n_pos": roc_gs["n_pos"] if roc_gs else None,
        }
        if ei == 0:
            lab_gr = np.asarray([x["label"] for x in gr_rows], dtype=float)
            m_gr = _pr_metrics(gr_rows)
            roc_gr = roc_auc_with_ci(gr_rows, seed=seed + 3)
            fr["gradsafe_real"] = {
                "n_eval": len(gr_rows),
                "metrics": m_gr,
                "roc_auc": roc_gr["roc_auc"] if roc_gr else None,
                "roc_ci": roc_gr["roc_ci"] if roc_gr else None,
                "n_pos": roc_gr["n_pos"] if roc_gr else None,
            }
        else:
            fr["gradsafe_real"] = {
                "n_eval": 0,
                "note": "真实 GradSafe 仅 seed0 帧有推理产物；跨帧需 GPU 重推理（待 P2-C 完成后排队）",
            }
        rows_out.append(fr)

    # ---- seed0 校验：ours vs cross_method_roc_auc.json ----
    if ref and rows_out:
        seed0 = rows_out[0]
        for k in ("shieldgemma", "wildguard", "gradsafe_proxy", "gradsafe_real"):
            ref_roc = (ref.get(k) or {}).get("roc_auc")
            ours_roc = (seed0.get(k) or {}).get("roc_auc")
            if ref_roc is None or ours_roc is None:
                continue
            diff = abs(ours_roc - ref_roc)
            validation["seed0_ref"][k] = ref_roc
            validation["seed0_ours"][k] = ours_roc
            validation["diff"][k] = round(diff, 4)
            if diff > 1e-2:
                validation["matched"] = False
                print(f"[VALIDATION] {k}: ref={ref_roc} ours={ours_roc} diff={diff:.4f} -> MISMATCH")

    print(f"[VALIDATION] matched={validation['matched']} diff={validation['diff']}")
    if not validation["matched"]:
        print("FAIL: seed0 复算与 cross_method_roc_auc.json 偏差>1e-2，不写结果")
        return 2

    # ---- 汇总：融合是否每帧胜过最强基线 ----
    summary = {"n_frames": len(rows_out), "fusion_wins_every_frame": True,
               "best_baseline_per_frame": [], "margins": []}
    for fr in rows_out:
        f_roc = (fr.get("fusion") or {}).get("roc_auc")
        best = None
        best_key = None
        for k in ("shieldgemma", "wildguard", "gradsafe_proxy", "gradsafe_real"):
            v = (fr.get(k) or {}).get("roc_auc")
            if v is None:
                continue
            if best is None or v > best:
                best, best_key = v, k
        margin = (f_roc - best) if (f_roc is not None and best is not None) else None
        summary["best_baseline_per_frame"].append(
            {"seed": fr["seed"], "best_key": best_key, "best_roc": best})
        summary["margins"].append({"seed": fr["seed"], "fusion_roc": f_roc,
                                   "best_baseline_roc": best, "best_key": best_key,
                                   "margin": margin})
        if margin is not None and margin <= 0:
            summary["fusion_wins_every_frame"] = False

    out = {
        "purpose": "跨 5 个独立 GroupShuffleSplit 测试帧的外部基线 ROC（纯 CPU 复评既有分数，零新推理）",
        "method": ("每帧 te_idx 取自 msrf_evaluation.json seed_detail；ShieldGemma/WildGuard 用 "
                   "responses/P2B 全量 16200 打分按 idx 对齐；GradSafe 代理每帧训练子集拟合、"
                   "测试子集评分（v6.6.1 无泄漏口径）；融合指标直接复用 multi_frame_roc.json。"),
        "validation": validation,
        "n_all": n_all,
        "sg_corrupt": sg_corrupt, "sg_invalid": sg_invalid,
        "wg_corrupt": wg_corrupt, "wg_invalid": wg_invalid,
        "frames": rows_out,
        "summary": summary,
        "honest_caveats": [
            "PR-AUC 依赖各帧正例率（pos_rate 逐帧披露）；跨帧对比以 ROC-AUC 与 TPR@FPR5 为主。",
            "GradSafe 代理与 MSRF 共用分支特征空间，非独立开源分类器，论文须与真实推理基线区分标注。",
            "GradSafe 真实仅 seed0 帧有产物（0.3152）；跨帧未测，论文不得做真实 GradSafe 的跨帧断言。",
            "ShieldGemma/WildGuard 为二元 unsafe 打分，ROC 为二分类分数阶梯，跨帧可比。",
        ],
    }
    (ROOT / "results" / "cross_frame_baselines.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- markdown 报告 ----
    md = [f"# 跨 5 帧外部基线评估（P2-CROSSFRAME, {STAGE}，纯 CPU）\n",
          f"- 样本: {n_all} | 帧数: {len(rows_out)}\n",
          f"- 校验: seed0 帧复算与 cross_method_roc_auc.json 对齐"
          f" {'✅' if validation['matched'] else '❌'} diff={validation['diff']}\n",
          "\n## 各帧 ROC-AUC（融合 vs 基线）\n",
          "| 帧 | 测试数 | pos率 | MSRF融合 | ShieldGemma | WildGuard | GradSafe代理 | GradSafe真实 |\n",
          "|---|---|---|---|---|---|---|---|\n"]
    for fr in rows_out:
        def _r(d, k):
            v = (d or {}).get(k)
            if isinstance(v, dict):
                rv = v.get("roc_auc")
                return f"{rv:.4f}" if isinstance(rv, (int, float)) else "—"
            return "—"
        fuz = (fr.get("fusion") or {}).get("roc_auc")
        fuz_s = f"{fuz:.4f}" if isinstance(fuz, (int, float)) else "—"
        md.append(f"| {fr['seed']}{'（部署帧）' if fr['deployment_frame'] else ''} | "
                  f"{fr['test_n']} | {fr['te_pos_rate']:.1%} | "
                  + fuz_s
                  + " | " + " | ".join(
                      _r(fr.get(k), "roc_auc") for k in
                      ("shieldgemma", "wildguard", "gradsafe_proxy", "gradsafe_real"))
                  + " |\n")
    md.append("\n## 每帧胜者与边际\n")
    for m in summary["margins"]:
        md.append(f"- seed {m['seed']}: 融合 {m['fusion_roc']} vs 最强基线 "
                  f"{m['best_baseline_roc']}（{m['best_key'] if 'best_key' in m else ''}），"
                  f"边际 {m['margin']:.4f}\n")
    md.append(f"\n## 结论\n- 融合在全部 {len(rows_out)} 帧胜过最强基线: "
              f"{'✅ 是' if summary['fusion_wins_every_frame'] else '❌ 否'}\n")
    md.append("\n## 诚实披露\n- " + "\n- ".join(out["honest_caveats"]) + "\n")
    rpt = ROOT / "report"
    rpt.mkdir(parents=True, exist_ok=True)
    (rpt / "cross_frame_baselines.md").write_text("".join(md), encoding="utf-8")
    print(f"OK: results/cross_frame_baselines.json + report/cross_frame_baselines.md")
    print(f"    fusion_wins_every_frame={summary['fusion_wins_every_frame']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
