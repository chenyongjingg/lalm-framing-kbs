# -*- coding: utf-8 -*-
"""P1-PILOT 补跑 DualJudge + 三口径重算合并脚本（v6.5 适配版）

背景：历史主评分进程跑旧版 scorer_utils（无 .device 修复），
其 DualJudge 阶段评分会 AttributeError 被吞 → dual_judge 全 None。
本脚本用修复版单独重跑 DualJudge 两轮（v6.5：Gemma-4-E4B + Gemma-4-E2B），
合并进已落盘的 parquet，并完整重算三口径 / N_main / N_x_A_s / 混合效应 /
敏感性，重写全部产物。

用法：python stage_p1_pilot_refill_dual.py --config pipeline_config.yaml
前置：results/p1_pilot_scored.parquet 已存在（主进程跑完落盘）
"""
import argparse, json, sys, time
from pathlib import Path

def _jsonable(o):
    """递归把 numpy 标量转 python 原生类型，保证 json.dumps 可序列化。"""
    import numpy as _np
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, _np.floating):
        return float(o)
    if isinstance(o, _np.bool_):
        return bool(o)
    if isinstance(o, _np.ndarray):
        return _jsonable(o.tolist())
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from common_utils import load_config, setup_logging
    cfg = load_config(args.config)
    root = Path(cfg["workdir"]).expanduser()  # KBS巡检修复: workdir 含字面 ~, 缺 expanduser 致日志路径 FileNotFoundError rc=1
    log = setup_logging(root / "logs" / "p1_pilot_refill.log")
    log.info("=== P1-PILOT 补跑 DualJudge 开始（方案 A）===")

    import pandas as pd
    import numpy as np
    sys.path.insert(0, str(root))
    from scorer_utils import (DualJudgeScorer, compute_three_way_metrics,
                              sensitivity_flip_report)
    # 复用 stage_p1_pilot 的统计工具
    from stage_p1_pilot import _bootstrap_ci, _fit_pilot_mixed

    scored_path = root / "results" / "p1_pilot_scored.parquet"
    if not scored_path.exists():
        log.error("parquet 不存在: %s（主进程评分未完成，先等评分落盘）", scored_path)
        return 1
    df = pd.read_parquet(scored_path)
    log.info("读入 parquet: %d 行, 列=%s", len(df), list(df.columns))

    # v6.5.24-fix：读取既有 effects.json 中的 manipulation_check / cross_validation，
    # 重写产物时透传——否则 G1 的操纵检验硬判据（§5）与异构交叉验证字段丢失，
    # 且缺 version 会触发 gate_g1 的 recalc_v64 反向覆盖（v6.4 口径污染）。
    _orig_eff = {}
    _orig_path = root / "results" / "p1_pilot_effects.json"
    if _orig_path.exists():
        try:
            _orig_eff = json.loads(_orig_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.warning("原 effects.json 读取失败，manipulation_check/cross_validation "
                        "无法透传（以缺失披露）")

    # ── 1) 补跑 DualJudge 两轮（v6.5：Gemma-4-E4B + E2B 顺序加载）──
    if df["judge_big_label"].notna().sum() == 0 or args.dry_run:
        # v6.5.29-fix（第十一轮审查 🔴）：补跑必须用真实 behavior（与主链
        # stage_p1_pilot 同口径）——原传 "" 空上下文使双 judge 在无任务上下文
        # 下判定，污染 dual_judge_label/第 4 票/majority/G1 C4。从冻结 PILOT 文件
        # 构建 query_id→zh 映射（主链同源）。
        _behav_map = {}
        _pf_f = root / "results" / "p1_pilot_queries_full.json"
        if _pf_f.exists():
            try:
                _pf = json.loads(_pf_f.read_text(encoding="utf-8"))
                for _q in _pf.get("queries", []):
                    _behav_map[str(_q.get("query_id"))] = \
                        _q.get("zh") or _q.get("text") or ""
            except Exception:  # noqa: BLE001
                log.warning("冻结 PILOT 文件读取失败，补跑 behavior 将回退空（已披露）")
        def _behav_for(_r):
            _bid = str(_r.get("query_id", ""))
            _b = _behav_map.get(_bid)
            return _b if _b else ""
        log.info("补跑 DualJudge E4B + E2B（%d 条）", len(df))
        dj = DualJudgeScorer(
            cfg["scorers"]["judge_big_model"],
            cfg["scorers"].get("judge_small_model",
                               cfg["scorers"].get("judge_mistral_model")),
            load_in_4bit=False)  # v6.5.24-fix：BF16 直载（§10.4，Gemma-4 基类忽略 4bit 参数）
        # 第一轮 E4B
        big_labels = []
        for i, r in df.iterrows():
            try:
                big_labels.append(dj.score_one_big(_behav_for(r), str(r["response"])))
            except Exception as e:  # noqa: BLE001
                big_labels.append(None)
            if (i + 1) % 1000 == 0:
                log.info("[diag] 补跑 DualJudge-E4B %d/%d", i + 1, len(df))
        dj.unload_big()
        # 第二轮 E2B
        # v6.5.14-fix 2026-08-08（问题 16）：DualJudgeScorer 无 _load_mistral
        # 方法（v6.5 迁移 Mistral-24B → Gemma-4-E2B-it 时方法已删）。
        # 改为 load_mistral_only（先 unload_big 再 _load_small）。
        dj.load_mistral_only()
        mist_labels = []
        for i, r in df.iterrows():
            try:
                mist_labels.append(dj.score_one_mistral(
                    _behav_for(r), str(r["response"])))
            except Exception as e:  # noqa: BLE001
                mist_labels.append(None)
            if (i + 1) % 1000 == 0:
                log.info("[diag] 补跑 DualJudge-E2B %d/%d", i + 1, len(df))
        dj.close()
        df["judge_big_label"] = big_labels
        df["judge_mistral_label"] = mist_labels
        log.info("补跑完成: big valid=%d, mistral valid=%d",
                 df["judge_big_label"].notna().sum(),
                 df["judge_mistral_label"].notna().sum())
    else:
        log.info("judge_big_label 已有 %d 条有效 → 跳过补跑",
                 df["judge_big_label"].notna().sum())

    if args.dry_run:
        log.info("dry-run 结束")
        return 0

    # ── 2) 完整重算三口径（复用 stage_p1_pilot 同款逻辑）──
    # 主评分器：读 P0 闸门
    primary_col = "hb_label"
    p0_gate = root / "gates" / "P0_scorers.json"
    if p0_gate.exists():
        try:
            p0j = json.loads(p0_gate.read_text(encoding="utf-8"))
            p0p = p0j.get("primary")
            colmap = {"judge_big": "judge_big_label", "harmbench": "hb_label",
                      "gemma": "gemma_label", "strongreject": "sr_label"}
            primary_col = colmap.get(p0p, "hb_label")
            if primary_col not in df.columns:
                primary_col = "hb_label"
        except Exception:  # noqa: BLE001
            pass
    log.info("主评分器口径列: %s", primary_col)

    # 三口径标签（4 票制 majority，dual_judge 一致判定）
    lc_cols = [c for c in ["hb_label", "sr_label", "gemma_label",
                           "judge_big_label", "judge_mistral_label"]
               if c in df.columns]
    vote_cols = [c for c in ["hb_label", "sr_label", "gemma_label",
                             "dual_judge_label"] if c in df.columns]
    # v6.6.1-fix（问题 51，与 stage_p1_pilot 问题 50 配套）：不再对
    # hb/sr/gemma 缺失列 fillna(0)——compute_three_way_metrics 的动态门槛
    # （问题 11）已按每行有效票数 notna() 统计；fillna(0) 会把"评分失败"
    # 的 None 重新填成 0（=安全票），虚高 n_valid 抬高门槛且冒充安全样本。
    # None 交由动态门槛/mean(skipna) 如实排除（与主进程 stage_p1_pilot
    # v6.6.1 修复后口径完全一致）。
    if "dual_judge_label" not in df.columns:
        if "judge_big_label" in df.columns and "judge_mistral_label" in df.columns:
            b = df["judge_big_label"]
            m = df["judge_mistral_label"]
            df["dual_judge_label"] = np.where(
                b.notna() & m.notna() & (b == m), b, np.nan)
        else:
            df["dual_judge_label"] = np.nan
    n_scorers = sum(df[c].notna().sum() > 0 for c in vote_cols)
    three = compute_three_way_metrics(
        df, label_cols=vote_cols, primary=primary_col,
        dual_cols=("judge_big_label", "judge_mistral_label"),
        n_boot=cfg.get("scorers", {}).get("n_bootstrap", 10000),
        seed=cfg.get("seeds", {}).get("bootstrap", 42))
    log.info("三口径汇总: %s", three)

    stats = {
        "stage": "P1-PILOT",
        # v6.5.24-fix：必须写 version=v6.5（STAGE_CONTRACTS 契约），否则 gate_g1
        # 检测到非 v6.5 会触发 recalc_v64 用 v6.4 口径反向覆盖本产物
        "version": "v6.5",
        "stats_model": "logit mixed-effects + bootstrap CI",
        "n_queries": int(df["query_id"].nunique()) if "query_id" in df.columns else 0,
        "n_cells": int(len(df)),
        "primary_scorer": primary_col,
        "n_scorers": n_scorers,
        "scorers_n": n_scorers,
        "three_way": three,
        # v6.5.24-fix：透传原进程的操纵检验与异构交叉验证结果（缺失时如实披露）
        "manipulation_check": _orig_eff.get("manipulation_check"),
        "cross_validation": _orig_eff.get("cross_validation"),
    }
    if stats["manipulation_check"] is None:
        log.warning("原 effects.json 无 manipulation_check → 重写产物缺失该字段，"
                    "G1 操纵检验硬判据将判不通过（如实，不伪造通过）")
    if stats["cross_validation"] is None:
        log.warning("原 effects.json 无 cross_validation → 异构交叉验证字段缺失披露")

    # N_main：三口径下 N=1 vs N=0 的 ASR 差 + bootstrap CI
    stats["N_main"] = {"primary": None, "dual_judge": None,
                       "majority": None, "scorers_n": n_scorers}
    for metric, col in [("primary", primary_col),
                        ("dual_judge", "dual_judge_label"),
                        ("majority", "majority_label")]:
        if col not in df.columns or df[col].dropna().empty:
            continue
        asr_n0 = df[df["N"] == 0][col].mean() * 100
        asr_n1 = df[df["N"] == 1][col].mean() * 100
        n_main_eff = asr_n1 - asr_n0
        ci = _bootstrap_ci(
            df, col,
            n_boot=cfg.get("scorers", {}).get("n_bootstrap", 10000),
            seed=cfg.get("seeds", {}).get("bootstrap", 42))
        entry = {
            "direction": "up" if n_main_eff > 0 else
                         ("down" if n_main_eff < 0 else "none"),
            "effect_pp": round(n_main_eff, 2),
            "asr_n0": round(asr_n0, 2), "asr_n1": round(asr_n1, 2),
            "ci": ci, "metric": metric,
        }
        # v6.5.9-fix 2026-08-05：refill 重写 effects 时补严格争议率字段，
        # 与 recalc_v64 同口径（双 judge 均评分样本中不一致比例），
        # 否则 gate_g1 C4 判据（争议率<0.5）只能走"缺失降级"。
        if metric == "dual_judge" and ("judge_big_label" in df.columns
                                       and "judge_mistral_label" in df.columns):
            _b = df["judge_big_label"]
            _m = df["judge_mistral_label"]
            _both = _b.notna() & _m.notna()
            _total = int(_both.sum())
            if _total > 0:
                # v6.5.26-fix（F6，审查发现 2026-08-08）：dispute_rate 分母必须为
                # 双 judge 均评分样本数 _total（原 .mean() 除以全样本 → 系统性偏低，
                # 可把严格争议率 ≥0.5 误判为 <0.5 → 翻转 G1 C4）。与主脚本
                # stage_p1_pilot 口径一致（dispute_n / n_both）。
                entry["dispute_rate"] = round(
                    float(int((_both & (_b != _m)).sum()) / _total), 4)
                entry["dual_coverage"] = round(float(_both.mean()), 4)
                entry["dual_agree_n"] = int((_both & (_b == _m)).sum())
                entry["dual_total_n"] = _total
        stats["N_main"][metric] = entry
    log.info("N_main: %s", stats["N_main"])

    # 混合效应（主评分器口径）
    mixed = None
    try:
        mixed = _fit_pilot_mixed(df, primary_col, log)
    except Exception as e:  # noqa: BLE001
        log.warning("混合效应拟合异常: %s", str(e)[:200])
    stats["mixed_effects"] = mixed

    # N_x_A_s：三口径下 N 效应在 text vs audio 的差异
    stats["N_x_A_s"] = {"primary": None, "dual_judge": None,
                        "majority": None, "scorers_n": n_scorers}
    for metric, col in [("primary", primary_col),
                        ("dual_judge", "dual_judge_label"),
                        ("majority", "majority_label")]:
        if col not in df.columns or df[col].dropna().empty:
            continue
        res = {"asr_text_n0": None, "asr_text_n1": None,
               "asr_audio_n0": None, "asr_audio_n1": None}
        try:
            # v6.6.0-fix: audio 口径与 stage_p1_pilot 主脚本统一——
            # 原实现把 neutral/styled 两水平简单平均（与主脚本 A_s != "text"
            # 合并口径不一致，导致补跑后 N_x_A_s 结果漂移）；改为合并样本计算。
            t_text = df[(df["A_s"] == "text") & (df["N"] == 1)]
            b_text = df[(df["A_s"] == "text") & (df["N"] == 0)]
            t_aud = df[(df["A_s"] != "text") & (df["N"] == 1)]
            b_aud = df[(df["A_s"] != "text") & (df["N"] == 0)]
            if not (t_text.empty or b_text.empty or t_aud.empty or b_aud.empty):
                res["asr_text_n1"] = round(float(t_text[col].mean()) * 100, 2)
                res["asr_text_n0"] = round(float(b_text[col].mean()) * 100, 2)
                res["asr_audio_n1"] = round(float(t_aud[col].mean()) * 100, 2)
                res["asr_audio_n0"] = round(float(b_aud[col].mean()) * 100, 2)
                d_text = res["asr_text_n1"] - res["asr_text_n0"]
                d_audio = res["asr_audio_n1"] - res["asr_audio_n0"]
                res["direction"] = ("audio_stronger" if abs(d_audio) > abs(d_text)
                                    else "text_stronger")
                res["diff_pp"] = round(d_audio - d_text, 2)
            stats["N_x_A_s"][metric] = res
        except Exception as e:  # noqa: BLE001
            log.warning("N_x_A_s %s 计算失败: %s", metric, str(e)[:120])
    log.info("N_x_A_s: %s", stats["N_x_A_s"])

    # both_models / model_heterogeneity_explainable（读原 parquet 已有，或重算）
    stats["both_models"] = {"consistent": None, "note": "补跑后重算"}
    try:
        # v6.6.0-fix: 与 stage_p1_pilot 主脚本口径统一——按"每个模型内 N 效应
        # 方向一致性"判定（原实现用 groupby 全量均值差 <10，口径漂移）。
        dirs = []
        for m in df["model"].unique():
            sub = df[df["model"] == m]
            if len(sub) < 2 or sub[primary_col].dropna().empty:
                continue
            d = (sub[sub["N"] == 1][primary_col].mean()
                 - sub[sub["N"] == 0][primary_col].mean())
            dirs.append("up" if d > 0 else ("down" if d < 0 else "none"))
        if len(dirs) >= 2:
            stats["both_models"] = {
                "consistent": len(set(dirs)) == 1,
                "model_dirs": dirs,
                "note": "各模型内 N 效应方向一致性（与主脚本同口径）",
            }
    except Exception:  # noqa: BLE001
        pass
    nx = stats.get("N_x_A_s", {}) or {}
    prim_nx = nx.get("primary") if isinstance(nx.get("primary"), dict) else {}
    het = None
    if stats["both_models"].get("consistent") is True:
        het = True
    elif stats["both_models"].get("consistent") is False and \
            prim_nx.get("direction") in ("audio_stronger", "text_stronger"):
        het = True
    elif stats["both_models"].get("consistent") is False:
        het = False
    stats["model_heterogeneity_explainable"] = het

    # 敏感性翻转报告
    try:
        sens_groups = {"N_main": {}}
        for metric, col in [("primary", primary_col),
                            ("dual_judge", "dual_judge_label"),
                            ("majority", "majority_label")]:
            if col not in df.columns or df[col].dropna().empty:
                continue
            g = df.groupby("N")[col].mean() * 100
            if len(g) == 2:
                sens_groups["N_main"][f"asr_{metric}_pct"] = round(
                    float(g.loc[1]), 2)
                sens_groups["N_main"].setdefault(
                    "dispute_rate",
                    round(float(df["dual_judge_label"].isna().mean()), 4))
        sensitivity_flip_report(sens_groups,
                                root / "report" / "p1_pilot_sensitivity.md")
        log.info("敏感性报告已重写")
    except Exception as e:  # noqa: BLE001
        log.warning("敏感性报告失败: %s", str(e)[:150])

    # ── 3) 重写全部产物 ──
    df.to_parquet(scored_path)
    log.info("parquet 已重写: %d 行", len(df))
    (root / "results" / "p1_pilot_effects.json").write_text(
        json.dumps(_jsonable(stats), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "gates").mkdir(parents=True, exist_ok=True)
    (root / "gates" / "G2_input.json").write_text(
        json.dumps(_jsonable(stats), ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("effects.json + G2_input.json 已重写")

    # 人类可读报告（沿用 stage_p1_pilot 的骨架 + 效应量）
    md = [
        "# P1-PILOT 配对析因预实验报告（补跑 DualJudge 后）\n",
        f"- 独立 query 集: {stats['n_queries']} 条",
        f"- 设计单元: {stats['n_cells']}",
        f"- 主评分器: {primary_col}",
        f"- 三口径汇总: {json.dumps(_jsonable(three), ensure_ascii=False)}",
        f"- N_main: {json.dumps(_jsonable(stats['N_main']), ensure_ascii=False)}",
        f"- 混合效应: {json.dumps(_jsonable(mixed or {}), ensure_ascii=False)}",
        f"- N_x_A_s: {json.dumps(_jsonable(stats['N_x_A_s']), ensure_ascii=False)}",
        "\n> 由补跑脚本重算（v6.5：原进程 hb/sr/gemma + 补跑 DualJudge E4B/E2B）",
    ]
    (root / "report" / "p1_pilot_stats.md").write_text("\n".join(md), encoding="utf-8")
    log.info("p1_pilot_stats.md 已重写")
    log.info("=== 补跑完成：三口径完整（4 票制）===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
