# -*- coding: utf-8 -*-
"""
stage_p3.py — 阶段 P3：统计补强（v6.2，纯计算无 GPU）

依据 v6.2 提示词 / RESEARCH_PROTOCOL §7 / STAGE_CONTRACTS §P3 / config.p3。

内容：
1. 功效分析：P1-FULL 各语言 × 各口径的统计功效（N=200，α=0.05，Bonferroni）
2. TOST 等价检验：±5pp 等价界（config.p3.tost_margin_pp）
3. 序贯扩样计划：O'Brien-Fleming alpha 消耗；功效<0.8 且方向符合预注册才扩
4. bootstrap CI 汇总（10,000 次，全阶段效应量）
5. Cohen's h 效应量

输出：report/power_analysis.md + report/tost_results.md + gates/sequential_plan.json

审计修复（AUDIT #174 / CODE_SCIENCE_REPORT §6 P1-7，2026-08-13）：
- M-3.1 n 真值：读 p1_full.queries_n_zh/en + p1_pilot.queries_n（fail-closed，
  删除原不存在键 p1_full.n_per_condition 的静默回退 200）。
- M-3.2 TOST 符号：统一 diff = p_story − p_base（原 p1/p2 为 base−story，
  与展示 Δ=story−base 符号相反）。
- M-3.3 Bonferroni：TOST 等价判据 p_tost < α/M（M=比较数），原硬编码 <0.05。
- M-3.4 配对化：新增 McNemar 配对检验 + paired_tost（McNemar SE = sqrt(b+c)/n），
  配对单位=query（同 query 跨 condition 重复测量）；parquet 缺失才用独立近似并
  显式披露。Δ_cell（单元格级点估计）与 Δ_pair（query 多数票级配对差）分列报告。
- M-3.5 bootstrap CI 主评分器：_calibers 以 _primary_label_col(root) 开头
  （原硬编码 hb/sr 漏 judge_big 主评分器）。

审计修复（CODE_SCIENCE_REPORT §6 P1-10 / DATA_VALIDATION_REPORT，2026-08-13）：
- R2 中文 FNR 披露：PILOT 全为中文 → 中文评分器 FNR 危机（harmbench 1.0 /
  strongreject 0.993 / gemma 0.96，仅 dual_judge 0.293 可辩护）→ PILOT 混合
  效应主标签列取 dual_judge_label（存在时），judge_big/majority 仅敏感性；
  落盘 label_disclosure。
- R3 响应长度协变量：主效应模型加 resp_len（响应字符数，logit 线性项），排除
  "篇幅→可判害"代理效应；并报告含/不含 len 的效应对比（长度敏感性分析）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from common_utils import Checkpoint, JsonlLogger, load_config, setup_logging

STAGE = "p3"


def cohens_h(p1: float, p2: float) -> float:
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def power_two_prop(n: int, p1: float, p2: float, alpha: float) -> float:
    from scipy.stats import norm
    p_bar = (p1 + p2) / 2
    se_null = np.sqrt(2 * p_bar * (1 - p_bar) / n)
    se_alt = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    z_a = norm.ppf(1 - alpha / 2)
    diff = abs(p1 - p2)
    power = 1 - norm.cdf((z_a * se_null - diff) / se_alt) + \
        norm.cdf((-z_a * se_null - diff) / se_alt)
    return float(power)


def _paired_table(base_bins, story_bins) -> dict:
    """从配对二元序列构建 2×2 表 (a,b,c,d)。
    b = 基线安全/故事有害（0→1），c = 基线有害/故事安全（1→0）。
    配对设计下 McNemar 只依赖分歧格 b、c（审计 M-3.4）。"""
    n = len(base_bins)
    a = sum(1 for x, y in zip(base_bins, story_bins) if x == 1 and y == 1)
    b = sum(1 for x, y in zip(base_bins, story_bins) if x == 0 and y == 1)
    c = sum(1 for x, y in zip(base_bins, story_bins) if x == 1 and y == 0)
    d = n - a - b - c
    return {"n_pairs": n, "a": a, "b": b, "c": c, "d": d}


def mcnemar_test(base_bins, story_bins) -> dict:
    """配对二元 McNemar 检验（Yates 连续性校正，χ²(df=1)）。
    H0：基线/故事判害率相同（b=c）。返回 p、χ² 与分歧格计数。"""
    from scipy.stats import chi2
    t = _paired_table(base_bins, story_bins)
    b, c = t["b"], t["c"]
    if (b + c) == 0:
        return {"p": None, "chi2": None, "n_discordant": 0, **t}
    chi2v = (abs(b - c) - 1) ** 2 / (b + c)  # 连续性校正（Yates）
    p = 1 - chi2.cdf(chi2v, 1)
    return {"p": round(float(p), 4), "chi2": round(float(chi2v), 4),
            "n_discordant": int(b + c), **t}


def paired_tost(base_bins, story_bins, margin: float, alpha: float) -> dict:
    """配对二元 TOST 等价检验（McNemar SE，审计 M-3.4/M-3.3）。
    diff = p_story − p_base = (b−c)/n；se = sqrt(b+c)/n。
    等价判据 p_tost < alpha（alpha 已含 Bonferroni 校正 M=比较数）。
    返回 diff/se/ci90/ci95/p_tost/equivalent/ci90_bonf（100(1−2α)% CI）。"""
    from scipy.stats import norm
    t = _paired_table(base_bins, story_bins)
    n = t["n_pairs"]
    diff = (t["b"] - t["c"]) / n
    se = np.sqrt(t["b"] + t["c"]) / n if (t["b"] + t["c"]) > 0 else 0.0
    if se == 0 or n == 0:
        return {"diff": float(diff), "se": float(se),
                "equivalent": bool(abs(diff) < margin), "p_tost": 0.0,
                "ci90": [float(diff), float(diff)],
                "ci95": [float(diff), float(diff)]}
    z_alpha = norm.ppf(1 - alpha)  # 100(1−2α)% CI（Bonferroni 控制下 α=α/M）
    z90, z95 = norm.ppf(0.95), norm.ppf(0.975)
    p_lo = norm.cdf((diff - margin) / se)    # H0: diff ≥ +margin
    p_hi = norm.cdf((-diff - margin) / se)   # H0: diff ≤ −margin
    p_tost = max(p_lo, p_hi)
    return {"diff": float(diff), "se": float(se),
            "equivalent": bool(p_tost < alpha), "p_tost": round(float(p_tost), 4),
            "ci90": [float(diff - z90 * se), float(diff + z90 * se)],
            "ci95": [float(diff - z95 * se), float(diff + z95 * se)],
            "ci90_bonf": [float(diff - z_alpha * se), float(diff + z_alpha * se)],
            "alpha_used": float(alpha)}


def tost_two_prop(p_story: float, p_base: float, n: int,
                  margin: float, alpha: float = 0.05) -> dict:
    """独立样本二比例 TOST（**非配对近似**；配对数据有 parquet 时请用 paired_tost）。
    diff = p_story − p_base（统一符号，审计 M-3.2）。alpha 含 Bonferroni 校正。
    仅在配对数据不可用时作保守近似，并显式标记 independent_approx=True 披露。"""
    from scipy.stats import norm
    diff = p_story - p_base
    se = np.sqrt(p_story * (1 - p_story) / n + p_base * (1 - p_base) / n)
    if se == 0:
        return {"diff": float(diff), "equivalent": bool(abs(diff) < margin),
                "p_tost": 0.0, "ci90": [float(diff), float(diff)],
                "independent_approx": True}
    p_lo = norm.cdf((diff - margin) / se)
    p_hi = norm.cdf((-diff - margin) / se)
    p_tost = max(p_lo, p_hi)
    ci90 = [float(diff - norm.ppf(0.95) * se), float(diff + norm.ppf(0.95) * se)]
    return {"diff": float(diff), "equivalent": bool(p_tost < alpha),
            "p_tost": float(p_tost), "ci90": ci90, "independent_approx": True}


def sequential_alpha(alpha_total: float, n_looks: int) -> list:
    from math import sqrt
    from scipy.stats import norm
    z_total = norm.ppf(1 - alpha_total / 2)
    boundaries = []
    for k in range(1, n_looks + 1):
        # v6.5.28-fix（第四轮审查 🔴）：O'Brien-Fleming 边界**早期严格、晚期宽松**
        # （z_k = z_total * sqrt(K/k)，随信息分数 t=k/K 递减消耗 alpha）。原实现
        # `sqrt(k/K)` 颠倒（早期宽松、晚期严格）→ 提前停止 alpha 膨胀（K=2,α=0.05
        # 首看正确 p≈0.0056，原实现给 0.166，宽约 30 倍）。
        z_k = z_total * sqrt(n_looks / k)
        p_k = 2 * (1 - norm.cdf(z_k))
        boundaries.append(round(float(p_k), 5))
    return boundaries


def bootstrap_ci(vals: np.ndarray, n_boot: int = 10000,
                 seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    if len(vals) < 10:
        return None
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() * 100
             for _ in range(n_boot)]
    boots = np.sort(boots)
    return [round(float(boots[int(0.025 * n_boot)]), 2),
            round(float(boots[int(0.975 * n_boot)]), 2)]


def _query_level_pairs(sub, label_col, cond_col="condition",
                       base="baseline", story="storytelling"):
    """从配对 parquet 子表构建 (query_id) → (base_bin, story_bin) 多数票二元对。

    每 query 在每 condition 下的多数票 = 该 query 该 condition 内跨模型/模板的
    label 均值（>0.5 → 1）。配对单位 = query_id（同 query 跨 condition 重复测量，
    审计 M-3.4）。样本不足返回 (None, None)（如实降级独立近似）。
    """
    if "query_id" not in sub.columns or label_col not in sub.columns:
        return None, None
    s = sub.dropna(subset=[label_col])
    if len(s) < 10:
        return None, None

    def _eq(v, ref):
        try:
            return str(v) == str(ref)
        except Exception:  # noqa: BLE001
            return v == ref

    mask_base = s[cond_col].map(lambda x: _eq(x, base))
    mask_story = s[cond_col].map(lambda x: _eq(x, story))
    gb = s[mask_base].groupby("query_id")[label_col].mean()
    gs = s[mask_story].groupby("query_id")[label_col].mean()
    common = gb.index.intersection(gs.index)
    if len(common) < 5:
        return None, None
    base_bins = (gb[common] > 0.5).astype(int).tolist()
    story_bins = (gs[common] > 0.5).astype(int).tolist()
    return base_bins, story_bins


def _pairs_for_group(source, group, full_df, pilot_df, main_models, primary_col):
    """按比较组返回配对二元序列 (base_bins, story_bins) 或 (None, None)。

    P1-FULL：与 stage_p1_full 的 N_x_lang/N_x_model 同过滤条件
    （main_mask = text × lang∈{zh,en} × model∈主模型）。
    P1-PILOT：text 模态，N∈{0,1}（同 150 query 跨 N 配对）。
    """
    if source == "P1-FULL":
        if full_df is None:
            return None, None
        if group.startswith("lang_"):
            lang = group.split("_", 1)[1]
            sub = full_df[(full_df["modality"] == "text")
                          & (full_df["lang"] == lang)
                          & (full_df["model"].isin(main_models))]
            return _query_level_pairs(sub, primary_col)
        if group.startswith("model_"):
            mdl = group.split("_", 1)[1]
            sub = full_df[(full_df["modality"] == "text")
                          & (full_df["lang"].isin(["zh", "en"]))
                          & (full_df["model"] == mdl)]
            return _query_level_pairs(sub, primary_col)
        return None, None
    if source == "P1-PILOT":
        if pilot_df is None:
            return None, None
        pdf = pilot_df[pilot_df.get("modality", "text") == "text"] \
            if "modality" in pilot_df.columns else pilot_df
        return _query_level_pairs(pdf, primary_col, cond_col="N",
                                  base=0, story=1)
    return None, None


# ---------------------------------------------------------------------------
# 混合效应回归（协议 P1P-3 / P1F-3 / P3-3）
# 真实实现：statsmodels BinomialBayesMixedGLM（logit 链接，随机截距按 query_id）
# 失败自动降级：全交互 → 二阶交互 → 主效应，并披露实际使用的公式。
# ---------------------------------------------------------------------------

def _primary_label_col(root: Path) -> str:
    """从 P0 闸门读主评分器标签列（与 stage_p1_pilot/full 同口径）。"""
    # v6.5.23-fix（问题 94，2026-08-08）：colmap 原仅 {harmbench, strongreject,
    # gemma}——服务器 P0_scorers.json 实测 primary=judge_big（Gemma-4-E4B-it，
    # acc=0.8538）不在 colmap → 恒返回 hb_label（HarmBench），混合效应/三口径
    # 主标签列与 P0 实际主评分器错位。修复：补全 judge_big / judge_small /
    # dual_judge。注意 E2B（judge_small）评分数据在 parquet 中存于 v6.4 遗留
    # 列名 judge_mistral_label（问题 79 记录：功能正确且贯穿消费链，暂不重命名），
    # 故 judge_small/judge_mistral 均映射到该真实列。
    colmap = {"harmbench": "hb_label", "strongreject": "sr_label",
              "gemma": "gemma_label", "judge_big": "judge_big_label",
              "judge_small": "judge_mistral_label",
              "judge_mistral": "judge_mistral_label",
              "dual_judge": "dual_judge_label"}
    p0 = root / "gates" / "P0_scorers.json"
    if p0.exists():
        try:
            p = json.loads(p0.read_text(encoding="utf-8")).get("primary")
            if p in colmap:
                return colmap[p]
        except Exception:  # noqa: BLE001
            pass
    return "hb_label"


def fit_logit_mixed(df, label_col, formula_templates, vc_formula,
                    log) -> dict:
    """BinomialBayesMixedGLM 拟合 logit 混合效应模型。

    formula_templates: 公式列表，按序尝试（全交互 → 降阶）。**公式只含固定效应
        项**——随机项一律经 vc_formula 声明（Patsy 随机语法 (1|..) 在
        BinomialBayesMixedGLM 主公式中会抛 PatsyError，见 v6.5.14 问题 15）。
    vc_formula: 随机效应分组列名，str（单随机截距，如 "query_id"）或
        list[str]（多随机截距，如 ["query_id", "model", "template_idx"]，
        v6.5.25 决策 D5 全混合效应）。
    返回统一 dict；任何失败返回 {"fitted": False, "reason": ...}，不抛出。
    """
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception as e:  # noqa: BLE001
        return {"fitted": False, "reason": f"statsmodels 不可用: {e}"}
    import pandas as pd  # noqa: PLC0415
    from scipy.stats import norm  # noqa: PLC0415

    # v6.5.25-fix（决策 D5）：vc_formula 支持 str 或 list[str]
    vc_groups = [vc_formula] if isinstance(vc_formula, str) else list(vc_formula)

    data = df.dropna(subset=[label_col]).copy()
    if len(data) < 50:
        return {"fitted": False,
                "reason": f"样本不足 {len(data)} < 50（统计功效无意义）"}
    # 标签列统一命名为 label（公式语义清晰），因子列统一字符串
    data = data.rename(columns={label_col: "label"})
    for c in ["E_t", "N", "R", "A_s", "model", "template_idx",
              "query_id", "lang", "condition"]:
        if c in data.columns:
            data[c] = data[c].astype(str)
    data["query_id"] = data["query_id"].astype(str)
    if data["label"].dropna().empty:
        return {"fitted": False, "reason": f"标签列 {label_col} 无有效值"}

    last_err = None
    for formula in formula_templates:
        try:
            log.info("MixedGLM 尝试公式: %s vc=%s", formula, vc_groups)
            # vc_formulas 是位置参数（dict: 随机效应列名 → 分组公式），
            # 随机截距用 "1"（每组独立截距），见 statsmodels 0.14.6 API
            vc_dict = {g: "1" for g in vc_groups if g in data.columns}
            if not vc_dict:
                return {"fitted": False,
                        "reason": f"vc 分组列均不在数据中: {vc_groups}"}
            model = BinomialBayesMixedGLM.from_formula(formula, vc_dict, data)
            # fit_vb(mean, sd, fit_method, minim_opts, scale_fe, verbose)
            # 无 n_vb/gd_iters 参数（0.14.6 签名）；BFGS 优化 + scale_fe 稳定
            res = model.fit_vb(fit_method="BFGS", scale_fe=True)
            # 0.14.6：params 是 ndarray；fep_names 是固定效应名；
            # cov_params() 返回 Series（键为参数名）
            params = res.params
            fep_names = list(getattr(model, "fep_names", None) or [])
            n_fe = len(fep_names)
            # 0.14.6：params = [固定效应...] + [随机效应方差...]；
            # fep_names 仅固定效应；vcp_names 是分组方差名（随机效应按分组）
            try:
                cov = res.cov_params()
                if isinstance(cov, dict):
                    cov = pd.Series(cov)
                bse_by_name = {}
                if cov is not None:
                    if isinstance(cov, pd.Series):
                        for k, v in cov.items():
                            bse_by_name[str(k)] = float(np.sqrt(float(v)))
                    else:
                        cov_a = np.asarray(cov, dtype=float)
                        if cov_a.ndim == 2 and cov_a.shape[0] == n_fe:
                            for i, nm in enumerate(fep_names):
                                bse_by_name[nm] = float(np.sqrt(cov_a[i, i]))
                bse = [bse_by_name.get(nm, float("nan"))
                       for nm in fep_names]
            except Exception:  # noqa: BLE001
                bse = [float("nan")] * n_fe
            fixed = []
            for i, name in enumerate(fep_names):
                if i >= len(params):
                    break
                coef = float(params[i])
                b = bse[i] if i < len(bse) else float("nan")
                if b and not np.isnan(b) and b > 0:
                    z = coef / b
                    p = 2 * (1 - norm.cdf(abs(z)))
                    lo, hi = coef - 1.96 * b, coef + 1.96 * b
                else:
                    z, p, lo, hi = float("nan"), float("nan"), float("nan"), float("nan")
                fixed.append({
                    "param": name,
                    "coef": round(coef, 4),
                    "or": round(float(np.exp(coef)), 4),
                    "ci95": [round(float(np.exp(lo)), 4),
                             round(float(np.exp(hi)), 4)],
                    "bse": round(b, 4),
                    "z": round(float(z), 3),
                    "p": round(float(p), 4),
                })
            rand = {}
            try:
                vcp_names = list(getattr(model, "vcp_names", None) or [])
                # params 尾段 = 各分组随机效应方差（vcp_names）
                tail = params[n_fe:] if n_fe < len(params) else params
                for i, nm in enumerate(vcp_names):
                    if i < len(tail):
                        rand[nm] = round(float(tail[i]), 4)
            except Exception:  # noqa: BLE001
                pass
            # v6.5.28-fix（M2，审查发现 2026-08-09）：vc_formula 为 list（多随机
            # 截距，如 ["query_id","model","template_idx"]）时 `data[list]` 返回
            # DataFrame，`.nunique()` 返回 Series，`int(Series)` 抛 TypeError →
            # 全混合效应分支恒失败（"全部公式拟合失败"→ 降级）。按首个分组列计。
            _vc_key = vc_groups[0] if isinstance(vc_formula, list) else vc_formula
            return {
                "fitted": True,
                "n_obs": int(len(data)),
                "n_groups": int(data[_vc_key].nunique()),
                "formula_used": formula,
                "vc_formula": f"{{'{_vc_key}': '1'}}",
                "method": "BinomialBayesMixedGLM (variational Bayes, BFGS)",
                "fit_details": f"fit_vb(fit_method='BFGS', scale_fe=True)",
                "fixed_effects": fixed,
                "random_variance": rand,
            }
        except Exception as e:  # noqa: BLE001
            last_err = str(e)[:300]
            log.warning("公式拟合失败: %s → %s", formula, last_err)
            continue
    return {"fitted": False,
            "reason": f"全部公式拟合失败: {last_err}"}


def _len_sensitivity(fit_with_len, fit_no_len, param_substr="[T.storytelling]"):
    """R3 长度敏感性对比：含/不含 resp_len 时主效应系数的变化。

    排除"篇幅→可判害"代理效应（DATA_VALIDATION_REPORT R3）：若加入响应长度
    协变量后主效应系数/OR 大幅衰减或翻转 → 效应主要源自长度共变而非框架本身，
    论文须如实披露为长度敏感（长度敏感性分析）。任一拟合失败 → None + 披露。

    返回 dict{param, or_with_len, or_no_len, delta_or, sensitivity} 或
    None（无法对比）。
    """
    def _find_or(fit, substr):
        if not isinstance(fit, dict) or not fit.get("fitted"):
            return None
        for fe in fit.get("fixed_effects") or []:
            if substr in (fe.get("param") or ""):
                return fe.get("or")
        return None

    or_a = _find_or(fit_with_len, param_substr)
    or_b = _find_or(fit_no_len, param_substr)
    if or_a is None or or_b is None:
        return None
    delta = float(or_a) - float(or_b)
    # 敏感判据：|ΔOR| > 0.2 或方向翻转（OR 跨 1）
    flip = (float(or_a) - 1) * (float(or_b) - 1) < 0
    sensitive = abs(delta) > 0.2 or flip
    return {"param": param_substr,
            "or_with_len": round(float(or_a), 4),
            "or_no_len": round(float(or_b), 4),
            "delta_or": round(delta, 4),
            "sensitive": bool(sensitive),
            "criterion": "|ΔOR|>0.2 或 OR 跨 1（方向翻转）→ 长度敏感",
            "note": ("ΔOR 显著 → 响应长度共变主效应（代理效应未排除），"
                     "须如实披露") if sensitive else "长度非主效应代理（稳健）"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["workdir"]).expanduser()
    log = setup_logging(str(root / "logs" / f"{STAGE}.log"), STAGE)
    jlog = JsonlLogger(str(root / "logs" / f"{STAGE}.jsonl"))
    ckpt = Checkpoint(str(root / "checkpoints" / f"{STAGE}.json"))

    log.info("=== P3 统计补强（v6.2）===")
    jlog.event(stage=STAGE, event="start")

    margin_pp = cfg.get("p3", {}).get("tost_margin_pp", 5)
    margin = margin_pp / 100
    alpha = 0.05
    max_n = cfg.get("p3", {}).get("sequential", {}).get("max_n", 500)

    # ---- 1. n 配置（fail-closed，审计 M-3.1）----
    # 原实现读不存在键 p1_full.n_per_condition → 静默回退 200（功效/样本断言虚假）。
    # 正确键：queries_n_zh / queries_n_en（各语言 200/条件）；缺失即 fail-closed。
    p1f_n_zh = cfg.get("p1_full", {}).get("queries_n_zh")
    p1f_n_en = cfg.get("p1_full", {}).get("queries_n_en")
    p1p_n = cfg.get("p1_pilot", {}).get("queries_n")
    if p1f_n_zh is None or p1f_n_en is None:
        raise RuntimeError(
            "config 缺 p1_full.queries_n_zh / queries_n_en（审计 M-3.1 fail-closed；"
            "n_per_condition 键不存在，禁止静默回退 200）")
    if p1p_n is None:
        raise RuntimeError("config 缺 p1_pilot.queries_n（审计 M-3.1 fail-closed）")
    lang_n = {"zh": int(p1f_n_zh), "en": int(p1f_n_en)}
    model_n = int(p1f_n_zh) + int(p1f_n_en)  # 模型级比较池化中英（每条件 N=400）

    # ---- 1b. 加载配对 parquet（存在才加载；缺失则配对统计降级独立近似并披露）----
    import pandas as pd  # noqa: PLC0415
    full_df = pilot_df = None
    _pf = root / "results" / "p1_full_scored.parquet"
    if _pf.exists():
        try:
            full_df = pd.read_parquet(_pf)
        except Exception as e:  # noqa: BLE001
            log.warning("p1_full_scored.parquet 读取失败（配对统计降级独立近似）: %s", e)
    _pp = root / "results" / "p1_pilot_scored.parquet"
    if _pp.exists():
        try:
            pilot_df = pd.read_parquet(_pp)
        except Exception as e:  # noqa: BLE001
            log.warning("p1_pilot_scored.parquet 读取失败（配对统计降级独立近似）: %s", e)
    primary_col = _primary_label_col(root)
    _models_cfg = cfg.get("models", {}) or {}
    main_models = {m for m, v in _models_cfg.items()
                   if (v or {}).get("role") != "architecture_control_only"}

    def _add_cmp(source, group, base, story, n_base, n_story, effect_pp):
        pairs = _pairs_for_group(source, group, full_df, pilot_df,
                                 main_models, primary_col)
        comp = {
            "source": source, "group": group,
            "p_base": base / 100, "p_story": story / 100,
            "n_base": int(n_base), "n_story": int(n_story),
            "effect_pp": round(effect_pp, 2),
            "paired": pairs is not None,
        }
        if pairs is not None:
            comp["base_bins"], comp["story_bins"] = pairs
        comparisons.append(comp)

    comparisons = []
    skipped = []  # 字段缺失被跳过的比较（必须落盘披露，严禁静默）
    p1f_stats = root / "results" / "p1_full_stats.json"
    if p1f_stats.exists():
        p1f = json.loads(p1f_stats.read_text(encoding="utf-8"))
        for lang, v in (p1f.get("N_x_lang") or {}).items():
            if v.get("effect_pp") is None:
                continue
            asr_base, asr_story = v.get("asr_baseline"), v.get("asr_storytelling")
            if asr_base is None or asr_story is None:
                skipped.append(f"P1-FULL lang_{lang}: asr_baseline/asr_storytelling 缺失")
                continue
            n_cond = lang_n.get(lang)
            if n_cond is None:
                skipped.append(f"P1-FULL lang_{lang}: 无 lang 级 n 配置（queries_n_zh/en）")
                continue
            _add_cmp("P1-FULL", f"lang_{lang}", asr_base, asr_story,
                     n_cond, n_cond, v["effect_pp"])
        for mdl, v in (p1f.get("N_x_model") or {}).items():
            if v.get("effect_pp") is None:
                continue
            asr_base, asr_story = v.get("asr_baseline"), v.get("asr_storytelling")
            if asr_base is None or asr_story is None:
                skipped.append(f"P1-FULL model_{mdl}: asr_baseline/asr_storytelling 缺失")
                continue
            _add_cmp("P1-FULL", f"model_{mdl}", asr_base, asr_story,
                     model_n, model_n, v["effect_pp"])
    # P1-PILOT 主评分器
    p1p = root / "results" / "p1_pilot_effects.json"
    if p1p.exists():
        eff = json.loads(p1p.read_text(encoding="utf-8"))
        n_main = eff.get("N_main", {}) or {}
        pv = n_main.get("primary")
        if isinstance(pv, dict) and pv.get("effect_pp") is not None:
            asr_n0, asr_n1 = pv.get("asr_n0"), pv.get("asr_n1")
            if asr_n0 is None or asr_n1 is None:
                skipped.append("P1-PILOT N_main_primary: asr_n0/asr_n1 缺失")
            else:
                _add_cmp("P1-PILOT", "N_main_primary", asr_n0, asr_n1,
                         p1p_n, p1p_n, pv["effect_pp"])
    for s in skipped:
        log.warning("跳过比较（字段缺失）: %s", s)
        jlog.event(stage=STAGE, event="skip_comparison", reason=s)
    log.info("比较对: %d（跳过 %d）", len(comparisons), len(skipped))

    # ---- 2. 功效分析 + TOST + Cohen's h ----
    # 配对设计（同 query 跨 condition）：McNemar 配对检验 + paired TOST（配对 SE）。
    # 功效公式为独立样本二比例保守近似（配对正相关 → 真实功效 ≥ 此值，方向安全），
    # 如实标注，不作配对功效虚报（审计 M-3.4）。
    m_total = max(len(comparisons), 1)
    alpha_bonf = alpha / m_total
    if not comparisons:
        # 输入缺失或字段不齐 → 必须如实披露，严禁空报告冒充"无比较"
        _reason = ("无可用比较对：p1_full_stats.json / p1_pilot_effects.json 缺失，"
                   "或全部比较字段不齐（见 logs/p3.jsonl skip_comparison）")
        jlog.event(stage=STAGE, event="empty_comparisons",
                   reason=_reason, n_skipped=len(skipped))
        log.warning("功效/TOST 无比较对：%s", _reason)
    pw_lines = ["# 功效分析报告（v6.2）\n",
                f"- α={alpha}（原始）/ α={alpha_bonf:.4f}（Bonferroni，M={m_total}）",
                f"- N={lang_n['zh']}+{lang_n['en']}/条件（P1-FULL 中英各 {lang_n['zh']}）| "
                f"{p1p_n}/条件（P1-PILOT）| 等价界 ±{margin_pp}pp\n",
                f"- 功效为独立样本二比例保守近似（配对正相关 → 真实功效 ≥ 此值）\n\n",
                "| 来源 | 组 | p_baseline | p_story | Cohen's h | "
                "功效(α) | 功效(Bonf α) |\n",
                "|---|---|---|---|---|---|---|\n"]
    tost_lines = ["# TOST 等价检验报告（v6.2）\n",
                  f"等价界: ±{margin_pp}pp | Bonferroni α={alpha_bonf:.4f}（M={m_total}）\n\n",
                  "| 来源 | 组 | Δ_cell(pp) | Δ_pair(pp) | 90% CI(配对) | TOST p | 等价? | 方法 |\n",
                  "|---|---|---|---|---|---|---|---|\n",
                  "> Δ_cell=单元格级 ASR 差（点估计，与 P1-FULL/PILOT 统计一致）；"
                  "Δ_pair=query 多数票级配对差（(b−c)/n，McNemar 估计量）。两者权重不同，"
                  "分别报告，不混淆。\n"]
    if not comparisons:
        pw_lines.append(f"\n> {_reason}\n")
        tost_lines.append(f"\n> {_reason}\n")
    need_extend = False
    for c in comparisons:
        h = cohens_h(c["p_story"], c["p_base"])
        pw = power_two_prop(c["n_base"], c["p_base"], c["p_story"], alpha)
        pw_b = power_two_prop(c["n_base"], c["p_base"], c["p_story"], alpha_bonf)
        pw_lines.append(f"| {c['source']} | {c['group']} | {c['p_base']:.3f} | "
                        f"{c['p_story']:.3f} | {h:.3f} | {pw:.3f} | {pw_b:.3f} |\n")
        # TOST：配对数据 → paired_tost（McNemar SE）；否则独立近似（显式披露）
        if c["paired"]:
            ts = paired_tost(c["base_bins"], c["story_bins"], margin, alpha_bonf)
            mc = mcnemar_test(c["base_bins"], c["story_bins"])
            method_tag = "配对 McNemar/paired-TOST"
            d_pair_pp = ts["diff"] * 100
        else:
            ts = tost_two_prop(c["p_story"], c["p_base"], c["n_base"],
                               margin, alpha_bonf)
            mc = None
            method_tag = "独立近似（parquet 缺失，披露）"
            d_pair_pp = c["effect_pp"]
        tost_lines.append(f"| {c['source']} | {c['group']} | {c['effect_pp']:.1f} "
                          f"| {d_pair_pp:.1f} "
                          f"| [{ts['ci90'][0] * 100:.1f}, "
                          f"{ts['ci90'][1] * 100:.1f}] | {ts['p_tost']:.4f} | "
                          f"{'是' if ts['equivalent'] else '否'} | {method_tag} |\n")
        if mc is not None and mc.get("p") is not None:
            tost_lines.append(
                f"|  └ McNemar χ²={mc['chi2']} p={mc['p']} "
                f"（b=基线安全/故事有害 {mc['b']}，c=基线有害/故事安全 {mc['c']}） |\n")
        direction_ok = c["effect_pp"] > 0
        if (not ts["equivalent"]) and pw_b < 0.8 and direction_ok:
            need_extend = True
    (root / "report" / "power_analysis.md").write_text(
        "".join(pw_lines), encoding="utf-8")
    tost_lines.append(f"\n## 结论\n- 比较数: {len(comparisons)}\n")
    if need_extend:
        tost_lines.append(f"- **建议扩样**: 存在 TOST 不成立 + 功效不足 + "
                          f"方向符合预注册的比较（N 扩展至 {max_n}/条件，"
                          f"O'Brien-Fleming alpha 消耗）\n")
    (root / "report" / "tost_results.md").write_text(
        "".join(tost_lines), encoding="utf-8")
    log.info("功效 + TOST 报告完成（%d 比较）", len(comparisons))

    # ---- 3. 序贯扩样计划 ----
    # 信息分数步长按 P1-FULL 单语言每条件 N 计（M-3.1 后 n 为真值，非回退 200）
    _full_n_cond = min(lang_n["zh"], lang_n["en"])
    n_looks = max(2, int(max_n / max(_full_n_cond, 1)))
    seq_plan = {
        "version": "v6.2",
        "method": "O'Brien-Fleming alpha spending",
        "alpha_total": alpha,
        "max_n_per_condition": max_n,
        "max_looks": n_looks,
        "of_boundaries": sequential_alpha(alpha, n_looks),
        "extend_candidates": [
            {"group": c["group"], "effect_pp": c["effect_pp"]}
            for c in comparisons
            if c["effect_pp"] > 0 and c["effect_pp"] < margin_pp],
        "rule": ("扩样仅当：TOST 不成立 + 功效<0.8 + 方向符合预注册；"
                 "最大 N=500/条件；最终检验用 O'Brien-Fleming 校正 alpha"),
    }
    gates = root / "gates"
    gates.mkdir(parents=True, exist_ok=True)
    (gates / "sequential_plan.json").write_text(
        json.dumps(seq_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("序贯计划: %s", gates / "sequential_plan.json")

    # ---- 4. bootstrap CI 汇总 + 混合效应回归（读评分 parquet）----
    scored = root / "results" / "p1_full_scored.parquet"
    mixed_summary = None
    if not scored.exists():
        # 输入缺失 → 必须落盘披露，严禁静默跳过（最高纪律 2）
        mixed_summary = {
            "label_col": _primary_label_col(root),
            "result": {"fitted": False,
                       "reason": f"评分数据缺失: {scored.name} 不存在（P1-FULL 未跑或未落盘）"},
            "note": "bootstrap CI + 混合效应回归整体跳过；请先完成 P1-FULL 评分",
        }
        jlog.event(stage=STAGE, event="skip_mixed_effects",
                   reason=str(mixed_summary["result"]["reason"]))
        (root / "report" / "mixed_effects.json").write_text(
            json.dumps(mixed_summary, ensure_ascii=False, indent=2),
            encoding="utf-8")
        log.warning("P3 混合效应跳过: %s", mixed_summary["result"]["reason"])
    else:
        import pandas as pd  # noqa: PLC0415
        df = pd.read_parquet(scored)
        ci_rows = []
        # 主评分器口径优先（审计 M-3.5：原硬编码 hb/sr 漏 judge_big 主评分器），
        # 再补三口径其余列。primary 由 P0_scorers.json 决定（_primary_label_col）。
        _primary_col = _primary_label_col(root)
        _calibers = [_primary_col] + [c for c in
                                      ("dual_judge_label", "majority_label",
                                       "hb_label", "sr_label")
                                      if c != _primary_col]
        for cond in ["baseline", "storytelling"]:
            sub = df[(df["modality"] == "text")
                     & (df["condition"] == cond)]
            for col in _calibers:
                if col in sub.columns:
                    vals = sub[col].dropna().to_numpy(dtype=float)
                    ci = bootstrap_ci(vals,
                                      n_boot=cfg.get("scorers", {}).get(
                                          "n_bootstrap", 10000),
                                      seed=cfg.get("seeds", {}).get(
                                          "bootstrap", 42))
                    ci_rows.append({"condition": cond, "scorer": col,
                                    "asr": round(vals.mean() * 100, 2),
                                    "ci95": ci})
        # v6.7-r5-fix（终审 Major G）：行级 bootstrap 是单条件水平估计，非
        # query-clustered、非配对 delta——CI 不得被解读为"条件间差异"或已考虑
        # 同 query 重复测量。如实标注口径于元数据。
        for _r in ci_rows:
            _r["ci_scope"] = ("行级（row-level），单条件 ASR 水平估计；未按 "
                              "query_id 聚类，非同 query 配对 delta——CI 仅反映 "
                              "该条件 ASR 抽样误差，不得用于条件间差异推断")
        (root / "report" / "bootstrap_ci_summary.json").write_text(
            json.dumps(ci_rows, ensure_ascii=False, indent=2),
            encoding="utf-8")
        log.info("bootstrap CI 汇总: %d 行（Major G：行级非聚类非 delta 已披露）",
                 len(ci_rows))

        # ---- 4b. 混合效应回归（协议 P1F-3：≥3 模型 × ≥3 模板 → 全混合效应）----
        # v6.5.28-fix（P3-1，审查发现 2026-08-09）：P1-FULL parquet 是 condition
        # 级设计（baseline/storytelling/unrestricted × lang），无 E_t/N/R/A_s 列；
        # 原公式引用不存在列 → PatsyError → 恒失败（死代码，FULL 统计补强不交付）。
        # 改 FULL 因子 condition × lang；并过滤主模型（排除架构对照 qwen）+ 中英
        # （排除 adv OOD，协议 §7/§6.2）。
        _main_models = [k for k, v in cfg.get("models", {}).items()
                        if v.get("role") != "architecture_control_only"]
        text_df = df[(df["modality"] == "text")
                     & (df["model"].isin(_main_models))
                     & (df["lang"].isin(["zh", "en"]))].copy()
        label_col = _primary_label_col(root)
        n_models = text_df["model"].nunique() if "model" in text_df.columns else 0
        n_tpl = (text_df["template_idx"].nunique()
                 if "template_idx" in text_df.columns else 0)
        mixed = None
        if len(text_df) >= 50 and n_models >= 3 and n_tpl >= 3:
            # 全混合效应：FULL 因子 = condition（framing）× lang + model/template
            # R3-fix（审计 v6.5.29，DATA_VALIDATION_REPORT R3）：响应长度与操纵
            # 因子系统性共变（R=1 +37.5% / N=1 +19.9%）→ 主效应模型加 resp_len
            # 协变量（logit 侧线性项），排除"篇幅→可判害"代理效应；并报告
            # 含/不含 len 的效应对比（长度敏感性分析）。
            _has_resp = "response" in text_df.columns
            if _has_resp:
                text_df["resp_len"] = text_df["response"].fillna("").astype(str).str.len()
            _len_term = " + resp_len" if _has_resp else ""
            formulas = [
                f"label ~ C(condition)*C(lang) + C(model) + C(template_idx){_len_term}",
                f"label ~ C(condition) + C(lang) + C(model) + C(template_idx){_len_term}",
                f"label ~ C(condition) + C(model) + C(template_idx){_len_term}",
            ]
            mixed = fit_logit_mixed(text_df, label_col, formulas,
                                    vc_formula="query_id",
                                    log=log)
            # R3 长度敏感性：不含 len 的对照拟合（仅当响应列存在）
            mixed_no_len = None
            if _has_resp:
                formulas_no_len = [
                    "label ~ C(condition)*C(lang) + C(model) + C(template_idx)",
                    "label ~ C(condition) + C(lang) + C(model) + C(template_idx)",
                    "label ~ C(condition) + C(model) + C(template_idx)",
                ]
                mixed_no_len = fit_logit_mixed(text_df, label_col,
                                               formulas_no_len,
                                               vc_formula="query_id",
                                               log=log)
            len_sens = _len_sensitivity(mixed, mixed_no_len)
            mixed_summary = {
                "label_col": label_col,
                "n_models": int(n_models), "n_templates": int(n_tpl),
                "n_obs": int(len(text_df)),
                "result": mixed,
                "length_covariate": {
                    "included": bool(_has_resp),
                    "column": "resp_len" if _has_resp else None,
                    "note": ("响应长度字符数，logit 线性协变量（R3）；"
                             "无 response 列则未纳入") if _has_resp else
                            ("响应列缺失 → 未加 resp_len 协变量，须披露为"
                             "长度共变未控制（R3 限制）"),
                },
                "length_sensitivity": len_sens,
                "note": ("协议 P1F-3 全混合效应（≥3 模型 × ≥3 模板）+ R3 "
                         "响应长度协变量；P1P-3 混合效应在 PILOT parquet 上"
                         "同构拟合（见下）"),
            }
        elif len(text_df) >= 50:
            mixed_summary = {
                "label_col": label_col,
                "n_models": int(n_models), "n_templates": int(n_tpl),
                "result": {"fitted": False,
                           "reason": "模型/模板数不足 3（协议 P1F-3 升级条件未满足）"},
                "note": "描述性统计 + bootstrap CI 仍然产出",
            }
        else:
            mixed_summary = {"label_col": label_col,
                             "result": {"fitted": False,
                                        "reason": "评分数据不足（<50 行）"}}
        # PILOT 混合效应（P1P-3：全因子 8 组合 × A_s 3 档）
        pilot_scored = root / "results" / "p1_pilot_scored.parquet"
        pilot_mixed = None
        if pilot_scored.exists():
            try:
                pdf = pd.read_parquet(pilot_scored)
                p_text = pdf[pdf.get("modality", "text") == "text"] \
                    if "modality" in pdf.columns else pdf
                if len(p_text) >= 50 and all(c in p_text.columns
                                             for c in ["E_t", "N", "R", "A_s",
                                                       "query_id"]):
                    # R2-fix（审计 v6.5.29，DATA_VALIDATION_REPORT R2）：PILOT
                    # 全为中文数据 → 中文评分器 FNR 危机（harmbench 1.0 /
                    # strongreject 0.993 / gemma 0.96，仅 dual_judge 0.293
                    # 可辩护）→ 中文主效应仅采用 dual_judge 共识口径；judge_big
                    # 单口径/4 票多数仅作敏感性分析（论文披露 zh FNR 表 + 单口径
                    # 政策）。故 PILOT 混合效应 label 列取 dual_judge_label
                    # （存在时），不沿用英文基准选的 primary=judge_big。
                    _pilot_label = label_col
                    _pilot_label_disclosure = None
                    if "dual_judge_label" in p_text.columns:
                        _pilot_label = "dual_judge_label"
                        _pilot_label_disclosure = (
                            "R2：PILOT 中文数据主效应仅采用 dual_judge 共识口径"
                            "（zh FNR=0.293 可辩护；harmbench 1.0/strongreject "
                            "0.993/gemma 0.96 单口径不作主测量），judge_big/"
                            "majority 仅作敏感性分析")
                        log.warning("R2：PILOT 混合效应 label=%s（中文 FNR 披露）",
                                    _pilot_label)
                    # R3-fix（审计 v6.5.29）：PILOT 混合效应同样加 resp_len 协变量
                    _p_has_resp = "response" in p_text.columns
                    if _p_has_resp:
                        p_text["resp_len"] = p_text["response"].fillna("").astype(str).str.len()
                    _p_len_term = " + resp_len" if _p_has_resp else ""
                    p_formulas = [
                        f"label ~ E_t*N*R*A_s + C(model) + C(template_idx){_p_len_term}",
                        f"label ~ E_t + N + R + A_s + C(model) + C(template_idx){_p_len_term}",
                        f"label ~ N + C(model) + C(template_idx){_p_len_term}",
                    ]
                    pilot_mixed = fit_logit_mixed(
                        p_text, _pilot_label, p_formulas,
                        vc_formula="query_id", log=log)
                    _pilot_no_len = None
                    if _p_has_resp:
                        _pilot_no_len = fit_logit_mixed(
                            p_text, _pilot_label,
                            ["label ~ E_t*N*R*A_s + C(model) + C(template_idx)",
                             "label ~ E_t + N + R + A_s + C(model) + C(template_idx)",
                             "label ~ N + C(model) + C(template_idx)"],
                            vc_formula="query_id", log=log)
                    _pilot_len_sens = _len_sensitivity(
                        pilot_mixed, _pilot_no_len,
                        param_substr="N[" if "N[" in str(
                            (pilot_mixed or {}).get("fixed_effects")) else "N[T.")
                    pilot_len_note = {
                        "length_covariate": {
                            "included": bool(_p_has_resp),
                            "column": "resp_len" if _p_has_resp else None,
                            "note": ("R3：PILOT 混合效应纳入响应长度协变量"
                                     if _p_has_resp else
                                     "响应列缺失 → 未加 resp_len（R3 限制）"),
                        },
                        "length_sensitivity": _pilot_len_sens,
                    }
                else:
                    pilot_mixed = {"fitted": False, "reason": "PILOT 因子列/样本不足"}
            except Exception as e:  # noqa: BLE001
                pilot_mixed = {"fitted": False, "reason": f"PILOT parquet 读取失败: {e}"}
            mixed_summary["pilot"] = {
                "label_col": _pilot_label if "pilot_len_note" in dir() and pilot_len_note else label_col,
                "label_disclosure": _pilot_label_disclosure,
                "result": pilot_mixed,
                "length_covariate": pilot_len_note.get("length_covariate")
                if "pilot_len_note" in dir() and pilot_len_note else None,
                "length_sensitivity": pilot_len_note.get("length_sensitivity")
                if "pilot_len_note" in dir() and pilot_len_note else None,
                "note": ("协议 P1P-3：logit(ASR) ~ E_t×N×R×A_s + model + "
                         "template + len + (1|query)（R3 含长度协变量；"
                         "R2 中文主效应仅 dual_judge 口径）"),
            }
        (root / "report" / "mixed_effects.json").write_text(
            json.dumps(mixed_summary, ensure_ascii=False, indent=2),
            encoding="utf-8")
        log.info("混合效应回归完成: fitted=%s reason=%s",
                 mixed_summary["result"].get("fitted"),
                 mixed_summary["result"].get("reason"))

    jlog.event(stage=STAGE, event="done", n_comparisons=len(comparisons),
               need_extend=need_extend)
    ckpt.mark_done("done")
    code = 0 if not need_extend else 2
    log.info("=== P3 完成（code=%d）===", code)
    return code


if __name__ == "__main__":
    sys.exit(main())
