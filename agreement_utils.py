"""
agreement_utils.py — 评分者一致性增强模块（κ=0.0568 修复核心）

背景：原 stage_p0a.py 只算 Cohen's κ(keyword, HarmBench)，
在极端边际分布（ASR 0% 或 100%）下 Cohen's κ 会崩（pe→1, 分母→0），
且 keyword 与 LLM 评分器本就不同质，κ 低不代表测量失败。

本模块提供：
  1. gwet_ac1(a, b)      — Gwet's AC1：对极端边际分布稳健的一致性指标
  2. balanced_kappa(a,b) — 平衡 κ：对少数类过采样后重算，消除类别不平衡伪影
  3. agreement_matrix(df) — 三评分器两两一致性矩阵（κ + AC1 + 观察一致率）
  4. kappa_interpret(k)  — Landis & Koch 解释

用法：
  from agreement_utils import agreement_matrix, gwet_ac1
  res = agreement_matrix(df)   # df 含 keyword_label/harmbench_label/sr_score_1_5
"""
import numpy as np
import pandas as pd


def _clean_pair(a, b):
    """对齐两个标签序列，剔除 NaN/None，返回 int 数组对。"""
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    mask = ~(pd.isna(a) | pd.isna(b))
    a, b = a[mask].astype(int), b[mask].astype(int)
    return a, b


def gwet_ac1(labels_a, labels_b):
    """Gwet's AC1：类不平衡下比 Cohen's κ 稳健的一致性指标。

    AC1 = (po - pe_gwet) / (1 - pe_gwet)
    pe_gwet = (1 / (K-1)) * Σ_k π_k (1 - π_k)    # π_k = P(raters choose k | not chance)
    Gwet 2008: 对 κ 的"期望一致率高估"修正。
    """
    a, b = _clean_pair(labels_a, labels_b)
    n = len(a)
    if n == 0:
        return {"ac1": float("nan"), "n": 0, "po": float("nan")}
    po = (a == b).mean()
    # 类别集合
    cats = np.unique(np.concatenate([a, b]))
    K = len(cats)
    if K < 2:
        return {"ac1": 0.0, "n": n, "po": po}  # 单类无意义
    # Gwet 的 pe: 每类"非偶然一致"概率之和
    pe = 0.0
    for c in cats:
        pa = (a == c).mean()
        pb = (b == c).mean()
        pe += pa * pb
    # Gwet AC1 公式：pe_gwet = (1/(K-1)) * Σ π_k(1-π_k)，
    # 其中 π_k = (pa_k + pb_k)/2 为平均边际
    pi_k = []
    for c in cats:
        pi_k.append(((a == c).mean() + (b == c).mean()) / 2)
    pi_k = np.array(pi_k)
    pe_gwet = (1.0 / (K - 1)) * np.sum(pi_k * (1 - pi_k))
    ac1 = (po - pe_gwet) / (1 - pe_gwet) if pe_gwet < 1 else 0.0
    return {"ac1": float(ac1), "n": n, "po": float(po)}


def balanced_kappa(labels_a, labels_b, n_boot=10000, seed=42):
    """平衡 Cohen's κ：对少数类 up-sample 至与多数类等量后重算。

    解决极端边际分布（ASR 0%/100%）下 κ 分母崩塌问题。
    返回 bootstrap 95% CI。
    """
    a, b = _clean_pair(labels_a, labels_b)
    n = len(a)
    if n == 0:
        return {"kappa": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}

    def _kappa(x, y):
        po = (x == y).mean()
        px, py = x.mean(), y.mean()
        pe = px * py + (1 - px) * (1 - py)
        return (po - pe) / (1 - pe) if pe < 1 else 0.0

    # 合并类别（取并集，处理 0/1 二元）
    cats = np.unique(np.concatenate([a, b]))
    if len(cats) != 2:
        return {"kappa": _kappa(a, b), "ci95": [None, None], "n": n}

    # 按 a 的类别平衡（对少数类 up-sample）
    c0, c1 = cats
    n0 = (a == c0).sum()
    n1 = (a == c1).sum()
    n_max = max(n0, n1)
    rng = np.random.default_rng(seed)

    def _balanced(a_, b_):
        a_ = np.asarray(a_); b_ = np.asarray(b_)
        idx0 = np.where(a_ == c0)[0]
        idx1 = np.where(a_ == c1)[0]
        # 若某类在子集中缺失，直接返回原始 κ（无可平衡）
        if len(idx0) == 0 or len(idx1) == 0:
            return _kappa(a_, b_)
        # up-sample 少数类到 n_max
        if len(idx0) < n_max:
            idx0 = rng.choice(idx0, n_max, replace=True)
        if len(idx1) < n_max:
            idx1 = rng.choice(idx1, n_max, replace=True)
        a_bal = np.concatenate([a_[idx0], a_[idx1]])
        b_bal = np.concatenate([b_[idx0], b_[idx1]])
        return _kappa(a_bal, b_bal)

    kappa_bal = _balanced(a, b)
    # bootstrap CI（对平衡后的配对重采样）
    idx = np.arange(n)
    boots = []
    for _ in range(n_boot):
        s = rng.choice(idx, n, replace=True)
        boots.append(_balanced(a[s], b[s]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"kappa": float(kappa_bal), "ci95": [float(lo), float(hi)], "n": n}


def agreement_matrix(df):
    """三评分器两两一致性矩阵。

    输入 df 需含列:
      keyword_label    (0/1)
      harmbench_label  (0/1)
      sr_score         (0-1 归一化，scorer_utils._parse_result 输出；或 sr_label 0/1)
    输出 dict: {
      'pairs': [ {pair, kappa, ac1, po, n}, ... ],
      'n': total,
    }
    注（v6.5.28-fix）：本模块当前未接线（死代码）。二值化边界统一为
    scorer_utils 口径（score_norm>=0.5 ⇔ score>=3 判有害），避免未来接线时
    score=3 归属相反（原 >3.0 判安全 与 scorer_utils >=3 判有害冲突）。
    """
    if "sr_label" in df.columns:
        sr_bin = df["sr_label"].astype(float)
    else:
        sr_bin = np.where(df["sr_score"].astype(float) >= 0.5, 1, 0)

    pairs_def = [
        ("keyword vs HarmBench", df["keyword_label"], df["harmbench_label"]),
        ("HarmBench vs StrongREJECT", df["harmbench_label"], sr_bin),
        ("keyword vs StrongREJECT", df["keyword_label"], sr_bin),
    ]
    results = []
    for name, x, y in pairs_def:
        a, b = _clean_pair(x, y)
        if len(a) == 0:
            results.append({"pair": name, "n": 0, "kappa": None,
                            "ac1": None, "po": None})
            continue
        # 标准 Cohen's κ
        po = (a == b).mean()
        px, py = a.mean(), b.mean()
        pe = px * py + (1 - px) * (1 - py)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
        # Gwet AC1
        ac1 = gwet_ac1(a, b)["ac1"]
        results.append({
            "pair": name, "n": int(len(a)),
            "kappa": round(float(kappa), 4),
            "ac1": round(float(ac1), 4),
            "po": round(float(po), 4),
        })
    return {"pairs": results, "n": int(len(df))}


def kappa_interpret(k):
    """Landis & Koch 一致性解释。"""
    if k is None:
        return "无数据"
    if k < 0: return "差（低于偶然）"
    if k < 0.2: return "极低"
    if k < 0.4: return "低"
    if k < 0.6: return "中等"
    if k < 0.8: return "良好"
    return "优秀"
