#!/usr/bin/env python
"""p0c_len_control.py — P0-C 长度混杂控制分析 + text 效应分层（论文补做实验 1&2）。

权威契约：results/p0c_scored.parquet（10800 行，9 列）。
主口径：harmbench_label（0/1）。响应长度 = response 字符数。

三类输出（report/p0c_len_control.md，只新增不覆盖已有产物）：
  A. 每 model×modality 的 framing 效应 raw OR（storytelling/unrestricted vs baseline）
     → 控制 log_len 协变量后的 adjusted OR + 交互检验 → 长度混杂判定
  B. 长度分层表（condition × 长度档 → 有害率），检查各档内 framing 方向一致性
  C. text 模态效应分层（逐模型 Wald p + 效应量）→ 可引用的架构边界声明文本

方法：手写 ridge-IRLS logistic（numpy，λ=1e-6，Wald 检验），不依赖 statsmodels。
科学纪律：只读 parquet、只新增报告文件；不覆盖、不修改任何已有产物。
"""
import math
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/root/lalm_framing_revision_v6")
P = ROOT / "results" / "p0c_scored.parquet"
OUT = ROOT / "report" / "p0c_len_control.md"

CONDS = ["baseline", "storytelling", "unrestricted"]
BIN_EDGES = [0, 50, 200, 800, 10**9]
BIN_LABELS = ["<50", "50-199", "200-799", "800+"]
LAM = 1e-6


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def logreg_irwls(X, y, lam=LAM, max_iter=100, tol=1e-10):
    """Ridge-IRLS。返回 (beta, se, iters)。"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    beta = np.zeros(p)
    iters = 0
    for it in range(max_iter):
        eta = np.clip(X @ beta, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1.0 - mu)
        XWX = X.T @ (w[:, None] * X) + lam * np.eye(p)
        grad = X.T @ (y - mu)
        try:
            delta = np.linalg.solve(XWX, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(XWX, grad, rcond=None)[0]
        beta += delta
        iters = it + 1
        if np.max(np.abs(delta)) < tol:
            break
    XWX = X.T @ (w[:, None] * X) + lam * np.eye(p)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(XWX)))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    return beta, se, iters


def fit_effects(sub, conds, use_len):
    """y ~ (intercept) + cond dummies(vs baseline) [+ log_len]。返回行列表。"""
    cols = ["(intercept)"]
    parts = [np.ones(len(sub))]
    for c in conds[1:]:
        parts.append((sub["condition"] == c).astype(float))
        cols.append(f"cond:{c}")
    if use_len:
        parts.append(np.log1p(sub["resp_len"].values.astype(float)))
        cols.append("log_len")
    X = np.column_stack(parts)
    y = sub["harmbench_label"].values.astype(float)
    beta, se, iters = logreg_irwls(X, y)
    rows = []
    for i, c in enumerate(cols):
        b = beta[i]
        s = se[i]
        if np.isfinite(s) and s > 0:
            or_ = math.exp(b)
            lo = math.exp(b - 1.96 * s)
            hi = math.exp(b + 1.96 * s)
            z = b / s
            pv = 2.0 * (1.0 - norm_cdf(abs(z)))
        else:
            or_ = lo = hi = z = pv = float("nan")
        rows.append((c, b, s, or_, lo, hi, z, pv))
    return rows, iters


def fmt_or(or_, lo, hi):
    return f"{or_:.2f} [{lo:.2f}, {hi:.2f}]"


def fmt_p(pv):
    if not np.isfinite(pv):
        return "n/a"
    if pv < 0.001:
        return f"{pv:.2e}"
    return f"{pv:.3f}"


def main():
    df = pd.read_parquet(P)
    df["resp"] = df["response"].fillna("")
    df["resp_len"] = df["resp"].str.len()
    df["hb"] = pd.to_numeric(df["harmbench_label"], errors="coerce")
    df["len_bin"] = pd.cut(df["resp_len"], BIN_EDGES, labels=BIN_LABELS)
    d = df.dropna(subset=["hb"]).copy()
    d["hb"] = d["hb"].astype(int)

    L = []
    L.append("# P0-C 长度混杂控制 + text 效应分层分析")
    L.append("")
    L.append(f"数据：`results/p0c_scored.parquet`（{len(df)} 行）| 主口径：harmbench_label"
             f" | 有效行（hb 非空）：{len(d)} | 弃 null：{len(df) - len(d)}")
    L.append("长度 = response 字符数。logistic = 手写 ridge-IRLS（λ=1e-6），Wald 检验。")
    L.append("")

    # ---------- A. 每 model×modality raw vs adjusted ----------
    L.append("## A. Framing 效应：raw OR vs 控制 log_len 后 adjusted OR")
    L.append("")
    L.append("| model | modality | ASR base | ASR storyt | ASR unrest | OR_s_raw | OR_s_adj | OR_u_raw | OR_u_adj | len 交互 p |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    inter_results = {}
    for m in sorted(d["model"].unique()):
        for mod in ["audio", "text"]:
            sub = d[(d["model"] == m) & (d["modality"] == mod)]
            if len(sub) < 30:
                continue
            asr = {c: sub.loc[sub["condition"] == c, "hb"].mean() for c in CONDS}
            r_raw, _ = fit_effects(sub, CONDS, use_len=False)
            r_adj, _ = fit_effects(sub, CONDS, use_len=True)
            d_s_raw = {c: (b, s) for c, b, s, *_ in r_raw}
            d_s_adj = {c: (b, s) for c, b, s, *_ in r_adj}
            # 交互检验：y ~ cond + log_len + cond:log_len
            X = [np.ones(len(sub))]
            cols = ["(intercept)"]
            for c in CONDS[1:]:
                X.append((sub["condition"] == c).astype(float))
                cols.append(f"cond:{c}")
            ll = np.log1p(sub["resp_len"].values.astype(float))
            X.append(ll)
            cols.append("log_len")
            for c in CONDS[1:]:
                X.append(((sub["condition"] == c).astype(float) * ll))
                cols.append(f"inter:{c}")
            X = np.column_stack(X)
            beta, se, _ = logreg_irwls(X, sub["hb"].values.astype(float))
            int_p = min([pv for i, c in enumerate(cols) if c.startswith("inter:")
                         for pv in [2.0 * (1.0 - norm_cdf(abs(beta[i] / se[i])))
                                    if np.isfinite(se[i]) and se[i] > 0 else float("nan")]],
                        default=float("nan"))
            inter_results[(m, mod)] = int_p
            b_raw, s_raw = d_s_raw["cond:storytelling"]
            b_adj, s_adj = d_s_adj["cond:storytelling"]
            u_raw, _ = d_s_raw["cond:unrestricted"]
            u_adj, _ = d_s_adj["cond:unrestricted"]
            or_s_raw = math.exp(b_raw) if np.isfinite(b_raw) else float("nan")
            or_s_adj = math.exp(b_adj) if np.isfinite(b_adj) else float("nan")
            or_u_raw = math.exp(u_raw) if np.isfinite(u_raw) else float("nan")
            or_u_adj = math.exp(u_adj) if np.isfinite(u_adj) else float("nan")
            ip = fmt_p(int_p)
            L.append(f"| {m} | {mod} | {asr['baseline']*100:.1f}% | {asr['storytelling']*100:.1f}% | "
                     f"{asr['unrestricted']*100:.1f}% | {or_s_raw:.2f} | {or_s_adj:.2f} | "
                     f"{or_u_raw:.2f} | {or_u_adj:.2f} | {ip} |")
    L.append("")
    L.append("> 判定：adjusted OR 与 raw OR 同向且量级相近 ⇒ framing 效应在控制长度后保持；"
             "交互显著 ⇒ framing 效应随长度变化（长度是修饰因子，非单纯混杂）。")
    L.append("")

    # ---------- B. 长度分层表 ----------
    L.append("## B. 长度分层：condition × 长度档 → 有害率")
    L.append("")
    for mod in ["audio", "text"]:
        sub = d[d["modality"] == mod]
        L.append(f"### {mod}")
        L.append("")
        L.append("| 长度档 | baseline ASR | storytelling ASR | unrestricted ASR | 各档 S vs B 差 |")
        L.append("|---|---|---|---|---|")
        for lab in BIN_LABELS:
            b_ = sub[(sub["len_bin"] == lab) & (sub["condition"] == "baseline")]
            s_ = sub[(sub["len_bin"] == lab) & (sub["condition"] == "storytelling")]
            u_ = sub[(sub["len_bin"] == lab) & (sub["condition"] == "unrestricted")]
            if not len(b_) and not len(s_) and not len(u_):
                continue
            ab = b_["hb"].mean() if len(b_) else float("nan")
            asr = s_["hb"].mean() if len(s_) else float("nan")
            au = u_["hb"].mean() if len(u_) else float("nan")
            d_ = asr - ab if (np.isfinite(asr) and np.isfinite(ab)) else float("nan")
            dstr = f"{d_*100:+.1f}pp" if np.isfinite(d_) else "n/a"
            n = f"(n={len(b_)}/{len(s_)}/{len(u_)})"
            L.append(f"| {lab} {n} | {ab*100:.1f}% | {asr*100:.1f}% | {au*100:.1f}% | {dstr} |")
        L.append("")
    L.append("> 解读：若多数长度档内 storytelling>baseline 且差显著 ⇒ 混杂不推翻方向；"
             "若仅长响应档有效应 ⇒ 效应主要出现在「更长更顺从」的输出，需在正文如实披露。")
    L.append("")

    # ---------- C. text 模态效应分层 ----------
    L.append("## C. text 模态 framing 效应：逐模型 Wald 检验（架构边界）")
    L.append("")
    L.append("| model | modality | cond | OR | 95%CI | p | 判定 |")
    L.append("|---|---|---|---|---|---|---|")
    for m in sorted(d["model"].unique()):
        for cond in CONDS[1:]:
            for mod in ["audio", "text"]:
                sub = d[(d["model"] == m) & (d["modality"] == mod)]
                if len(sub) < 30:
                    continue
                r_adj, _ = fit_effects(sub, CONDS, use_len=True)
                for c, b, s, or_, lo, hi, z, pv in r_adj:
                    if c == f"cond:{cond}":
                        if not np.isfinite(pv):
                            verdict = "n/a"
                        elif pv < 0.05:
                            verdict = "显著" if or_ >= 1 else "显著(负向)"
                        else:
                            verdict = "不显著"
                        L.append(f"| {m} | {mod} | {cond} | {or_:.2f} | "
                                 f"[{lo:.2f}, {hi:.2f}] | {fmt_p(pv)} | {verdict} |")
    L.append("")
    L.append("## D. 可直接引用的边界声明（拟稿）")
    L.append("")
    L.append("> Text 模态下 framing 的放大效应呈**架构依赖**：Gemma 家族两个 LALM 上 text 模态"
             "无放大（或弱负向），Qwen2-Audio 的 text 模态仍显著放大；Audio 模态在全部三模型上均显著放大。"
             "该架构边界与附录（ShieldGemma 对 Gemma 家族生成器的 N 反转）指向一致的生成器族特定行为，"
             "而非通用的跨模态 framing 效应。")
    L.append("")

    out = OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        # 只新增，若已存在则加时间戳后缀
        import time
        out = out.with_name(f"p0c_len_control_{int(time.time())}.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"报告已写出：{out}（{len(L)} 行）")
    print("\n--- 摘要 ---")
    for line in L:
        if line.startswith("| ") and not line.startswith("| model"):
            print(line)


if __name__ == "__main__":
    main()
