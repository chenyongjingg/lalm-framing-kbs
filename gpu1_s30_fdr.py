#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU1 S30：FDR 多重比较校正（CPU，2026-08-14）。

动机：S19-S28 已报告大量显著性检验（N/E_t/R 主效应、模板分层、攻击族分层、
交互、模态效应）。审稿人必问多重比较问题：报告的所有 p<0.05 中，有多少在
Benjamini-Hochberg 校正下仍显著？本实验收集各结果 JSON 中所有显式 p 值，
做 BH-FDR 校正（q=0.05 / 0.10），如实披露：
  - 收集到多少检验、来源分布；
  - 原始显著（p<0.05）数、BH 校正后仍显著数；
  - 哪些结果在校正后失去显著性（诚实披露，不掩盖）。

纪律：
  - 纯 CPU、零生成；只读 results/gpu1_pipeline/*.json；只写 s30_* 产物。
  - 只处理"显式 p 值"（key 匹配 p 模式），不臆造 p。仅含 CI 的结果不纳入
    FDR（无 p 无法校正），在覆盖率中如实披露。
  - 多个 p 值天然相关（同一实验的多维度），BH 处理独立检验；相关结构作披露。

用法：python gpu1_s30_fdr.py [--q 0.05,0.10]
"""
import argparse
import json
import sys
import re
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _log(m):
    print("[s30] %s" % m, flush=True)


def _is_p_key(k):
    kl = str(k).strip().lower().replace("-", "_").replace(" ", "_")
    if kl in ("p", "p_value", "pval", "pvalue", "p_two_sided",
              "p_one_sided", "fisher_p", "p_exact", "boot_p"):
        return True
    if kl.endswith("_p") or kl.startswith("p_"):
        return True
    if "_p_" in kl:
        return True
    return False


def _walk(node, path, out, src):
    if isinstance(node, dict):
        for k, v in node.items():
            if _is_p_key(k) and isinstance(v, (int, float)) and \
                    0.0 <= v <= 1.0:
                out.append({"src": src, "path": ".".join(path + [str(k)]),
                            "p": float(v)})
            else:
                _walk(v, path + [str(k)], out, src)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk(v, path + ["[%d]" % i], out, src)


def _bh(pvals, q):
    """Benjamini-Hochberg：返回 (校正阈值数组, 存活掩码)。"""
    import numpy as np
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = np.array([q * i / m for i in range(1, m + 1)])
    # 存活：存在 j>=i 使 ranked[j] <= thresh[j]
    # 用累计 max 技巧：先找最大 k 满足 ranked[k] <= thresh[k]（向前累积 min）
    valid = ranked <= thresh
    keep = np.zeros(m, dtype=bool)
    if valid.any():
        k = int(np.flatnonzero(valid)[-1])  # 最大满足下标（排序后）
        keep[: k + 1] = True
    # 放回原顺序
    mask = np.empty_like(keep)
    mask[order] = keep
    # 校正后 q 值（单调化）
    qvals = np.full(m, np.nan)
    qvals[order] = np.minimum.accumulate(  # 反向累积 min 单调化
        (p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return qvals.tolist(), mask.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--qs", default="0.05,0.10")
    args = ap.parse_args()
    qs = [float(x) for x in args.qs.split(",") if x.strip()]

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from gpu1_common import resolve_root
    root = resolve_root(cfg)
    out_dir = root / "results" / "gpu1_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    jsons = sorted(
        [p for p in out_dir.glob("s*.json")
         if not any(p.name.startswith(x)
                    for x in ("s28", "s29", "s30", "s31", "s32"))])
    found = []
    covered = []
    for p in jsons:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            _log("跳过 %s: %s" % (p.name, e))
            continue
        before = len(found)
        _walk(data, [], found, p.name)
        covered.append({"file": p.name,
                        "n_p_values": len(found) - before})
        _log("%s: 提取 p 值 %d 个" % (p.name, len(found) - before))

    _log("共收集 %d 个显式 p 值（来源 %d 文件）" % (
        len(found), sum(1 for c in covered if c["n_p_values"] > 0)))
    if not found:
        _log("未找到任何显式 p 值——仅 CI 的结果无法做 FDR，输出覆盖率并退出")
        result = {"stage": "S30", "note": "无显式 p 值可校正",
                  "coverage": covered}
        (out_dir / "s30_fdr.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    ps = [f["p"] for f in found]
    n_raw_sig = sum(1 for p in ps if p < 0.05)
    per_q = {}
    for q in qs:
        qvals, mask = _bh(ps, q)
        n_surv = int(sum(mask))
        for i, f in enumerate(found):
            f["q_%s" % q] = round(qvals[i], 4)
            f["survive_q%s" % q] = bool(mask[i])
        per_q[q] = {"q": q, "n_survive": n_surv,
                    "raw_n_sig_0_05": n_raw_sig}
        _log("q=%.2f: 原始显著(0.05)=%d, BH 校正后存活=%d" % (
            q, n_raw_sig, n_surv))

    # 哪些在 0.05 原始显著但 q=0.05 存活失败（诚实披露）
    dropped = [f for f in found if f["p"] < 0.05 and not f.get(
        "survive_q0.05", False)]
    _log("原始显著但 q=0.05 校正后丢失: %d 个" % len(dropped))

    result = {
        "stage": "S30", "date": "2026-08-14",
        "method": ("收集 S19-S28 结果 JSON 中所有显式 p 值，"
                   "Benjamini-Hochberg FDR 校正"),
        "n_p_values": len(found), "n_files_with_p": sum(
            1 for c in covered if c["n_p_values"] > 0),
        "n_raw_sig_0_05": n_raw_sig,
        "per_q": {str(q): per_q[q] for q in qs},
        "coverage": covered,
        "dropped_by_fdr_q0_05": [
            {"src": f["src"], "path": f["path"], "p": f["p"]}
            for f in dropped],
        "all_tests": found,
        "disclosure": ("BH 假定独立检验；本结果存在天然相关（同一实验多维度），"
                       "相关结构未调整，属保守-激进边界问题，如实披露。仅含 CI "
                       "无显式 p 值的结果未纳入校正。"),
    }
    (out_dir / "s30_fdr.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# S30：FDR 多重比较校正（CPU）\n",
        "- 收集 **%d** 个显式 p 值（来源 %d 个结果 JSON）" % (
            len(found), sum(1 for c in covered if c["n_p_values"] > 0)),
        "- 原始显著（p<0.05）：**%d**" % n_raw_sig,
    ]
    for q in qs:
        lines.append("- **q=%.2f** BH 校正后存活：**%d**" % (
            q, per_q[q]["n_survive"]))
    lines.append("\n## 校正后丢失的显著结果（诚实披露）")
    if dropped:
        lines.append("\n| 来源 | 路径 | p |")
        lines.append("|---|---|---|")
        for f in dropped:
            lines.append("| %s | `%s` | %.4f |" % (f["src"], f["path"], f["p"]))
    else:
        lines.append("\n无（原始显著全部通过 q=0.05）")
    lines.append("\n## 覆盖率")
    lines.append("\n| 文件 | 提取 p 值 |")
    lines.append("|---|---|")
    for c in covered:
        lines.append("| %s | %d |" % (c["file"], c["n_p_values"]))
    lines.append("\n## 披露")
    lines.append("> %s" % result["disclosure"])
    (root / "report" / "s30_fdr.md").write_text(
        "\n".join(lines), encoding="utf-8")
    _log("已落盘 s30_fdr.json + report/s30_fdr.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
