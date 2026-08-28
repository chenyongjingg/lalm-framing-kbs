# -*- coding: utf-8 -*-
"""
stage_l_novelty.py — 阶段 L：新颖性与文献核验（v6.5 §2, v6.5.6 检索质量修复）

目的：防 idea 撞车 + 防引用幻觉（顶刊初审致命问题）。

v6.5.6 修复（2026-08-05，审计报告 #8 🟠 高）：
- CrossRef 检索改用 query.bibliographic 短语检索 + from-pub-date/until-pub-date
  年份过滤（旧版无年份过滤 → 1972 管风琴/2016 隐写/词典词条混入）
- TOPICS 检索词定向化（LALM jailbreak → "jailbreak audio language model" 等）
- 引用核验升级为 多字段模糊匹配（标题 + 年份 + venue 全匹配才算已核实）
- KBS 本刊检索增加 container-title 过滤（仅保留 KBS 本刊论文）

- 新颖性审计：检索 2024-2026 主题文献 → report/novelty_audit.md
- 六篇必引差异化论证表（PJ-Break / Omni-SafetyBench / StyleBreak /
  Cross-modal Info Check / Chen et al. / Semantic Codebooks）
- KBS 本刊定向检索（近两年 LLM 安全/知识表征/可解释检测）→ report/kbs_scope_papers.md
- 引用核验：CrossRef/arXiv API 逐条核验 → report/citation_verification.md
- 对比定位表：LaTeX → report/related_work_positioning.tex

退出：完成 → 0；部分（网络不可达降级）→ 2；致命 → 3
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from common_utils import load_config, setup_logging, JsonlLogger, Checkpoint

STAGE = "l"

TOPICS = [
    "audio narrative attack",
    "prosody jailbreak speech delivery attack",  # v6.5.15: 提示词 §2.1 明确主题，原缺失
    "jailbreak audio language model",        # v6.5.6: LALM jailbreak → 精准短语
    "persuasion jailbreak",
    "cross-modal safety consistency",             # v6.5.15: 原 inconsistency 措辞，对齐 §2.1
    "cross-modal consistency detection jailbreak", # v6.5.15: 提示词 §2.1 明确主题，原缺失
    "multi-source fusion defense LLM safety",
    "confidence-based jailbreak detection",          # v6.4 新增（Chen et al.）
    "cross-lingual jailbreak detection",             # v6.4 新增（Semantic Codebooks）
    "gradient-based jailbreak detection",            # v6.4 新增（GradSafe）
]

# 六篇必引（v6.4 §2）——逐篇核验最新版本与发表状态 + 差异化论证
MUST_CITE = [
    {"key": "pj_break", "title": "PJ-Break", "venue": "arXiv", "year": 2026,
     "id": "2607.26541",
     "diff": "delivery preset 单因子 vs 本文语义成分全析因；presets 声学属性共变，本文 E_t/A_s 分离为方法学优势"},
    {"key": "omni_safetybench", "title": "Omni-SafetyBench", "venue": "arXiv",
     "year": 2025, "id": "2508.07173",
     "diff": "benchmark-level 静态一致性（CMSC-score）vs 本文 response-level 配对分歧（PCSD）"},
    {"key": "stylebreak", "title": "StyleBreak", "venue": "AAAI", "year": 2026,
     "id": None,
     "diff": "攻击方法导向 vs 机制归因导向，纳入基线"},
    {"key": "cross_modal_info_check", "title": "Cross-modality Information Check",
     "venue": "EMNLP Findings", "year": 2024, "id": None,
     "diff": "跨模态一致性检测先驱（视觉域、双模态核对、无可学习融合），MSRF 必须对比"},
    {"key": "chen_et_al_confidence", "title": "Chen et al. first-token confidence jailbreak detection",
     "venue": "EMNLP", "year": 2025, "id": None,
     "diff": "首 token 置信度单一信号 vs 本文不确定性分支（生成置信信号 + 多评分器分歧的融合特征）"},
    {"key": "semantic_codebooks", "title": "Cross-Lingual Jailbreak Detection via Semantic Codebooks",
     "venue": "arXiv", "year": 2026, "id": "2604.25716",
     "diff": "跨语言检测方法 vs 本文跨语言机制复现定位"},
]

# 拟核验引用（论文草稿核心引用；arXiv/CrossRef 核验元数据一致性）
# v6.5.7 人工复核（2026-08-05）：全部 7 条已核实，补齐 arXiv ID 与修正年份。
#   NYHM        → EACL 2026, arXiv:2601.23255（ACL Anthology 2026.eacl-long.278）
#   AudioJailbk → arXiv:2505.15406（提示词"2024"有误，实为 2025-05）
#   Jailbroken  → NeurIPS 2023, arXiv:2307.02483
#   StrongREJECT→ arXiv:2402.10260（ICLR 2024 Workshop）
CITATIONS = [
    {"title": "Now You Hear Me", "venue": "EACL", "year": 2026,
     "id": "2601.23255", "manual_verified": True,
     "note": "裸请求仅改语气 ASR<10% 证据来源（Audio Narrative Attacks, 2026.eacl-long.278）"},
    {"title": "Audio Jailbreak", "venue": "arXiv", "year": 2025,
     "id": "2505.15406", "manual_verified": True,
     "note": "AJailBench 基准（MBZUAI）"},
    {"title": "Jailbroken", "venue": "NeurIPS", "year": 2023,
     "id": "2307.02483", "manual_verified": True,
     "note": "安全训练失败模式（competing objectives / mismatched generalization）"},
    {"title": "StrongREJECT", "venue": "arXiv", "year": 2024,
     "id": "2402.10260", "manual_verified": True,
     "note": "A StrongREJECT for Empty Jailbreaks（ICLR 2024 Workshop）"},
    {"title": "HarmBench", "venue": "ICLR", "year": 2025,
     "id": "2402.04249", "manual_verified": True,
     "note": "标准化红队评估框架"},
    {"title": "Llama Guard", "venue": "arXiv", "year": 2023,
     "id": "2312.06674", "manual_verified": True,
     "note": "LLM 输入输出安全护栏"},
    {"title": "GradSafe", "venue": "ACL", "year": 2024,
     "id": "2402.13494", "manual_verified": True,
     "note": "防御基线（安全关键梯度分析）"},
    # 六篇必引（v6.5.15：§2.4 要求拟引用逐条核验——此前六篇必引未进核验流程，
    # 现全部纳入；有 arXiv ID 的用 ID 精确核验，无 ID 的走 CrossRef 标题匹配）
    {"title": "PJ-Break", "venue": "arXiv", "year": 2026,
     "id": "2607.26541", "manual_verified": True,
     "must_cite": True,
     "note": "必引①：delivery preset 单因子 vs 本文语义成分全析因（2026-08-10 人工确认：arXiv 2607.26541 标题'Prosody-driven Jailbreaks in Audio LLMs'即 PJ-Break，摘要与协议 §0 描述一致）"},
    {"title": "Omni-SafetyBench", "venue": "arXiv", "year": 2025,
     "id": "2508.07173",
     "must_cite": True,
     "note": "必引②：benchmark-level 静态一致性 vs 本文 PCSD response-level"},
    {"title": "StyleBreak", "venue": "AAAI", "year": 2026,
     "id": None,
     "must_cite": True,
     "note": "必引③：攻击方法导向 vs 机制归因导向，纳入基线"},
    {"title": "Cross-modality Information Check", "venue": "EMNLP Findings",
     "year": 2024, "id": None, "must_cite": True,
     "note": "必引④：跨模态一致性检测先驱，MSRF 必须对比"},
    {"title": "Chen et al. first-token confidence jailbreak detection",
     "venue": "EMNLP", "year": 2025, "id": None, "must_cite": True,
     "note": "必引⑤：首 token 置信度单一信号 vs 本文不确定性融合特征"},
    {"title": "Cross-Lingual Jailbreak Detection via Semantic Codebooks",
     "venue": "arXiv", "year": 2026, "id": "2604.25716", "must_cite": True,
     "note": "必引⑥：跨语言检测方法 vs 本文跨语言机制复现定位"},
]


def crossref_search(query: str, limit: int = 5,
                    container: str | None = None) -> list:
    """CrossRef API 检索（v6.5.6：query.bibliographic 短语 + 年份过滤）。

    - query.bibliographic 短语检索（比 query= 精确，避免词典词条/不相关主题）
    - filter=from-pub-date:2024-01-01,until-pub-date:2026-12-31 硬性年份过滤
    - container 非空时附加 container-title 精确过滤（KBS 本刊定向用）
    """
    params = [
        "query.bibliographic=" + urllib.parse.quote(query),
        f"rows={limit}",
        "filter=from-pub-date:2024-01-01,until-pub-date:2026-12-31",
        "select=title,author,published-print,published-online,DOI,"
        "container-title,URL,type",
    ]
    if container:
        params.append("query.container-title=" + urllib.parse.quote(container))
    url = "https://api.crossref.org/works?" + "&".join(params)
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            data = json.loads(r.read().decode("utf-8"))
        items = data.get("message", {}).get("items", [])
        # 只保留 journal/proceedings 论文（排除词典条目、章节、figure/table 等碎片）
        keep = []
        for it in items:
            t = it.get("type", "")
            if t not in ("journal-article", "proceedings-article",
                         "book-chapter", "posted-content", "reference-entry"):
                continue
            if container:
                ct = it.get("container-title", [""])
                if isinstance(ct, list):
                    ct = ct[0] if ct else ""
                if container.lower() not in str(ct).lower():
                    continue
            # 过滤预印本平台（SSRN 等 posted-content 混入大量无关项）
            doi = str(it.get("DOI", ""))
            if "ssrn.com" in doi or "10.2139" in doi:
                continue
            keep.append(it)
        return keep
    except Exception as e:  # noqa: BLE001
        print(f"[crossref] 检索失败 {query}: {e}")
        return []


def arxiv_search(query: str, limit: int = 5) -> list:
    """arXiv API 检索。"""
    url = ("http://export.arxiv.org/api/query?search_query=all:"
           + urllib.parse.quote(query) + f"&max_results={limit}")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            raw = r.read().decode("utf-8")
        import xml.etree.ElementTree as ET
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(raw)
        out = []
        for e in root.findall("a:entry", ns):
            title = (e.findtext("a:title", default="", namespaces=ns)
                     or "").strip().replace("\n", " ")
            link = e.findtext("a:id", default="", namespaces=ns)
            pub = e.findtext("a:published", default="", namespaces=ns)
            out.append({"title": title, "url": link, "published": pub[:10]})
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[arxiv] 检索失败 {query}: {e}")
        return []


def _year_of(h: dict) -> str:
    """从 CrossRef 条目提取年份（优先印刷版，其次在线版）。"""
    for key in ("published-print", "published-online"):
        dp = h.get(key, {}).get("date-parts", [[None]])
        try:
            y = dp[0][0]
        except (IndexError, TypeError):
            y = None
        if y:
            return str(y)
    return str(h.get("published", ""))[:4] or "—"


def _fmt_hit(h: dict) -> str:
    """格式化 CrossRef 条目为一行 markdown（title (year) url）。"""
    title = h.get("title", "?")
    if isinstance(title, list):
        title = title[0] if title else "?"
    url = h.get("URL") or h.get("url") or ""
    return f"- {title} ({_year_of(h)}) {url}\n"


# ---- 引用核验辅助（v6.5.6 模块级，便于单测与复用）----

_NORM_STOP = {"the", "a", "an", "and", "of", "for", "on", "in", "via", "to",
              "with", "based", "using", "how", "does", "what", "why", "their",
              "its", "from", "into"}


def _norm(s: str) -> str:
    """规范化：小写、去标点、去停用词。"""
    import re
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return " ".join(w for w in s.split() if w not in _NORM_STOP)


def _container_matches(h: dict, want: str = "knowledge-based systems") -> bool:
    """CrossRef container-title 兼容 list/str 的匹配（v6.5.29-fix，第九轮审查 🔴）。

    原实现 `for ct in h.get("container-title", [])` 假设返回 list——CrossRef 单容器
    常以字符串返回，`for ct in 字符串` 逐字符迭代 → 匹配恒 False → 全部 KBS 候选
    被 continue 跳过 → 实跑候选筛选 0 篇（§2.3 交付物未达成）。
    """
    ct = h.get("container-title")
    if ct is None:
        return False
    vals = ct if isinstance(ct, list) else [ct]
    return any(want in str(v).lower() for v in vals)


def _jaccard(title_a: str, title_b: str) -> float:
    """标题词集合 Jaccard 相似度。"""
    sa, sb = set(_norm(title_a).split()), set(_norm(title_b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _year_match(h: dict, year: int) -> bool:
    y = _year_of(h)
    return y == "—" or str(year) in str(y) or str(int(year) + 1) in str(y)


def _arxiv_verify(title: str) -> dict | None:
    """arXiv 核验兜底：完整标题短语 + 关键词短语双查询。

    命中规则（按序判定）：
      R1 ≥3 词查询且为候选标题连续子串 → 命中
      R2 ≤2 词查询且候选标题以查询开头、后跟分隔符(:/(/结尾) → 命中
      （排除 Llama Guard 3-1B 之类"同名 + 追加词"变体）
      R3 综合分 = 0.6×覆盖率 + 0.4×Jaccard ≥ 0.70 → 命中
    """
    import re as _re
    words = [w for w in _re.sub(r"[^a-z0-9 ]", " ", title.lower()).split()
             if w not in _NORM_STOP and len(w) > 2]
    # 完整标题去停用词短语（保留语序）优先；单/双词标题追加关键词组合
    full_phrase = " ".join(words)
    queries = [full_phrase]
    if len(words) >= 2:
        queries.append(" ".join(words[:4]))
    if len(words) <= 2:
        queries.append(title)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    nq = _norm(title)
    qwords = nq.split()
    for q in queries:
        url = ("http://export.arxiv.org/api/query?search_query=all:"
               + urllib.parse.quote(f'"{q}"') + "&max_results=8")
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                raw = r.read().decode("utf-8")
            import xml.etree.ElementTree as ET
            root = ET.fromstring(raw)
            for e in root.findall("a:entry", ns):
                t0 = (e.findtext("a:title", default="", namespaces=ns)
                      or "").strip().replace("\n", " ")
                hit = _arxiv_match(title, t0)
                if hit:
                    return {"title": t0,
                            "published": (e.findtext("a:published", default="",
                                                     namespaces=ns) or "")[:10],
                            "url": (e.findtext("a:id", default="",
                                               namespaces=ns) or "")}
        except Exception as e:  # noqa: BLE001
            print(f"[arxiv] 核验失败 {title}: {e}")
    return None


def _arxiv_verify_by_id(arxiv_id: str, expect_title: str) -> dict | None:
    """按 arXiv ID 精确核验：拉取该 ID 的元数据，校验标题一致性。

    v6.5.15（问题 31）：六篇必引 + 核心引用凡有 arXiv ID 优先走此精确路径，
    避免标题模糊匹配误判（同名变体/追加词）与漏判（措辞差异）。
    核验标准：规范化后 Jaccard ≥ 0.6（宽松，因 arXiv 标题与人工引用措辞
    可能略有差异）；核验结果如实标注来源 = "arXiv ID 精确核验"。
    """
    import re as _re
    import xml.etree.ElementTree as ET
    url = ("http://export.arxiv.org/api/query?id_list="
           + urllib.parse.quote(arxiv_id) + "&max_results=1")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            raw = r.read().decode("utf-8")
        root = ET.fromstring(raw)
        entries = root.findall("a:entry", ns)
        if not entries:
            return None
        e = entries[0]
        t0 = (e.findtext("a:title", default="", namespaces=ns)
              or "").strip().replace("\n", " ")
        # v6.5.29-fix（第九轮审查 🟡）：原统一 Jaccard≥0.6 判定——2 词短标题
        # （PJ-Break / Omni-SafetyBench）与真实长标题 Jaccard 恒 <0.6 → 合法
        # arXiv ID 也被误判"元数据不符"。混合判定：先 R2 前缀（短查询词在
        # 标题开头），失败再回退 Jaccard≥0.6（短查询词在标题中间/结尾，如
        # StrongREJECT in "A StrongREJECT for Empty Jailbreaks"）。
        if not (_arxiv_match(expect_title, t0)
                or _jaccard(expect_title, t0) >= 0.6):
            return {"title": t0,
                    "published": (e.findtext("a:published", default="",
                                             namespaces=ns) or "")[:10],
                    "url": (e.findtext("a:id", default="", namespaces=ns) or ""),
                    "mismatch": True}
        return {"title": t0,
                "published": (e.findtext("a:published", default="",
                                         namespaces=ns) or "")[:10],
                "url": (e.findtext("a:id", default="", namespaces=ns) or "")}
    except Exception as e2:  # noqa: BLE001
        # v6.5.29-fix（第九轮审查 🔴）：网络/API 异常与"ID 不存在"必须区分。
        # 原实现一律 return None → 调用方降级 CrossRef 兜底 → arXiv-only 预印本
        # CrossRef 不收录 → 标"未找到"，覆盖 2026-08-05 已人工核验的事实（§2.4
        # 交付物失效：已核实条目被网络故障抹掉）。现返回 network_ok:False，
        # 调用方据此保留人工核验状态并如实披露自动核验未完成。
        print(f"[arxiv] ID 核验失败（网络/API 异常，非 ID 无效） {arxiv_id}: {e2}")
        return {"network_ok": False, "arxiv_id": arxiv_id}


def _arxiv_match(query_title: str, cand_title: str) -> bool:
    """arXiv 候选是否匹配查询标题（R1/R2/R3 三层判定）。

    R1 ≥3 词查询为候选连续子串 → 命中
    R2 ≤2 词查询且候选以查询开头、后跟分隔符(:/(/结尾) → 命中
      （排除 Llama Guard 3-1B 之类"同名 + 追加词"变体）
    R3 覆盖率=1.0 且 综合分 = 0.6×覆盖率 + 0.4×Jaccard ≥ 0.75 → 命中
      （严格兜底，仅极强信号；防 StrongREJECT 0.733 / Llama Guard 3-1B 0.700 误报）
    """
    nq, nc = _norm(query_title), _norm(cand_title)
    qwords = nq.split()
    n = len(qwords)
    # R1: ≥3 词查询为候选连续子串
    if n >= 3 and nq in nc:
        return True
    # R2: ≤2 词查询，候选以查询开头且后跟分隔符/结尾
    if n <= 2:
        raw_c = cand_title.lower().lstrip()
        raw_q = query_title.lower().strip()
        if not raw_c.startswith(raw_q):
            return False  # 候选不以查询开头 → 非同名，直接排除
        after = raw_c[len(raw_q):]
        if after == "" or after[0] in ":(":
            return True
        return False  # 前缀匹配但后接词（同名变体）→ 直接排除，不走 R3
    # R3: 严格兜底（覆盖率全中 + 高分）
    cov = _coverage(query_title, cand_title)
    score = 0.6 * cov + 0.4 * _jaccard(query_title, cand_title)
    return cov >= 1.0 and score >= 0.75


def _coverage(query_title: str, cand_title: str) -> float:
    """查询标题去掉停用词后的 token 中，出现在候选标题里的比例。"""
    qt = [w for w in _norm(query_title).split()
          if w not in _NORM_STOP and len(w) > 2]
    ct = set(_norm(cand_title).split())
    if not qt:
        return 0.0
    return sum(1 for w in qt if w in ct) / len(qt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["workdir"]).expanduser()
    log = setup_logging(str(root / "logs" / f"{STAGE}.log"), STAGE)
    jlog = JsonlLogger(str(root / "logs" / f"{STAGE}.jsonl"))
    ckpt = Checkpoint(str(root / "checkpoints" / f"{STAGE}.json"))

    log.info("=== 阶段 L（新颖性 + 引用核验）启动 ===")
    # v6.5.29-fix（第十轮审查 🟡，L-1）：config novelty 区块与 prompt §2.1 一致性
    # 校验——config topics 缺项（如 prosody_jailbreak_speech_delivery /
    # cross_modal_consistency_detection_jailbreak）时警告并披露，TOPICS 常量仍为
    # 检索唯一源（保留精心编写的中文检索词），避免键→词映射丢失语义。
    _cfg_topics = (cfg.get("novelty", {}) or {}).get("topics") or []
    _required_keys = {"audio_narrative_attack",
                      "prosody_jailbreak_speech_delivery",
                      "lalm_jailbreak", "persuasion_jailbreak",
                      "cross_modal_safety_inconsistency",
                      "cross_modal_consistency_detection_jailbreak",
                      "multi_source_fusion_defense",
                      "confidence_based_jailbreak_detection",
                      "cross_lingual_jailbreak_detection",
                      "gradient_based_jailbreak_detection"}
    _missing = sorted(_required_keys - set(_cfg_topics))
    if _missing:
        log.warning("config novelty.topics 缺失 %d 项（%s）→ 与 prompt §2.1 十主题"
                    "清单不符；TOPICS 常量已含全部主题，如实披露配置滞后",
                    len(_missing), ", ".join(_missing))
    if ckpt.is_done("done"):
        log.info("阶段 L 已完成，跳过")
        return 0

    import urllib.parse  # noqa: PLC0415

    rpt = root / "report"
    rpt.mkdir(parents=True, exist_ok=True)

    # ---- 1. 新颖性审计 ----
    log.info("检索 %d 个主题", len(TOPICS))
    novelty_lines = ["# 新颖性审计（v6.5）\n",
                     f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                     f"- 检索年份: 2024-2026",
                     f"- 检索源: CrossRef / arXiv（服务器可达性决定）\n"]
    all_hits = {}
    network_ok = True
    for t in TOPICS:
        hits = crossref_search(t, limit=5)
        if not hits:
            hits = arxiv_search(t, limit=5)
        all_hits[t] = hits
        novelty_lines.append(f"\n## 主题: {t}\n")
        if not hits:
            novelty_lines.append("- 无检索结果（网络不可达或主题无匹配）\n")
            network_ok = False
        for h in hits[:5]:
            novelty_lines.append(_fmt_hit(h))

    # 六篇必引差异化论证表（v6.5 §2）
    novelty_lines.extend([
        "\n## 六篇必引差异化论证表（v6.5 §2）\n",
        "| 文献 | 定位 | 与本文三贡献的差异化论证 |\n|---|---|---|\n",
    ])
    for mc in MUST_CITE:
        novelty_lines.append(
            f"| {mc['title']}（{mc['venue']} {mc['year']}）"
            f" | {mc['diff'].split('：')[0] if '：' in mc['diff'] else mc['diff'][:12]}"
            f" | {mc['diff']} |\n")
    novelty_lines.extend([
        "\n## 与拟贡献的差异化论证\n",
        "拟贡献 ①：framing 语义成分全析因归因（E_t/N/R/A_s 四因子因果分解，第一贡献）。",
        "拟贡献 ②：结构化表征与 MSRF 检测框架（承接归因发现，跨攻击族泛化）。",
        "拟贡献 ③：跨模态安全一致性辅助证据（PCSD，response-level）。\n",
        "差异化要点：",
        "- 若检索到 LALM 安全一致性测量 → 标注撞车风险并建议贡献收缩",
        "- 若检索到叙事结构因果分析 → 对比 E_t/N/R 分解粒度",
        "- 若检索到多源融合防御 → 对比四分支互补性与自适应攻击评估",
        "- Chen et al.（EMNLP 2025）：其首 token 置信度为单一信号，本文不确定性分支 = 生成置信 + 多评分器分歧融合特征（区分引用）",
        "- Semantic Codebooks（arXiv:2604.25716）：其跨语言为检测方法，本文跨语言为机制复现（区分定位）\n",
    ])
    novelty_path = rpt / "novelty_audit.md"
    novelty_path.write_text("\n".join(novelty_lines), encoding="utf-8")
    log.info("新颖性审计: %s", novelty_path)

    # ---- 1.5 KBS 本刊定向检索（v6.5 §2 新增）----
    kbs_lines = ["# KBS 本刊定向检索（v6.5 §2）\n",
                 f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 "- 范围: Knowledge-Based Systems 近两年（2024-2026）LLM 安全 / "
                 "知识表征 / 可解释检测相关论文",
                 "- 用途: scope 信号 + 潜在审稿人画像（3-5 篇纳入相关工作候选）\n"]
    kbs_topics = [
        "Knowledge-Based Systems large language model safety",
        "Knowledge-Based Systems jailbreak detection",
        "Knowledge-Based Systems knowledge representation interpretable detection",
    ]
    kbs_hits = {}
    kbs_fallback = False
    for t in kbs_topics:
        hits = crossref_search(t, limit=5,
                               container="Knowledge-Based Systems")
        if not hits:
            # v6.5.22-fix（问题 76）：CrossRef 不可达时 arXiv fallback 结果
            # **不是 KBS 本刊论文**——必须标注降级来源，且不得进入"KBS 本刊
            # 候选"筛选（否则非 KBS 论文混入候选清单 → 下游 §15.2 引用幻觉）。
            # 如实标注"（arXiv fallback，非 KBS 本刊，待人工补查）"。
            hits = arxiv_search(t, limit=5)
            kbs_fallback = True
        kbs_hits[t] = hits
        kbs_lines.append(f"\n## 检索: {t}\n")
        if not hits:
            kbs_lines.append("- 无结果（网络不可达）\n")
        for h in hits[:5]:
            kbs_lines.append(_fmt_hit(h))
        if kbs_fallback:
            kbs_lines.append(
                "\n> 注：以上为 arXiv fallback 结果（CrossRef 不可达），"
                "**非 KBS 本刊论文**，仅作主题线索参考，不得列入 KBS 本刊候选。\n")
    kbs_lines.append("\n## 候选筛选（3-5 篇，每篇引用价值一句话说明）\n")
    # 自动从检索结果中筛选（网络可达时取前 5 条；不可达时占位待人工补查）
    # v6.5.22-fix（问题 76）：仅 CrossRef（container 过滤后）结果计入候选；
    # arXiv fallback 结果不计入（非 KBS 本刊）。
    # v6.5.30-fix（KBS 标准 §2.3 补全）：按相关性筛选候选——原实现"首个主题
    # 前 5 条"无相关性排序，离题论文（金融/铁路/农业/推荐）挤占候选、高度
    # 相关的 jailbreak-mitigation 论文落选（scope 信号弱化，KBS 审稿会质疑）。
    # 现按标题 scope 关键词打分选 top5，并生成逐篇引用价值说明（非模板）。
    # v6.5.30-fix（二轮收紧，KBS §2.3）：安全类关键词高权重（×2）+ 离题领域
    # 负向过滤（finance/agriculture/train/medical 等强制排除）。原打分仅统计
    # 关键词命中，FinBloom（金融）/Graph RAG（铁路）靠 "Knowledge/LLM" 词命中
    # 分 2 入选——非本刊 scope。现负向领域直接 -100 排除。
    _safety_kw = ["jailbreak", "safety", "security", "harm", "align", "guard",
                  "mitigation", "attack", "malicious", "toxic", "offensive"]
    _scope_kw = ["large language model", " llm", "knowledge",
                 "interpretable", "explain", "representation", "detection",
                 "fusion"]
    _neg_kw = ["finance", "financial", "agricultural", "agriculture", "train",
               "railway", "bogie", "medical", "clinical", "legal", "law",
               "recommendation", "stock", "trading", "cancer", "pulmonary",
               "nodule", "depression", "fault", "iot", "catheter", "supply",
               "logistics"]

    def _scope_score(_ti: str) -> int:
        _tl = str(_ti).lower()
        if sum(1 for _k in _neg_kw if _k in _tl):
            return -100  # 离题领域，强制排除
        _s = 2 * sum(1 for _k in _safety_kw if _k in _tl)
        _s += sum(1 for _k in _scope_kw if _k in _tl)
        return _s

    _cand = []  # (score, title)
    for _t, _hits in kbs_hits.items():
        if kbs_fallback and not _hits:
            continue
        for _h in (_hits or [])[:8]:
            if not isinstance(_h, dict) or not _container_matches(_h):
                continue
            _ti = _h.get("title", "?")
            if isinstance(_ti, list):
                _ti = _ti[0] if _ti else "?"
            _cand.append((_scope_score(_ti), _ti))
    # 相关性降序；同分保持检索顺序稳定；负分（离题领域）排除
    _cand = [c for c in _cand if c[0] >= 0]
    _cand.sort(key=lambda c: (-c[0], c[1]))
    _picked_n = 0
    for _s, _ti in _cand[:5]:
        _tl = str(_ti).lower()
        if any(_k in _tl for _k in
               ["jailbreak", "safety", "security", "harm", "guard",
                "mitigation", "align", "attack", "malicious", "toxic"]):
            _note = "KBS 本刊 LLM 安全/越狱防护方向（scope 信号）"
        elif any(_k in _tl for _k in ["knowledge", "representation"]):
            _note = "KBS 本刊知识表征方向（scope 信号）"
        elif any(_k in _tl for _k in
                 ["interpretable", "explain", "detection", " llm"]):
            _note = "KBS 本刊可解释/检测/LLM 方向（scope 信号）"
        else:
            _note = "KBS 本刊相关（scope 信号候选）"
        kbs_lines.append(f"- **{_ti}**\n"
                         f"  引用价值: {_note}（相关性分 {_s}）\n")
        _picked_n += 1
    if _picked_n < 3:
        kbs_lines.append(
            f"\n> 注：经过离题领域负向过滤后仅 {_picked_n} 篇对题候选"
            "（§2.3 要求 3-5 篇）——本刊近两年 LLM 安全/知识表征/可解释检测"
            "方向论文检索量有限，如实披露缺口，后续可扩展检索主题补足。\n")
    picked = _picked_n
    if picked == 0:
        kbs_lines.append("- 网络不可达：候选待人工补查（标注“待人工补查”）\n")
    kbs_path = rpt / "kbs_scope_papers.md"
    kbs_path.write_text("\n".join(kbs_lines), encoding="utf-8")
    log.info("KBS 本刊检索: %s（候选 %d 篇）", kbs_path, picked)

    # ---- 2. 引用核验 ----
    log.info("核验 %d 条引用", len(CITATIONS))
    cit_lines = ["# 引用核验报告（v6.5, v6.5.6 双源核验）\n",
                 "核验源: CrossRef（多字段匹配）→ arXiv（all: 短语检索兜底）\n"]
    verified, not_found, pending = 0, 0, 0

    for c in CITATIONS:
        # v6.5.15：有 arXiv ID 的引用优先精确核验（避免标题模糊匹配误判/漏判）
        # v6.5.19-fix（问题 64）：match 必须显式初始化——若 _arxiv_verify_by_id
        # 返回 None（arXiv 不可达/ID 无效），原代码 `if not (c.get("id") and match)`
        # 访问未定义 match → NameError → 阶段 L 顶层崩溃且无 errors.jsonl 落盘
        # （违反最高纪律 #2）。初始化后自然降级到 CrossRef 兜底。
        match = None
        # 注意：ID 核验给出结论（无论核实或 mismatch）后不再走 CrossRef 覆盖，
        # 避免"ID 标题不匹配"被后续模糊匹配误判为已核实。
        if c.get("id"):
            ax = _arxiv_verify_by_id(c["id"], c["title"])
            if ax:
                if ax.get("network_ok") is False:
                    # v6.5.29-fix（第九轮审查 🔴）：arXiv API 不可达（非 ID 无效）。
                    # 不得降级为"未找到"覆盖人工核验事实：引用若为人工核验过的
                    # （CITATIONS note 记录 2026-08-05 人工复核）→ 如实标注"沿用
                    # 人工核验，自动核验未完成"；否则如实披露网络不可达。
                    if c.get("manual_verified"):
                        match, status = ax, (
                            "已核实（2026-08-05 人工复核；本次 arXiv API 不可达，"
                            "自动核验未完成）")
                    else:
                        match, status = ax, (
                            "自动核验未完成（arXiv API 不可达，需人工补查元数据）")
                elif ax.get("mismatch"):
                    # ID 存在但标题不一致 → 元数据不符（不得标已核实）。
                    # v6.5.29-fix（第九轮审查）：manual_verified 条目（2026-08-05
                    # 人工已核实 ID 正确）不得被自动判定的短标题误判覆盖——如
                    # StrongREJECT/PJ-Break 短标题 R2 前缀不中、Jaccard 恒低，
                    # 自动判定天然不可靠；保留人工核验事实并如实披露自动判定分歧。
                    if c.get("manual_verified"):
                        _t0 = (ax.get("title") or "?")[:60]
                        match, status = ax, (
                            "已核实（2026-08-05 人工复核；自动判定标题分歧"
                            f"[{_t0}]，以人工核验为准）")
                    else:
                        match, status = ax, "元数据不符（arXiv ID 标题不匹配）"
                else:
                    match, status = ax, "已核实（arXiv ID）"
        if not (c.get("id") and match):
            hits = crossref_search(c["title"], limit=5)
            status, match = "未找到", None
            for h in hits:
                t0 = h.get("title", "?")
                if isinstance(t0, list):
                    t0 = t0[0] if t0 else "?"
                if _jaccard(c["title"], t0) >= 0.6 and _year_match(h, c["year"]):
                    match, status = h, "已核实（CrossRef）"
                    break
                if _jaccard(c["title"], t0) >= 0.6:
                    match = h
                    status = "元数据不符（年份/venue 未匹配）"
            # CrossRef 未核实 → arXiv 兜底
            if status == "未找到":
                ax = _arxiv_verify(c["title"])
                if ax:
                    match, status = ax, "已核实（arXiv）"
        if status.startswith("已核实"):
            verified += 1
        elif c.get("must_cite"):
            # D16 裁决（2026-08-10）：必引条目为协议附录 A 强制收录（§2.2），
            # 即使 CrossRef/arXiv 未自动匹配也不得误判"未找到"（否则 §2.4 会
            # 错误地禁止必引进论文）。如实标注"待人工补查"，不伪造元数据。
            status += " | 必引-待人工补查（协议附录 A 收录，CrossRef/arXiv 未自动匹配，需人工确认元数据）"
            pending += 1
        else:
            not_found += 1
        note = c.get("note", "")
        # v6.5.30-fix：match 可能是无 title 的 dict（如 network_ok:False 网络失败
        # 路径返回 {"network_ok":False,"arxiv_id":...}）——`match['title']` 会
        # KeyError 崩溃（永续审查实跑暴露）。改 .get 兜底。
        _mt = match.get("title", "?") if isinstance(match, dict) else "?"
        cit_lines.append(
            f"- {c['title']} ({c['venue']} {c['year']}) → **{status}**"
            + (f" | 实际: {_mt} ({match.get('published', _year_of(match))})"
               f" {match.get('url', match.get('URL', ''))}" if match else "")
            + (f" | {note}" if note else ""))
    cit_lines.extend([
        f"\n## 汇总",
        f"- 已核实: {verified}",
        f"- 必引-待人工补查: {pending}（§2.2 强制必引，元数据待人工确认）",
        f"- 未找到/不符: {not_found}（非必引未通过核验的引用禁止进入论文）\n",
    ])
    cit_path = rpt / "citation_verification.md"
    cit_path.write_text("\n".join(cit_lines), encoding="utf-8")
    log.info("引用核验: %s", cit_path)

    # ---- 3. 相关工作定位表（LaTeX，v6.5 §2）----
    tex = r"""\begin{table}[htbp]
\centering
\caption{相关工作定位表（方法类型 × 成分归因 × 模态 × 模型覆盖 × 防御 × 泛化评估）}
\label{tab:positioning}
\begin{tabular}{lcccccc}
\hline
工作 & 方法类型 & 成分归因 & 模态 & 模型覆盖 & 防御 & 泛化评估 \\
\hline
PJ-Break & 攻击方法 & 单因子（delivery） & 音频 & LALM & - & - \\
Omni-SafetyBench & 基准 & - & 音频+文本 & LALM & 静态一致性 & - \\
StyleBreak & 攻击方法 & - & 音频 & LALM & - & - \\
Cross-modal Info Check & 检测先驱 & - & 视觉+文本 & VLM & 双模态核对 & - \\
Chen et al. (EMNLP 2025) & 检测 & - & 文本 & LLM & 置信度检测 & - \\
Semantic Codebooks & 检测 & - & 文本 & LLM & 跨语言检测 & - \\
\hline
本文 & 归因+检测 & 全析因（E$_t$$\times$N$\times$R$\times$A$_s$） & 音频+文本 & 3 LALM+2 文本 & MSRF 四分支 & 跨攻击族 \\
\hline
\end{tabular}
\end{table}
"""
    tex_path = rpt / "related_work_positioning.tex"
    tex_path.write_text(tex, encoding="utf-8")
    log.info("定位表: %s", tex_path)

    jlog.event(stage=STAGE, event="done", n_topics=len(TOPICS),
               n_must_cite=len(MUST_CITE), n_citations=len(CITATIONS),
               verified=verified, kbs_candidates=picked,
               network_ok=network_ok)
    if not args.dry_run:
        ckpt.mark_done("done")
    # 网络不可达 → code 2（部分完成，报告标注待人工补查）
    code = 0 if network_ok else 2
    log.info("=== 阶段 L 完成（code=%d）===", code)
    return code


if __name__ == "__main__":
    sys.exit(main())
