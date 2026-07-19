"""LlamaIndex PSE — resume-tailor 任务：RAG 加持的简历定制/推荐。

两种模式：
    1. JD 定制模式：传入 JD，RAG 检索文档，定制针对性简历
    2. 自由推荐模式：无需 JD，根据你的经历 + 国内招聘行情，推荐最适合的岗位

用法:
    python run.py --jd path/to/jd.md          # JD 定制模式
    python run.py --recommend                  # 自由推荐模式（无需 JD）
    python run.py --docs /path/to/docs         # 指定文档目录（默认 work/docs）
    python run.py --provider agnes              # 使用 Agnes 网关
    python run.py --provider scnet-minimax      # 使用 SCNet MiniMax 网关
    python run.py --provider scnet-kimi         # 使用 SCNet Kimi 网关
"""

import argparse
import asyncio
import difflib
import json
import os
import re
import sys
from pathlib import Path

# LlamaIndex Workflow 内部创建 event loop，需要 nest_asyncio 允许嵌套
import nest_asyncio
nest_asyncio.apply()

# 运行时脱敏：发往外部 LLM / Embedding API 前遮蔽直接个人标识符
from privacy import redact, finalize
from llamaindex_pse.prompts import _substitute

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _get_company_periods() -> dict[str, tuple[str, str]]:
    """从 .env RESUME_COMPANY_PERIODS 解析公司任职期间。

    JSON 格式: {"公司名": "入职~离职"}（用 ~ 分隔入职/离职）
    """
    try:
        from llamaindex_pse.config import settings
        raw = settings.RESUME_COMPANY_PERIODS
        if raw and raw != "{}":
            data = json.loads(raw)
            return {k: tuple(v.split("~", 1)) if "~" in v else (v, v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _get_required_companies() -> list[str]:
    """从 .env RESUME_REQUIRED_COMPANIES 解析必须覆盖的公司列表。"""
    try:
        from llamaindex_pse.config import settings
        raw = settings.RESUME_REQUIRED_COMPANIES
        if raw:
            return [c.strip() for c in raw.split(",") if c.strip()]
    except Exception:
        pass
    return []


def _normalize_github_links(text: str) -> str:
    """归一开源链接：LLM 偶发写出 github.com/<user>(<url>) 破损写法，
    统一修正为标准 Markdown [url](url)。"""
    from llamaindex_pse.config import settings
    _gh_user = getattr(settings, "GITHUB_USERNAME", "")
    if _gh_user:
        text = re.sub(rf"github\.com/{re.escape(_gh_user)}\(<?(https?://[^)>]+)>?\)", r"[\1](\1)", text)
    text = re.sub(r"\(<(https?://[^>\s]+)>\)", r"[\1](\1)", text)
    return text


def _apply_target_role(text: str, role: str) -> str:
    """确定性固定目标岗位：大标题与求职意向·期望职位均强制为 role。

    仅用于自由推荐模式且设置了 RESUME_TARGET_ROLE 时，保证标题不再随 LLM 采样漂移。
    兼容 DeepSeek 格式（# 姓名 | 岗位）和 Agnes 格式（# 姓名 - 岗位/岗位）。
    """
    # 大标题：匹配 # 姓名 <分隔符> 任意岗位 两种变体
    #   DeepSeek: # ［姓名］ | AI 工程师
    #   Agnes:    # ［姓名］ - AI 工程化工程师 / LLM 应用工程师
    # 用 [^|\n-]+ 吃姓名（不含 | 和 -），再用 [|\-] 匹配分隔符
    text = re.sub(
        r"^(#\s[^|\n-]+)[|\-].*$",
        lambda m: f"{m.group(1).strip()} | {role}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    # 求职意向·期望职位（兼容 期望职位/期望岗位/期望方向 三种表述）
    text = re.sub(
        r"(\*\*(?:期望职位|期望岗位|期望方向)\*\*[：:]\s*).*",
        lambda m: f"{m.group(1)}{role}",
        text,
    )
    return text


def _load_base_tagline() -> str:
    """读取基准简历的 `>` 标语行（缺失时回填，保证多 provider 产出样式一致）。"""
    try:
        from llamaindex_pse.config import settings
        repo_root = BASE.parent.parent.parent.parent
        name = getattr(settings, "RESUME_SOURCE_FILE", None) or "ai-engineering.md"
        base_path = repo_root / "work" / "docs" / "resume2026ppcnlean-v2" / name
        if base_path.exists():
            for ln in base_path.read_text(encoding="utf-8").splitlines():
                if ln.strip().startswith(">"):
                    return ln.strip()
    except Exception:
        pass
    return ""


def _normalize_job_intent(resume: str) -> str:
    """归一化「求职意向/期望」章节：无论 LLM 生成几种变体
    （顶部 ## 求职意向 / 底部 ## 求职期望 / 标题下 inline 求职意向：…），
    统一合并为恰好一个 ## 求职意向 章节，且期望职位强制为目标岗位（RESUME_TARGET_ROLE）。

    根因：后处理只「补全」不「去重」，LLM 偶发同时产出顶部 ## 求职意向 与
    底部 ## 求职期望（且顶部误用 **求职意向** key 逃过 _apply_target_role 的匹配），
    造成重复且自相矛盾。此处用逐行解析确定性合并，彻底消除该问题。
    """
    from llamaindex_pse.config import settings
    role = getattr(settings, "RESUME_TARGET_ROLE", "") or ""

    collected: dict[str, str | None] = {"期望职位": None, "工作地点": None, "理想方向": None}
    _INTENT_KEY_RE = re.compile(
        r"^\s*\*{0,2}(期望职位|期望岗位|期望方向|求职意向|工作地点|理想方向)\*{0,2}[：:]\s*(.+?)\s*$"
    )
    _SECTION_RE = re.compile(r"^##\s*求职(意向|期望|意向[·\-]?期望职位|期望职位)\b")

    def _ingest_line(line: str) -> None:
        m = _INTENT_KEY_RE.match(line)
        if not m:
            return
        key, val = m.group(1), m.group(2).strip()
        norm = "期望职位" if key in ("期望职位", "期望岗位", "期望方向", "求职意向") else key
        if collected[norm] is None:
            collected[norm] = val

    lines = resume.split("\n")
    first_section = next((i for i, ln in enumerate(lines) if re.match(r"^##\s", ln)), len(lines))

    # 1a. 收集标题块内 inline 求职行
    for ln in lines[:first_section]:
        _ingest_line(ln)

    # 1b. 收集所有 ## 求职* 章节区间
    section_ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if _SECTION_RE.match(lines[i]):
            start = i
            i += 1
            while i < len(lines) and not re.match(r"^##\s", lines[i]):
                _ingest_line(lines[i])
                i += 1
            section_ranges.append((start, i))
        else:
            i += 1

    # 2. 删除章节整段 + 标题块内 inline 求职行
    remove_idx: set[int] = set()
    for s, e in section_ranges:
        for j in range(s, e):
            remove_idx.add(j)
    for j in range(first_section):
        if _INTENT_KEY_RE.match(lines[j]):
            remove_idx.add(j)
    new_lines = [ln for j, ln in enumerate(lines) if j not in remove_idx]

    # 3. 组装规范章节（期望职位强制为目标岗位，消除顶部/底部自相矛盾）
    position = role or collected["期望职位"] or getattr(settings, "RESUME_DEFAULT_POSITION", "") or ""
    location = collected["工作地点"] or getattr(settings, "RESUME_DEFAULT_LOCATION", "") or ""
    direction = collected["理想方向"] or getattr(settings, "RESUME_DEFAULT_DIRECTION", "") or ""
    if not (position or location or direction):
        return resume

    block = ["## 求职意向", "", f"**期望职位**：{position}"]
    if location:
        block.append(f"**工作地点**：{location}")
    if direction:
        block.append(f"**理想方向**：{direction}")

    # 4. 插入到首个 ## 之前
    fs2 = next((i for i, ln in enumerate(new_lines) if re.match(r"^##\s", ln)), len(new_lines))
    prefix = new_lines[:fs2]
    while prefix and not prefix[-1].strip():
        prefix.pop()
    result = prefix + ["", ""] + block + [""] + new_lines[fs2:]
    return "\n".join(result)


def _ensure_tagline(resume: str) -> str:
    """确保标题后存在一行 `>` 标语（与基准简历一致）；缺失则回填，保证多 provider 样式统一。"""
    if re.search(r"(?m)^>\s", resume):
        return resume
    tagline = _load_base_tagline()
    if not tagline:
        return resume
    lines = resume.split("\n")
    title_idx = next((i for i, ln in enumerate(lines) if re.match(r"^#\s", ln)), None)
    if title_idx is None:
        return resume
    insert_idx = len(lines)
    for j in range(title_idx + 1, len(lines)):
        if lines[j].strip().startswith("---") or lines[j].strip().startswith("## "):
            insert_idx = j
            break
    lines.insert(insert_idx, tagline)
    return "\n".join(lines)


# 代表项目在基准简历中的「项目名关键词 → 确认起止区间」映射。
# 关键词用于匹配生成的 ### 标题（agnes 用「项目名 (公司 | 区间)」、deepseek 用「公司 - 项目名」+ 粗体行两种写法）。
_PROJECT_KEYWORDS = ["迁移验证", "身份验证", "迪士尼", "营销落地页", "国际站内容平台"]

# 兜底：基准简历未按纪律列出 CIP（避免稀释 AI 品牌），但 LLM 偶发把 CIP 结束日
# 从 PayPal 任职 tenure（2026.07）推断，而非真实项目结束日。真值来源为
# work/docs/resume-fragments/paypal-cip.md（2025.01-2025.12）。此处硬编码兜底，
# 确保 _normalize_project_dates 即使基准无 CIP 也能正确归一，不受 rebuild 影响。
_PROJECT_DATE_OVERRIDES = {"身份验证": "2025.01-2025.12"}


def _build_project_date_map(base_text: str) -> dict[str, str]:
    """从基准简历「代表项目」段解析 关键词→确认起止区间（YYYY.MM-YYYY.MM）。"""
    m = re.search(r"##\s*代表项目\s*\n(.*?)(?=\n##\s|\Z)", base_text, re.DOTALL)
    section = m.group(1) if m else ""
    out: dict[str, str] = {}
    for kw in _PROJECT_KEYWORDS:
        hm = re.search(r"^###\s+.*?" + re.escape(kw) + r".*?$", section, re.M)
        if not hm:
            continue
        dm = re.match(r"\s*\*\*([\d]{4}(?:\.\d\d)?\s*-\s*[\d]{4}(?:\.\d\d)?)\s*\|", section[hm.end():])
        if dm:
            out[kw] = dm.group(1)
    out.update(_PROJECT_DATE_OVERRIDES)
    return out


def _normalize_project_dates(resume: str, base_text: str) -> str:
    """以基准简历确认的起止区间为准，强制归一生成简历中的项目日期。

    根因：LLM 偶发不照抄基准区间，而是从任职 tenure 推断（如把 CIP 写成 2025.01-2026.07
    而非基准确认的 2025.01-2025.12），且 agnes/deepseek 两种标题写法并存。
    此处确定性地把每个项目的日期替换为基准确认值，保证多 provider 产物日期一致且准确。
    """
    date_map = _build_project_date_map(base_text)
    if not date_map:
        return resume
    lines = resume.split("\n")
    for i, line in enumerate(lines):
        if not line.startswith("### "):
            continue
        matched = next((kw for kw in date_map if kw in line), None)
        if not matched:
            continue
        canonical = date_map[matched]
        # case A: 标题内嵌 (公司 | RANGE) —— agnes 写法
        new_line, n = re.subn(
            r"(\|\s*)[\d]{4}(?:\.\d\d)?\s*-\s*[\d]{4}(?:\.\d\d)?(?=\s*\))",
            lambda mm: mm.group(1) + canonical,
            line,
        )
        if n:
            lines[i] = new_line
            continue
        # case B: 标题下（可能隔 1 个空行）的 **RANGE | 技术栈** —— deepseek 写法
        # deepseek 常在 ### 标题与日期行之间插一个空行，故向前扫描若干行定位日期行。
        for j in range(i + 1, min(i + 4, len(lines))):
            if lines[j].startswith("**") and re.match(
                r"^\*\*[\d]{4}(?:\.\d\d)?\s*-\s*[\d]{4}(?:\.\d\d)?", lines[j]
            ):
                lines[j], _ = re.subn(
                    r"^(\*\*)[\d]{4}(?:\.\d\d)?\s*-\s*[\d]{4}(?:\.\d\d)?",
                    lambda mm: mm.group(1) + canonical,
                    lines[j],
                )
                break
    return "\n".join(lines)


def _normalize_opensource_section(resume: str, base_text: str) -> str:
    """开源章节永远以基准简历为准，杜绝 LLM 把 prompt 内部指令或越界仓库抄进产物。

    根因：agnes 曾把 recommend_specialist.md 的开源"硬性限制"规则 + 6 仓库候选列表原样复制到
    简历末段，标题还带"（⚠️ 违反以下规则即不合格）"。此处确定性地删除任何 LLM 生成的开源章节
    （不论标题写法多脏，含括号后缀），再从基准简历"## 个人开源与 AI 实验"块原样重插，
    保证与基准精简口径（当前 3 个仓库）一致。
    """
    m = re.search(
        r"##\s*个人开源与 AI 实验\s*\n(.*?)(?=\n##\s|\Z)", base_text, re.DOTALL
    )
    if not m:
        return resume
    canonical = "## 个人开源与 AI 实验\n" + m.group(1).strip()

    # 1. 删除 LLM 生成的任何开源章节（标题可能带括号后缀）
    resume = re.sub(
        r"(?m)^##\s*(?:个人开源与 AI 实验|开源项目|开源贡献|开源经历)\b.*$\n(?:.*\n)*?(?=^##\s|\Z)",
        "",
        resume,
    )

    # 2. 在"## 教育背景"前（或文末）重插规范块
    if re.search(r"(?m)^##\s*教育背景", resume):
        resume = re.sub(
            r"(?m)(^##\s*教育背景)",
            canonical + "\n\n" + r"\1",
            resume,
            count=1,
        )
    else:
        resume = resume.rstrip() + "\n\n" + canonical + "\n"
    return resume


def _reorder_projects_by_date(resume: str, latest_end: str = "", latest_company: str = "") -> str:
    """确定性地把所有「重点项目」块按 (结束年月, 起始年月) 倒序重排。

    不依赖 LLM 是否产出 ``## 重点项目`` 章节：只要 ``###`` 标题或紧跟行含
    ``(公司 | YYYY.MM - YYYY.MM)`` 即视为项目块，按结束月份倒序、起始月份倒序
    稳定排列（平局保持原顺序）。非项目块（工作经历 ``### 公司 | 职位``、``##`` 章节）
    位置不变。

    排序键 (end_ym, start_ym) 倒序 => 最新结束的排最前；同结束时起始更晚的排更前。
    """
    # 当前公司「至今」替换为实际离职时间（与历史逻辑一致）
    if latest_end:
        # 项目标题格式：(公司 | X-至今) —— 仅在 latest_company 能精确匹配时生效
        if latest_company:
            resume = re.sub(
                rf"\({re.escape(latest_company)} \| (\d{{4}}\.\d{{2}})-至今\)",
                f"({latest_company} | \\1-{latest_end})",
                resume,
            )
        # 公司级标题格式：### 公司 | X-至今（无括号）。标题里的公司名可能是
        # 「PayPal 贝宝支付」而 latest_company 仅「PayPal」，故不能用精确匹配，
        # 改为：仅对公司标题含 latest_company 子串的「X-至今」兜底为「X-{latest_end}」
        # （latest_company 为空时退化为任意 ### 标题，保持通用兜底）。
        # 已带确定日期的项目块如 CIP(2025.01-2025.12) 不含「至今」，不受影响；
        # 其结束日由 _normalize_project_dates 的 override 另行兜底。
        anchor = re.escape(latest_company) if latest_company else r".*?"
        resume = re.sub(
            rf"(^###\s+.*?{anchor}.*?\|\s*\d{{4}}\.\d{{2}}\s*[-–—]\s*)至今(?=\s*$|\s*\))",
            lambda m: m.group(1) + latest_end,
            resume,
            flags=re.MULTILINE,
        )

    def _project_date(seg: str):
        """从项目块提取 (结束年月, 起始年月)；非项目块返回 None。

        判定为项目的依据（可靠，不会误伤工作经历块）：
        - 标题含 ``(公司 | YYYY.MM - YYYY.MM)``（agnes 写法）；或
        - 标题之后首个非空行是 ``**YYYY.MM - YYYY.MM | 技术栈**``（deepseek 写法，
          标题与日期间允许有空行）。
        工作经历块（``### 公司 | 职位``）标题无日期、首行也非 ``**日期**``，必返回 None。
        """
        seg = seg.lstrip()
        if not seg.startswith("###"):
            return None
        title = seg[3:].split("\n", 1)[0]
        m = re.search(r"\|(\d{4})\.(\d{2})\s*[-–—]\s*(\d{4})\.(\d{2})", title)
        if m:
            sy, sm, ey, em = (int(m.group(i)) for i in (1, 2, 3, 4))
            return (ey * 12 + em, sy * 12 + sm)
        rest = seg.split("\n", 1)[1] if "\n" in seg else ""
        for line in rest.split("\n"):
            if not line.strip():
                continue
            m = re.match(r"^\*\*(\d{4})\.(\d{2})\s*[-–—]\s*(\d{4})\.(\d{2})", line)
            if m:
                sy, sm, ey, em = (int(m.group(i)) for i in (1, 2, 3, 4))
                return (ey * 12 + em, sy * 12 + sm)
            break
        return None

    def _is_project(seg: str) -> bool:
        return _project_date(seg) is not None

    # 用户确认的「代表项目」canonical 顺序：覆盖纯日期排序，避免同年结束的项目
    # （如 CIP 与营销落地页均结束 2025.12）在重排时抖动。关键字按出现优先级
    # 匹配标题，agnes / deepseek 两种标题写法都能命中。
    canonical_order = [
        "迁移验证",     # AI 支付迁移验证与自动化工程化（已含方法论预研）
        "客户身份验证", # 客户身份验证平台 (CIP)
        "营销",         # 营销落地页平台组件开发
        "迪士尼",       # 迪士尼 AI 数字人客服系统
        "Trip.com",     # Trip.com 国际站内容平台
    ]

    def _rank(seg: str) -> int:
        title = seg.lstrip()[3:].split("\n", 1)[0]
        for idx, kw in enumerate(canonical_order):
            if kw in title:
                return idx
        return len(canonical_order)  # 未知项目靠后，按日期倒序

    def _key(seg):
        d = _project_date(seg)
        rank = _rank(seg)
        # rank 升序优先；同 rank（未知项目）时按 (结束月,起始月) 倒序
        if d is not None:
            return (rank, -d[0], -d[1])
        return (rank, 0, 0)

    parts = re.split(r"(?=\n###\s)", resume)
    out = list(parts)
    i, n = 0, len(parts)
    while i < n:
        if _is_project(parts[i]):
            j = i
            while j < n and _is_project(parts[j]):
                j += 1
            out[i:j] = sorted(parts[i:j], key=_key, reverse=False)
            i = j
        else:
            i += 1
    return "".join(out)


def _postprocess_resume(resume: str, base_text: str = "") -> str:
    """确定性后处理：修正 LLM 不遵循的结构性规则。

    1. 强制删除配置的年限表述
    2. 把项目标题下一行的 **(公司 | 时间)** 合并到标题中
    3. 重点项目按时间倒序重排
    4. 项目日期以基准简历确认区间为准强制归一（防 LLM 推断偏差）
    """
    from llamaindex_pse.config import settings

    # 0. 大小写规范化（LLM 偶发 Typescript 小写 s）
    resume = re.sub(r"\bTypescript\b", "TypeScript", resume)

    # 0a. 剥离 LLM 对话前言：流水线保存的是 Fix/Specialist 的完整回复，
    #     可能以"根据问题清单和真实数据…"之类的 wrapper 开头。
    #     从首个 Markdown 标题(# )起截取正文，丢弃标题前的所有聊天内容。
    _h = re.search(r"(?m)^#\s", resume)
    if _h:
        resume = resume[_h.start():]
    # 去掉正文开头可能残留的分隔线/空行
    resume = re.sub(r"^\s*(?:\*\*\*|------+|---+)\s*\n+", "", resume, count=1)

    # 0b. 修正开源项目链接格式（详见 _normalize_github_links）
    resume = _normalize_github_links(resume)

    # 0c. 求职意向/期望 章节归一化：合并所有变体为单一 ## 求职意向（确定性，防 LLM 重复产出）
    resume = _normalize_job_intent(resume)
    # 0d. 标语行回填：缺失时从基准简历注入，保证多 provider 产出样式一致
    resume = _ensure_tagline(resume)
    # 0e. 项目日期以基准简历确认区间为准强制归一（防 LLM 从 tenure 推断偏差）
    if base_text:
        resume = _normalize_project_dates(resume, base_text)
        # 0f. 开源章节以基准简历为准重插（防 LLM 抄 prompt 内部规则 / 越界仓库）
        resume = _normalize_opensource_section(resume, base_text)

    # 1. 删除配置的年限表述（如"20年"、"二十年"等）
    banned_years = settings.RESUME_BANNED_YEARS
    if banned_years:
        # 支持多种模式：20年、20 年、二十年
        for pattern in banned_years.split("|"):
            pattern = pattern.strip()
            if not pattern:
                continue
            # 删除"具备 X年..."句式
            resume = re.sub(rf"具备\s*{pattern}\s*年[^，。,.\n]*[，。,.\n]?", "", resume)
            resume = re.sub(rf"{pattern}\s*年[^，。,.\n]*经验", "", resume)
            resume = re.sub(rf"{pattern}[年\s]+[^，。,.\n]*经验", "", resume)
            # 清理残留变体
            resume = re.sub(rf"\b{pattern}\s*年\b", "", resume)
            resume = re.sub(rf"{pattern}年", "", resume)
            # 连同前面的程度副词（近/约/已/超过/达）一起删除，避免留下「近 ，」这类破句
            resume = re.sub(rf"(近|约|已|超过|达)\s*{pattern}\s*年", "", resume)

    # 2. 合并项目标题下一行的时间到标题中
    #    匹配: ### 项目名\n**(公司 | 时间)**  →  ### 项目名 (公司 | 时间)
    #    也匹配: ### 项目名 (已有公司)\n**(公司 | 时间)**  →  ### 项目名 (公司 | 时间)
    lines = resume.split("\n")
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 检查是否是项目标题（### 开头，且不在标题中包含年份）
        if line.startswith("### ") and not re.search(r"\d{4}", line):
            # 看下一行是否是 **(公司 | 时间)** 格式
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                m = re.match(r"^\*\*\((.+?)\)\*\*$", next_line)
                if m:
                    time_info = m.group(1)
                    # 合并到标题
                    line = f"{line} ({time_info})"
                    i += 1  # 跳过下一行
        merged.append(line)
        i += 1

    resume = "\n".join(merged)

    # 2b. 当前公司"至今"替换为实际离职时间（从 .env 读取）
    try:
        from llamaindex_pse.config import settings
        latest_end = settings.RESUME_LATEST_COMPANY_END
        latest_company = settings.RESUME_LATEST_COMPANY
        latest_start = settings.RESUME_LATEST_COMPANY_START
    except Exception:
        latest_end, latest_company, latest_start = "", "", ""
    # 当前公司「至今」统一替换为实际离职时间（latest_end）。
    # 采用「日期区间锚定」正则：凡 "YYYY.MM - 至今" 不论粗体/斜体/括号/职位前缀、
    # 不论 | 前后有无空格，一律替换为 "YYYY.MM - {latest_end}"。仅匹配真实日期区间，
    # 不会误伤正文里的「至今」二字。
    if latest_end:
        resume = re.sub(
            rf"(\d{{4}}\.\d{{2}}\s*[-–—]\s*)至今",
            rf"\g<1>{latest_end}",
            resume,
        )

    # 2c. 安全网：修正已知 LLM 非确定性错误

    # PayPal 项目时间修正：LLM 常将迁移验证/测试体系项目时间写成晚于入职的起始时间，
    # 实际这些项目从入职即开始。通过配置识别需要修正的项目。
    _late_start_cfg = getattr(settings, "RESUME_LATE_START_FIX", "")
    # 格式: "2026.01:测试体系|工程化|多语言.*电商" (起始时间:内容关键词)
    if latest_start and _late_start_cfg and ":" in _late_start_cfg:
        _late_start_time, _late_keywords = _late_start_cfg.split(":", 1)
        _paypal_late_start_pattern = re.compile(
            rf"\({re.escape(latest_company)} \| {re.escape(_late_start_time)}-(\d{{4}}\.\d{{2}})\)"
        )
        _late_start_matches = list(_paypal_late_start_pattern.finditer(resume))
        for m in reversed(_late_start_matches):
            block_start = m.start()
            header_start = resume.rfind("\n### ", 0, block_start)
            if header_start == -1:
                header_start = resume.rfind("### ", 0, block_start)
            next_section = re.search(r"\n(?:###|##)\s", resume[block_start:])
            block_end = block_start + next_section.start() if next_section else len(resume)
            block_content = resume[header_start:block_end]
            if re.search(_late_keywords, block_content):
                resume = resume[:m.start()] + f"({latest_company} | {latest_start}-{m.group(1)})" + resume[m.end():]

    # 当前公司抬头：强制修正为配置的正确职位（源文档一致）
    # 格式1：### 公司名 | 错误职位
    if latest_company:
        _wrong_title = getattr(settings, "RESUME_WRONG_TITLE", "")  # e.g. "全栈工程师"
        _correct_title = getattr(settings, "RESUME_CORRECT_TITLE", "")  # e.g. "高级全栈工程师"
        if _wrong_title and _correct_title:
            resume = re.sub(
                rf"(### {re.escape(latest_company)}[^|]*)\| {re.escape(_wrong_title)}",
                rf"\1| {_correct_title}",
                resume,
            )
    # 格式2：**错误职位 | YYYY.MM - YYYY.MM**（加粗变体）
    if _wrong_title and _correct_title:
        resume = re.sub(
            rf"\*\*{re.escape(_wrong_title)} \| (\d{{4}}\.\d{{2}} - (?:至今|\d{{4}}\.\d{{2}}))\*\*",
            rf"**{_correct_title} | \1**",
            resume,
        )

    # 已知幻觉数字清理：LLM 编造但源文档不存在的数据（从配置读取）
    _hallucinated = getattr(settings, "RESUME_HALLUCINATIONS", "").split("|") if getattr(settings, "RESUME_HALLUCINATIONS", "") else []
    for _h in _hallucinated:
        while _h in resume:
            # 删除该模式及其后面的逗号/顿号/空格/的
            idx = resume.find(_h)
            end = idx + len(_h)
            if end < len(resume) and resume[end] in ("、", "，", ","):
                end += 1
            elif end < len(resume) and resume[end] == " ":
                end += 1
            elif resume[end:end+1] == "的":
                end += 1
            resume = resume[:idx] + resume[end:]
        # 清理残留：删除后可能产生的「覆盖 」→「覆盖」(空格多余)或 「、」(开头残留顿号)
        resume = re.sub(r"覆盖\s+、", "覆盖、", resume)
        resume = re.sub(r"覆盖\s+，", "覆盖，", resume)
        # 行首残留的两个空格
        resume = re.sub(r"\n\s+、", "\n", resume)
        resume = re.sub(r"\n\s+，", "\n", resume)

    # 3. 重点项目按 (结束年月, 起始年月) 倒序确定性重排（对所有 ### 项目块生效）
    resume = _reorder_projects_by_date(resume, latest_end, latest_company)

    # 4. 求职意向章节补全（如果只有标题没有内容）
    intent_match = re.search(r"(##\s*求职意向\n+)(.*?)(?=\n##\s|\Z)", resume, re.DOTALL)
    if intent_match:
        intent_body = intent_match.group(2).strip()
        if not intent_body or len(intent_body) < 10:
            # 从简历标题提取期望职位，或使用 .env 配置的默认值
            title_match = re.search(r"#\s+.+\|\s*(.+)", resume.split("\n")[0])
            position = title_match.group(1).strip() if title_match else settings.RESUME_DEFAULT_POSITION
            location = settings.RESUME_DEFAULT_LOCATION
            direction = settings.RESUME_DEFAULT_DIRECTION
            default_intent = "\n"
            if position:
                default_intent += f"**期望职位**：{position}\n"
            if location:
                default_intent += f"**工作地点**：{location}\n"
            if direction:
                default_intent += f"**理想方向**：{direction}\n"
            resume = resume.replace(
                intent_match.group(0),
                intent_match.group(1) + default_intent + "\n",
            )

    # 5. 求职意向章节：确保各要点独立成行
    #    Markdown 中连续行（无空行）会被合并为一个段落，导致
    #    **期望职位** / **工作地点** / **理想方向** 渲染时挤成一行。
    #    在相邻非空、非标题行之间插入空行，使每条独立成段、渲染分行。
    intent_match = re.search(r"(##\s*求职意向\n+)(.*?)(?=\n##\s|\Z)", resume, re.DOTALL)
    if intent_match:
        body = intent_match.group(2)
        lines = body.split("\n")
        new_lines = []
        for i, ln in enumerate(lines):
            new_lines.append(ln)
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if ln.strip() and not ln.strip().startswith("#") and nxt.strip() and not nxt.strip().startswith("#"):
                new_lines.append("")
        fixed = intent_match.group(1) + "\n".join(new_lines)
        resume = resume.replace(intent_match.group(0), fixed)

    # 6. 跨章节确定性去重：删除「工作经历」中与「重点项目」逐字/高度相似的要点
    #    保留更详细的「重点项目」版本，避免依赖 LLM 重试仍残留重复。
    def _norm(b: str) -> str:
        return re.sub(r"\s+", "", b)

    work_m = re.search(r"##\s*工作经历(.*?)(?=\n##\s|\Z)", resume, re.DOTALL)
    proj_m = re.search(r"##\s*重点项目(.*?)(?=\n##\s|\Z)", resume, re.DOTALL)
    if work_m and proj_m:
        proj_bullets = [
            _norm(ln.strip()[2:].strip())
            for ln in proj_m.group(1).splitlines()
            if ln.strip().startswith("- ") and len(_norm(ln.strip()[2:].strip())) >= 10
        ]
        work_lines = work_m.group(1).split("\n")
        out_lines = []
        for ln in work_lines:
            if ln.strip().startswith("- "):
                wb = _norm(ln.strip()[2:].strip())
                if len(wb) >= 10 and any(
                    wb == pb or difflib.SequenceMatcher(None, wb, pb).ratio() >= 0.82
                    for pb in proj_bullets
                ):
                    continue  # 与重点项目重复 → 删除（保留项目版）
            out_lines.append(ln)
        # 折叠连续空行
        collapsed = []
        for ln in out_lines:
            if collapsed and not collapsed[-1].strip() and not ln.strip():
                continue
            collapsed.append(ln)
        resume = resume[: work_m.start(1)] + "\n".join(collapsed) + resume[work_m.end(1):]

    return resume


def _verify_resume(resume: str, rag_context: str) -> tuple[list, list]:
    """程序化验证：简历中的关键声明应能在 RAG 上下文中找到出处。"""
    from llamaindex_pse.config import settings

    bad: list[str] = []
    ok: list[str] = []

    if not rag_context:
        bad.append("无 RAG 上下文可供核查，简历可能包含未验证内容")
        return bad, ok

    # 1. 检查简历中的年份/年限是否在上下文中出现
    #    宽松策略：只标记明显不在源文档范围中的年份（如 1990、2030），
    #    常见范围 2001-2026 内的年份视为合理（源文档覆盖该范围）
    year_pattern = r"\b(20\d{2})\b"
    resume_years = set(re.findall(year_pattern, resume))
    context_years = set(re.findall(year_pattern, rag_context))
    # 源文档覆盖的年份范围
    context_year_min = min(int(y) for y in context_years) if context_years else 2000
    context_year_max = max(int(y) for y in context_years) if context_years else 2030
    for y in resume_years:
        y_int = int(y)
        if y in context_years:
            ok.append(f"年份 {y} 在源文档中存在")
        elif context_year_min <= y_int <= context_year_max:
            ok.append(f"年份 {y} 在源文档范围 ({context_year_min}-{context_year_max}) 内，合理")
        else:
            bad.append(f"年份 {y} 不在源文档范围 ({context_year_min}-{context_year_max}) 内，可能虚构")

    # 2. 检查量化数据（数字 + 单位）是否在上下文中出现
    #    百分比（如 95%、99%）是常见估算值（覆盖率/成功率），不严格校验
    quant_pattern = r"(\d+(?:\.\d+)?%|\d+(?:,\d{3})+(?:\+)?|\d+\+?\s*(?:人|万|倍|ms|GB|TB|个))"
    resume_quants = set(re.findall(quant_pattern, resume))
    context_quants = set(re.findall(quant_pattern, rag_context))
    for q in resume_quants:
        if q in context_quants:
            ok.append(f"量化数据 {q} 在源文档中存在")
        elif q.endswith("%"):
            ok.append(f"百分比 {q} 为合理估算（覆盖率/成功率等），跳过校验")
        else:
            # 模糊匹配：去掉空格后比较（"50万" vs "50 万"）
            q_norm = q.replace(" ", "")
            context_norms = [c.replace(" ", "") for c in context_quants]
            if q_norm in context_norms:
                ok.append(f"量化数据 {q} 在源文档中存在（空格差异）")
            # 金额模糊："75万" 匹配 "75万美金"、"50万" 匹配 "50万美金"
            elif "万" in q and any(q_norm in cn for cn in context_norms):
                ok.append(f"量化数据 {q} 在源文档中模糊匹配到（金额后缀差异）")
            else:
                bad.append(f"量化数据 {q} 未在源文档中找到，可能编造")

    # 2b. 检查工期/时长表述（天/周/月）是否在源文档中出现
    #     避免 LLM 编造具体工期（如「为期 12 天的 POC 项目」），这类跨度单位不在量化正则内
    dur_pattern = r"\d+(?:\.\d+)?\s*(?:天|周|个月|月)"
    resume_durs = re.findall(dur_pattern, resume)
    context_dur_norms = [c.replace(" ", "") for c in re.findall(dur_pattern, rag_context)]
    for d in resume_durs:
        if d.replace(" ", "") in context_dur_norms:
            ok.append(f"工期表述 {d} 在源文档中存在")
        else:
            bad.append(f"工期表述 {d} 未在源文档中找到，可能虚构（如编造的项目周期）")

    # 2c. 跨章节重复检测：「工作经历」与「重点项目」的要点不得重复
    #     程序化兜底（LLM 自评不可靠）：抽取两章节的 - 要点，归一化后比对
    #     逐字相同直接判重；高度相似（difflib 相似度 ≥ 0.82）也判重，触发重试闭环
    def _section_bullets(text: str, header: str) -> list[str]:
        m = re.search(rf"##\s*{header}(.*?)(?=\n##\s|\Z)", text, re.DOTALL)
        if not m:
            return []
        return [ln.strip()[2:].strip() for ln in m.group(1).splitlines() if ln.strip().startswith("- ")]

    def _norm_bullet(b: str) -> str:
        return re.sub(r"\s+", "", b)

    work_bullets = _section_bullets(resume, "工作经历")
    proj_bullets = _section_bullets(resume, "重点项目")
    for wb in work_bullets:
        wn = _norm_bullet(wb)
        if len(wn) < 10:
            continue
        for pb in proj_bullets:
            pn = _norm_bullet(pb)
            if len(pn) < 10:
                continue
            if wn == pn:
                bad.append(f"工作经历与重点项目重复要点（逐字）: {wb}")
                break
            if difflib.SequenceMatcher(None, wn, pn).ratio() >= 0.82:
                bad.append(f"工作经历与重点项目存在高度相似要点（相似度高）: 工作经历「{wb}」≈ 重点项目「{pb}」")
                break

    # 2d. 已知幻觉数据检测：源文档不存在的编造数字/表述
    _hallucinated = ["13 个电商平台", "13 平台", "57 个文档字段", "57 个字段"]
    for _h in _hallucinated:
        if _h in resume:
            bad.append(f"已知幻觉数据「{_h}」出现在简历中，该表述不在任何源文档中，必须删除")

    # 3. 基本长度检查
    if len(resume) < 200:
        bad.append("简历内容过短（< 200 字），可能不完整")
    elif len(resume) > 10000:
        bad.append("简历内容过长（> 10000 字），可能包含冗余")

    # 4. 禁止配置的年限表述
    #    注：后处理会强制删除，此处不再作为 hard error，仅记录
    banned_years = settings.RESUME_BANNED_YEARS
    if banned_years:
        banned_patterns = "|".join(p.strip() for p in banned_years.split("|") if p.strip())
        if re.search(rf"({banned_patterns})\s*年|({banned_patterns})年", resume):
            ok.append("年限表述将在后处理中自动删除")
        else:
            ok.append("无年限表述")
    else:
        ok.append("未配置年限过滤（RESUME_BANNED_YEARS）")

    # 5. 重点项目必须有起止时间
    #    匹配 ## 重点项目 下的所有 ### 标题
    project_section = re.search(r"##\s*重点项目(.*?)(?=\n##\s|\Z)", resume, re.DOTALL)
    if project_section:
        section_text = project_section.group(1)
        project_blocks = re.findall(r"^###\s+.+$", section_text, re.MULTILINE)
        projects_without_time = []
        for p in project_blocks:
            # 检查标题中是否包含 YYYY 格式的年份
            if not re.search(r"\d{4}", p):
                projects_without_time.append(p.strip())
        if projects_without_time:
            bad.append(
                f"重点项目缺少起止时间（格式应为'项目名 (公司 | YYYY.MM-YYYY.MM)'）："
                f"{projects_without_time}"
            )
        elif project_blocks:
            ok.append(f"所有 {len(project_blocks)} 个重点项目均包含时间范围")

        # 5b. 项目起止时间必须与对应公司任职时间匹配
        company_periods = _get_company_periods()
        for p in project_blocks:
            company_m = re.search(r"\((\w+)\s*\|", p)
            time_m = re.search(r"(\d{4}\.\d{2})\s*[-–—]\s*(\d{4}\.\d{2})", p)
            if company_m and time_m:
                company = company_m.group(1)
                proj_start, proj_end = time_m.group(1), time_m.group(2)
                if company in company_periods:
                    comp_start, comp_end = company_periods[company]
                    if proj_start < comp_start:
                        bad.append(f"项目 {p.strip()} 起始时间 {proj_start} 早于 {company} 入职 {comp_start}")
                    # 项目结束时间不应超过公司离职时间
                    if proj_end > comp_end:
                        bad.append(f"项目 {p.strip()} 结束时间 {proj_end} 晚于 {company} 离职 {comp_end}")
                    else:
                        ok.append(f"项目 {company} {proj_start}-{proj_end} 在任职期间 {comp_start}-{comp_end} 内")

        # 6. 重点项目应按时间倒序排列（最新的在前）
        if project_blocks:
            # 提取每个项目的起始年份
            project_start_years = []
            for p in project_blocks:
                # 匹配 YYYY.MM 或 YYYY 格式的起始年份
                m = re.search(r"(\d{4})(?:\.\d{2})?\s*[-–—]", p)
                if m:
                    project_start_years.append(int(m.group(1)))
                else:
                    # 尝试匹配括号内的年份
                    m2 = re.search(r"\|\s*(\d{4})", p)
                    project_start_years.append(int(m2.group(1)) if m2 else 0)

            # 检查是否倒序（允许相同年份）
            out_of_order = []
            for i in range(1, len(project_start_years)):
                if project_start_years[i] > project_start_years[i - 1]:
                    out_of_order.append(
                        f"{project_blocks[i].strip()}({project_start_years[i]}) "
                        f"排在 {project_blocks[i-1].strip()}({project_start_years[i-1]}) 前面"
                    )
            if out_of_order:
                ok.append("项目时序将在后处理中自动重排")
            else:
                ok.append("重点项目按时间倒序排列")

        # 7. 每家公司至少 1 个项目（检查公司覆盖）
        required_companies = _get_required_companies()
        project_text = " ".join(project_blocks)
        missing_companies = [c for c in required_companies if c not in project_text]
        if missing_companies:
            bad.append(f"重点项目缺少以下公司的项目：{missing_companies}（每家公司至少 1 个代表性项目）")
        else:
            ok.append("重点项目覆盖所有公司")

    # 8. 求职意向必须存在
    if "## 求职意向" not in resume:
        bad.append("简历缺少'求职意向'章节")
    elif not re.search(r"期望职位|期望岗位", resume):
        bad.append("'求职意向'章节缺少期望职位")
    else:
        ok.append("求职意向章节完整")

    return bad, ok


def _verify_state(state: dict) -> tuple[list, list]:
    # 事实来源：scan_result（完整语料）> resume_source > rag_context 合并，均为有效出处
    task_input = state.get("task_input", "")
    rag_context = state.get("task_data", {}).get("rag_context", "") or state.get("rag_context", "")
    resume_source = state.get("task_data", {}).get("resume_source", "")
    scan_result = state.get("task_data", {}).get("scan_result", "")
    # 合并完整事实语料 + 简历全文 + RAG 上下文作为事实来源
    parts = []
    if scan_result:
        parts.append(scan_result)
    if resume_source:
        parts.append(resume_source)
    if rag_context:
        parts.append(rag_context)
    source_context = "\n\n".join(parts) if parts else task_input
    return _verify_resume(state.get("artifact", ""), source_context)


def _build_index(docs_dir: Path, subdirs: list[str], index_cache_dir: Path,
                  top_k: int, rebuild: bool = False):
    """为指定子目录构建 RAG 索引，支持本地持久化缓存。

    Args:
        docs_dir:    文档根目录 (work/docs)
        subdirs:     要索引的子目录名列表（相对于 docs_dir）
        index_cache_dir: 索引缓存目录
        top_k:       retriever top_k
        rebuild:     是否强制重建
    Returns:
        retriever 或 None
    """
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Document
    from llama_index.core.node_parser import SentenceSplitter

    # 收集文件
    all_files = []
    for subdir in subdirs:
        dir_path = docs_dir / subdir
        if dir_path.exists():
            for p in dir_path.rglob("*"):
                if p.is_file() and not p.name.startswith(".") and p.suffix in (".md", ".txt", ".pdf"):
                    all_files.append(str(p))

    if not all_files:
        print(f"   ⚠️ 分区 {subdirs} 无可索引文件")
        return None

    label = "+".join(subdirs)
    print(f"📚 构建分区索引 [{label}]: {len(all_files)} 个文件")

    # 尝试从缓存加载
    if not rebuild and index_cache_dir.exists():
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(index_cache_dir))
            index = load_index_from_storage(storage_context)
            retriever = index.as_retriever(similarity_top_k=top_k)
            print(f"   索引缓存命中，retriever top_k={top_k}")
            return retriever
        except Exception as e:
            print(f"   ⚠️ 缓存加载失败: {e}，将重新构建")

    # 构建新索引
    try:
        documents = SimpleDirectoryReader(input_files=all_files).load_data()
        if not documents:
            print(f"   ⚠️ 分区 [{label}] 加载 0 个文档片段")
            return None

        # 手术式脱敏：构建索引前遮蔽直接个人标识符（源文件不改动）。
        # 注意：本版本 llama-index 的 Document.text 为只读属性，需重建对象。
        documents = [
            Document(text=redact(d.text), metadata=getattr(d, "metadata", None), id_=getattr(d, "id_", None))
            for d in documents
        ]

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
        retriever = index.as_retriever(similarity_top_k=top_k)
        print(f"   索引构建完成: {len(documents)} 个文档片段, top_k={top_k}")

        index.storage_context.persist(persist_dir=str(index_cache_dir))
        print(f"   索引已缓存 → {index_cache_dir}")
        return retriever
    except Exception as e:
        print(f"❌ 构建分区索引 [{label}] 失败: {e}")
        return None


# ─── 文档分区定义 ───
# resume_source: 简历事实来源（Specialist 用）
RESUME_PARTITIONS = ["resume2026ppcnlean-v2", "resume-fragments"]
# market_intel: 面试/JD/工作细节情报（Planner 用）
# 注意：目录名须与 work/docs 下真实目录一致。原 interview/interview2026/jd/linkedin
# 在整理时已重组进 jobs/（JD+面试）、resume-story/（面试准备/求职策略），故此处对齐。
MARKET_PARTITIONS = ["jobs", "technical", "paypal", "work", "resume-story"]


async def main():
    ap = argparse.ArgumentParser(description="RAG 加持的简历定制/推荐 (llamaindex-pse)")
    ap.add_argument("--jd", type=str, help="JD 文件路径（定制模式）")
    ap.add_argument("--jd-text", type=str, help="JD 文本（定制模式，直接传入）")
    ap.add_argument("--recommend", action="store_true",
                    help="自由推荐模式：无需 JD，根据你的经历 + 国内行情推荐最适合的岗位")
    ap.add_argument("--docs", type=str,
                    default=os.getenv("RESUME_DOCS_PATH", ""),
                    help="文档目录路径（默认从 PSE_ROOT/work/docs 加载）")
    ap.add_argument("--provider", choices=["deepseek", "agnes", "scnet-kimi", "scnet-minimax"], default="deepseek",
                    help="LLM 网关（deepseek / agnes / scnet-kimi / scnet-minimax）")
    ap.add_argument("--top-k", type=int, default=8,
                    help="RAG 检索 top-k 文档数（默认 8）")
    ap.add_argument("--rebuild", action="store_true",
                    help="强制重建索引（忽略本地缓存）")
    args = ap.parse_args()

    # 模式检查
    is_recommend = args.recommend
    has_jd = args.jd or args.jd_text
    if not is_recommend and not has_jd:
        print("❌ 请提供 --jd/--jd-text（定制模式）或 --recommend（推荐模式）")
        sys.exit(1)

    # 读取 JD（定制模式）
    jd_text = ""
    if has_jd:
        if args.jd:
            jd_text = Path(args.jd).read_text(encoding="utf-8")
        else:
            jd_text = args.jd_text

    # 确定文档目录
    docs_dir = Path(args.docs) if args.docs else None
    if not docs_dir:
        pse_root = os.getenv("PSE_ROOT", str(Path.cwd()))
        docs_dir = Path(pse_root) / "work" / "docs"
    if not docs_dir.exists():
        print(f"❌ 文档目录不存在: {docs_dir}")
        sys.exit(1)

    # 加载 LlamaIndex 核心
    try:
        from llamaindex_pse.config import settings
        from llamaindex_pse.model import create_llm, create_embedding
        from llamaindex_pse.workflow import build_workflow
    except Exception as e:
        print(f"❌ 无法加载 llamaindex 运行环境: {e}\n（请先 `uv sync`）")
        sys.exit(1)

    # 配置 LLM + Embedding
    llm = create_llm(args.provider)
    print(f"   LLM: {llm.model_name}")
    try:
        from llama_index.core import Settings
        Settings.embed_model = create_embedding()
        embed_info = f"{settings.EMBEDDING_PROVIDER}/{settings.EMBEDDING_MODEL}"
        print(f"   Embedding: {embed_info}")
    except RuntimeError as e:
        print(f"   ⚠️ {e}，将使用 LlamaIndex 默认 embedding")

    # ── 分区构建 RAG 索引 ──
    # Specialist 用简历源索引，Planner 用市场情报索引
    index_cache_root = BASE / ".index_cache"
    resume_retriever = _build_index(
        docs_dir, RESUME_PARTITIONS,
        index_cache_root / "resume", args.top_k, args.rebuild,
    )
    market_retriever = _build_index(
        docs_dir, MARKET_PARTITIONS,
        index_cache_root / "market", args.top_k, args.rebuild,
    )

    # 加载核心简历全文（供 verify_fn 事实校验用，不再注入 task_input）
    resume_src_name = settings.RESUME_SOURCE_FILE or "ai-engineering.md"
    resume_src = docs_dir / "resume2026ppcnlean-v2" / resume_src_name
    resume_full_for_verify = ""
    if resume_src.exists():
        resume_full_for_verify = resume_src.read_text(encoding="utf-8")
        print(f"   📄 核心简历已加载（供校验用）: {resume_src.name} ({len(resume_full_for_verify)} 字)")

    # 事实核查完整语料：覆盖 resume2026ppcnlean-v2 / paypal / work / resume-fragments 分区（递归）。
    # 作为 Evaluator / Fix 的"真实数据"(scan_result)：
    #  - resume2026ppcnlean-v2 是简历源本身，纳入后能直接核对生成简历与源简历的逐字一致性
    #    （如"30+ 次生产发布"被写成"10+"这类偏差可被捕获）；
    #  - paypal / work / resume-fragments 提供补充事实，避免 LLM 评审把真实事实误判为幻觉。
    verify_corpus_parts = ["resume2026ppcnlean-v2", "paypal", "work", "resume-fragments"]
    _corpus_chunks = []
    for _p in verify_corpus_parts:
        _pd = docs_dir / _p
        if _pd.exists():
            for _f in sorted(_pd.rglob("*")):
                if _f.is_file() and not _f.name.startswith(".") and _f.suffix in (".md", ".txt"):
                    try:
                        _corpus_chunks.append(_f.read_text(encoding="utf-8"))
                    except Exception:
                        pass
    verify_corpus = "\n\n".join(_corpus_chunks)
    print(f"   📚 事实核查语料已构建: {len(_corpus_chunks)} 个文件, {len(verify_corpus)} 字")

    # 构建 PSE workflow
    max_retries = settings.PSE_MAX_RETRIES or 3

    # 构建 Cross-Encoder Reranker（可选，提升 RAG 检索精度）
    reranker = None
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        reranker = SentenceTransformerRerank(model=reranker_model, top_n=args.top_k)
        print(f"   Reranker: {reranker_model}")
    except Exception as e:
        print(f"   ⚠️ Reranker 不可用 ({e})，将使用原始检索排序")

    if is_recommend:
        # ── 自由推荐模式 ──
        # 加载推荐模式的专用提示词
        prompts_dir = BASE / "prompts"
        # 提示词同样可能含真实标识符（如 recommend_specialist.md 的 GitHub 链接），
        # 发往外部 LLM 前先脱敏；最终落盘由 finalize() 从本地配置回填。
        planner_prompt = redact(_substitute((prompts_dir / "recommend_planner.md").read_text(encoding="utf-8")))
        specialist_prompt = redact(_substitute((prompts_dir / "recommend_specialist.md").read_text(encoding="utf-8")))
        evaluator_prompt = redact(_substitute((prompts_dir / "evaluator.md").read_text(encoding="utf-8")))

        workflow = build_workflow(
            llm=llm,
            task="resume-tailor",
            verify_fn=_verify_state,
            use_planner=True,
            max_retries=max_retries,
            provider=args.provider,
            retriever=resume_retriever,        # Specialist: 简历源数据
            planner_retriever=market_retriever, # Planner: 市场/JD 情报
            rag_top_k=args.top_k,
            reranker=reranker,
        )
        # 覆盖提示词为推荐模式专用
        workflow._planner_prompt = planner_prompt
        workflow._specialist_prompt = specialist_prompt
        workflow._evaluator_prompt = evaluator_prompt

        # task_input: 简历全文注入（事实基础）+ 个人特点注入 + 市场情报由 Planner RAG 检索补充
        task_input = (
            "请分析我的职业背景，结合当前国内招聘市场行情，"
            "推荐最适合我的岗位方向，并为排名第一的岗位定制简历。\n\n"
        )
        if resume_full_for_verify:
            task_input += f"## 我的完整简历（以下为事实来源，必须基于此撰写）\n\n{redact(resume_full_for_verify)}\n\n"

        # 注入个人特点文档（直接读取，不依赖 RAG 检索）
        personal_docs_dir = docs_dir / "work"
        if personal_docs_dir.exists():
            personal_parts = []
            for p in sorted(personal_docs_dir.rglob("*.md")):
                if p.name.startswith("."):
                    continue
                content = p.read_text(encoding="utf-8").strip()
                if content:
                    # 截取前 2000 字避免过长
                    if len(content) > 2000:
                        content = content[:2000] + "\n...(截断)"
                    personal_parts.append(f"### {p.stem}\n\n{redact(content)}")
            if personal_parts:
                task_input += (
                    "## 个人特点与职业发展（以下为辅助参考，丰富简历内容）\n\n"
                    + "\n\n".join(personal_parts) + "\n\n"
                )
                print(f"   📋 个人特点文档已注入: {len(personal_parts)} 个文件")

        # 注入市场趋势与项目映射（项目选择的决策指南）
        mapping_file = docs_dir / "resume-fragments" / "market-trend-mapping.md"
        if mapping_file.exists():
            mapping_content = mapping_file.read_text(encoding="utf-8").strip()
            if mapping_content:
                task_input += f"## 市场趋势与项目映射（项目选择决策指南）\n\n{mapping_content}\n\n"
                print(f"   🎯 市场趋势映射已注入")

        rag_keywords = settings.RESUME_RAG_KEYWORDS
        if rag_keywords:
            task_input += (
                "## 补充检索\n"
                f"检索关键词：{rag_keywords}\n"
            )

        target_role = settings.RESUME_TARGET_ROLE
        if target_role:
            task_input += (
                "## 目标岗位（固定，必须严格遵守）\n"
                f"本次简历的目标岗位固定为：{target_role}\n"
                "简历大标题（# 姓名 | 目标岗位）与文末「求职意向·期望职位」必须严格使用该称谓，"
                "不得改写为架构师 / 专家 / 负责人 / 工程师 等其他词。\n\n"
            )
            print(f"   🎯 目标岗位已固定: {target_role}")

        print(f"\n🚀 自由推荐模式 (provider={args.provider}, max_retries={max_retries})")
        print(f"   RAG 分区: Planner→市场情报, Specialist→简历源数据")
        handler = workflow.run(
            task_input=task_input,
            task_data={"resume_source": resume_full_for_verify, "scan_result": verify_corpus},
            max_retries=max_retries,
        )
        result = await handler

        artifact = result.get("artifact", "")
        artifact = _postprocess_resume(artifact, resume_full_for_verify)
        artifact = finalize(artifact)
        artifact = _normalize_github_links(artifact)  # 兜底：finalize 注入的段落也可能含破损链接
        if is_recommend and settings.RESUME_TARGET_ROLE:
            artifact = _apply_target_role(artifact, settings.RESUME_TARGET_ROLE)
        out_path = BASE / f"recommended_resume_{args.provider}.md"
        out_path.write_text(artifact, encoding="utf-8")
        print(f"\n✅ 推荐简历已保存 → {out_path}")

    else:
        # ── JD 定制模式 ──
        workflow = build_workflow(
            llm=llm,
            task="resume-tailor",
            verify_fn=_verify_state,
            use_planner=True,
            max_retries=max_retries,
            provider=args.provider,
            retriever=resume_retriever,        # Specialist: 简历源数据
            planner_retriever=market_retriever, # Planner: 市场/JD 情报
            rag_top_k=args.top_k,
            reranker=reranker,
        )

        # task_input: 简历全文注入（事实基础）+ 个人特点注入
        task_input = f"请根据以下 JD 定制简历：\n\n## 岗位描述\n{jd_text}\n\n"
        if resume_full_for_verify:
            task_input += f"## 我的完整简历（以下为事实来源，必须基于此撰写）\n\n{redact(resume_full_for_verify)}\n\n"

        # 注入个人特点文档（直接读取，不依赖 RAG 检索）
        personal_docs_dir = docs_dir / "work"
        if personal_docs_dir.exists():
            personal_parts = []
            for p in sorted(personal_docs_dir.rglob("*.md")):
                if p.name.startswith("."):
                    continue
                content = p.read_text(encoding="utf-8").strip()
                if content:
                    if len(content) > 2000:
                        content = content[:2000] + "\n...(截断)"
                    personal_parts.append(f"### {p.stem}\n\n{redact(content)}")
            if personal_parts:
                task_input += (
                    "## 个人特点与职业发展（以下为辅助参考，丰富简历内容）\n\n"
                    + "\n\n".join(personal_parts) + "\n\n"
                )
                print(f"   📋 个人特点文档已注入: {len(personal_parts)} 个文件")

        # 注入市场趋势与项目映射
        mapping_file = docs_dir / "resume-fragments" / "market-trend-mapping.md"
        if mapping_file.exists():
            mapping_content = mapping_file.read_text(encoding="utf-8").strip()
            if mapping_content:
                task_input += f"## 市场趋势与项目映射（项目选择决策指南）\n\n{mapping_content}\n\n"
                print(f"   🎯 市场趋势映射已注入")

        print(f"\n🚀 JD 定制模式 (provider={args.provider}, max_retries={max_retries})")
        print(f"   RAG 分区: Planner→市场情报, Specialist→简历源数据")
        handler = workflow.run(
            task_input=task_input,
            task_data={"resume_source": resume_full_for_verify, "scan_result": verify_corpus},
            max_retries=max_retries,
        )
        result = await handler

        resume = result.get("artifact", "")
        resume = _postprocess_resume(resume, resume_full_for_verify)
        resume = finalize(resume)
        resume = _normalize_github_links(resume)  # 兜底：finalize 注入的段落也可能含破损链接
        out_path = BASE / f"tailored_resume_{args.provider}.md"
        out_path.write_text(resume, encoding="utf-8")
        print(f"\n✅ 定制简历已保存 → {out_path}")

    # 打印 token 消耗统计
    from llamaindex_pse.model import token_stats
    print(f"\n{token_stats.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
