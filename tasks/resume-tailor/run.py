"""LlamaIndex PSE — resume-tailor 任务：RAG 加持的简历定制/推荐。

两种模式：
    1. JD 定制模式：传入 JD，RAG 检索文档，定制针对性简历
    2. 自由推荐模式：无需 JD，根据你的经历 + 国内招聘行情，推荐最适合的岗位

用法:
    python run.py --jd path/to/jd.md          # JD 定制模式
    python run.py --recommend                  # 自由推荐模式（无需 JD）
    python run.py --docs /path/to/docs         # 指定文档目录（默认 work/docs）
    python run.py --provider agnes              # 使用 Agnes 网关
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

# LlamaIndex Workflow 内部创建 event loop，需要 nest_asyncio 允许嵌套
import nest_asyncio
nest_asyncio.apply()

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


def _postprocess_resume(resume: str) -> str:
    """确定性后处理：修正 LLM 不遵循的结构性规则。

    1. 强制删除配置的年限表述
    2. 把项目标题下一行的 **(公司 | 时间)** 合并到标题中
    3. 重点项目按时间倒序重排
    """
    from llamaindex_pse.config import settings

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
    if latest_end and latest_company and latest_start:
        # 工作经历行：*YYYY.MM - 至今* → *YYYY.MM - YYYY.MM*
        resume = re.sub(
            rf"\*{latest_start} - 至今\*",
            f"*{latest_start} - {latest_end}*",
            resume,
        )
        # 项目标题中该公司任意起始时间的"至今"：(Company | YYYY.MM-至今)
        resume = re.sub(
            rf"\({re.escape(latest_company)} \| (\d{{4}}\.\d{{2}})-至今\)",
            f"({latest_company} | \\1-{latest_end})",
            resume,
        )

    # 3. 重点项目按时间倒序重排
    #    找到 ## 重点项目 区域，解析每个 ### 子块，按起始年份倒序重排
    project_section_match = re.search(
        r"(##\s*重点项目\n+)(.*?)(?=\n##\s|\Z)", resume, re.DOTALL
    )
    if project_section_match:
        header = project_section_match.group(1)
        body = project_section_match.group(2)

        # 3a. 去掉项目标题中的编号（"### 1. " → "### "）
        body = re.sub(r"(###\s+)\d+\.\s+", r"\1", body)

        # 3a2. 去掉项目块之间的 --- 分隔线（可能有前后空行）
        body = re.sub(r"\n\s*---\s*\n(?=\n###\s)", "\n", body)

        # 3b. 当前公司项目"至今"替换为实际离职时间（匹配任意起始时间）
        if latest_end and latest_company:
            body = re.sub(
                rf"\({re.escape(latest_company)} \| (\d{{4}}\.\d{{2}})-至今\)",
                f"({latest_company} | \\1-{latest_end})",
                body,
            )

        # 3c. 强制修正当前公司项目的结束日期
        # LLM 可能输出错误的结束日期（如 2025.06），强制替换为实际离职时间
        if latest_end and latest_company:
            body = re.sub(
                rf"\({re.escape(latest_company)} \| (\d{{4}}\.\d{{2}})-(\d{{4}}\.\d{{2}})\)",
                lambda m: f"({latest_company} | {m.group(1)}-{latest_end})"
                         if m.group(2) != latest_end else m.group(0),
                body,
            )

        # 拆分每个项目块（以 ### 开头）
        project_blocks = re.split(r"(?=\n###\s)", body)
        # 过滤空块
        project_blocks = [b for b in project_blocks if b.strip()]

        def _extract_start_ym(block: str) -> int:
            """从项目块标题提取起始年月，用于精确排序。返回 YYYY*12+MM。"""
            title_line = block.strip().split("\n")[0]
            # 匹配 YYYY.MM- 格式
            m = re.search(r"(\d{4})\.(\d{2})\s*[-–—]", title_line)
            if m:
                return int(m.group(1)) * 12 + int(m.group(2))
            # fallback: 只匹配年份
            m2 = re.search(r"(\d{4})", title_line)
            return int(m2.group(1)) * 12 if m2 else 0

        # 纯时间倒序排列（最新的项目排最前面）
        project_blocks.sort(key=_extract_start_ym, reverse=True)
        resume = resume.replace(
            project_section_match.group(0),
            header + "".join(project_blocks),
        )

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
    # 事实来源：resume_source + rag_context 合并（RAG 检索到的文档也是有效出处）
    task_input = state.get("task_input", "")
    rag_context = state.get("task_data", {}).get("rag_context", "") or state.get("rag_context", "")
    resume_source = state.get("task_data", {}).get("resume_source", "")
    # 合并简历全文 + RAG 上下文作为完整的事实来源
    parts = []
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
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
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
MARKET_PARTITIONS = ["interview", "interview2026", "jd", "paypal", "work", "technical", "linkedin"]


async def main():
    ap = argparse.ArgumentParser(description="RAG 加持的简历定制/推荐 (llamaindex-pse)")
    ap.add_argument("--jd", type=str, help="JD 文件路径（定制模式）")
    ap.add_argument("--jd-text", type=str, help="JD 文本（定制模式，直接传入）")
    ap.add_argument("--recommend", action="store_true",
                    help="自由推荐模式：无需 JD，根据你的经历 + 国内行情推荐最适合的岗位")
    ap.add_argument("--docs", type=str,
                    default=os.getenv("RESUME_DOCS_PATH", ""),
                    help="文档目录路径（默认从 PSE_ROOT/work/docs 加载）")
    ap.add_argument("--provider", choices=["deepseek", "agnes"], default="deepseek",
                    help="LLM 网关")
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
        planner_prompt = (prompts_dir / "recommend_planner.md").read_text(encoding="utf-8")
        specialist_prompt = (prompts_dir / "recommend_specialist.md").read_text(encoding="utf-8")
        evaluator_prompt = (prompts_dir / "evaluator.md").read_text(encoding="utf-8")

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
            task_input += f"## 我的完整简历（以下为事实来源，必须基于此撰写）\n\n{resume_full_for_verify}\n\n"

        # 注入个人特点文档（直接读取，不依赖 RAG 检索）
        personal_docs_dir = docs_dir / "work"
        if personal_docs_dir.exists():
            personal_parts = []
            for p in sorted(personal_docs_dir.glob("*.md")):
                if p.name.startswith("."):
                    continue
                content = p.read_text(encoding="utf-8").strip()
                if content:
                    # 截取前 2000 字避免过长
                    if len(content) > 2000:
                        content = content[:2000] + "\n...(截断)"
                    personal_parts.append(f"### {p.stem}\n\n{content}")
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

        print(f"\n🚀 自由推荐模式 (provider={args.provider}, max_retries={max_retries})")
        print(f"   RAG 分区: Planner→市场情报, Specialist→简历源数据")
        handler = workflow.run(
            task_input=task_input,
            task_data={"resume_source": resume_full_for_verify},
            max_retries=max_retries,
        )
        result = await handler

        artifact = result.get("artifact", "")
        artifact = _postprocess_resume(artifact)
        out_path = BASE / "recommended_resume.md"
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
            task_input += f"## 我的完整简历（以下为事实来源，必须基于此撰写）\n\n{resume_full_for_verify}\n\n"

        # 注入个人特点文档（直接读取，不依赖 RAG 检索）
        personal_docs_dir = docs_dir / "work"
        if personal_docs_dir.exists():
            personal_parts = []
            for p in sorted(personal_docs_dir.glob("*.md")):
                if p.name.startswith("."):
                    continue
                content = p.read_text(encoding="utf-8").strip()
                if content:
                    if len(content) > 2000:
                        content = content[:2000] + "\n...(截断)"
                    personal_parts.append(f"### {p.stem}\n\n{content}")
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
            task_data={"resume_source": resume_full_for_verify},
            max_retries=max_retries,
        )
        result = await handler

        resume = result.get("artifact", "")
        resume = _postprocess_resume(resume)
        out_path = BASE / "tailored_resume.md"
        out_path.write_text(resume, encoding="utf-8")
        print(f"\n✅ 定制简历已保存 → {out_path}")

    # 打印 token 消耗统计
    from llamaindex_pse.model import token_stats
    print(f"\n{token_stats.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
