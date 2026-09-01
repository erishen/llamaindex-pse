"""LlamaIndex PSE — hot-news 任务：RAG 加持的热点营销内容生成。

流水线：
    热点主题 / 新闻目录 → RAG 索引（新闻为 grounding 源）→
    Planner(选题) → Specialist(写作) → Evaluator(首轮 LLM 评审) +
    verify_compliance(每轮确定性合规核查：违禁词/格式/AI 标注/事实对照) →
    Fix(修正) 循环 → 成稿

用法:
    # 标准：提供抓取好的新闻目录（RAG grounding 最强，事实可对照）
    python run.py --topic "AI 新规落地" --news-dir /path/to/news \\
                  --platform xiaohongshu --category tech_ai

    # 降级：仅给主题，无新闻源（verify 会警告，无事实对照）
    python run.py --topic "AI 新规落地" --platform douyin --provider agnes

合规：确定性核查见 compliance.py（按品类+平台配置）。
"""

import argparse
import asyncio
import hashlib
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

# LlamaIndex Workflow 内部创建 event loop，需要 nest_asyncio 允许嵌套
import nest_asyncio

# 同目录模块
from compliance import (
    CATEGORIES,
    EXCLUDED_TOPICS,
    PLATFORMS,
    load_banned_words,
    make_compliance_verifier,
)

# LlamaIndex Workflow 内部创建 event loop，允许嵌套（须在所有导入之后执行）
nest_asyncio.apply()

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent.parent  # llamaindex-pse/

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

sys.path.insert(0, str(PROJECT_ROOT / "src"))


PLATFORM_LABELS = {
    "xiaohongshu": "小红书",
    "douyin": "抖音",
    "zhihu": "知乎",
    "toutiao": "今日头条",
}
CATEGORY_LABELS = {
    "tech_ai": "技术/AI 个人 IP",
    "beauty": "美妆",
    "food": "食品",
    "education": "教育",
    "finance": "金融",
    "medical": "医疗",
    "ecommerce": "电商带货",
}

# 自动选题偏好：命中有 AI/科技属性即视为 AI 相关选题（tech_ai 品类要求）
AI_TOPIC_KEYWORDS = [
    "AI", "人工智能", "大模型", "LLM", "GPT", "DeepSeek", "智谱", "GLM",
    "算力", "芯片", "半导体", "机器人", "自动驾驶", "量子", "算法",
    "华为", "鸿蒙", "苹果", "特斯拉", "小米", "英伟达", "nvidia",
    "折叠", "端侧", "开源", "数据", "云", "新能源", "智能", "科技",
    "互联网", "元宇宙", "AR", "VR", "XR", "5G", "6G", "卫星", "航天",
    "具身", "Agent", "智能体", "模型",
]

# 拉黑选题统一在 compliance.EXCLUDED_TOPICS（与 fetch_news --exclude 默认值同源，改一处即可）；
# 自动选题与媒体兜底一律跳过命中项。
def _is_excluded(title: str) -> bool:
    """标题是否命中拉黑词（大小写不敏感，忽略空格）。命中则自动选题 / 兜底一律跳过。"""
    t = title.lower().replace(" ", "")
    return any(k.lower().replace(" ", "") in t for k in EXCLUDED_TOPICS)


def _is_ai_topic(title: str) -> bool:
    """标题是否命中 AI/科技关键词（大小写不敏感）。"""
    t = title.lower()
    return any(k.lower() in t for k in AI_TOPIC_KEYWORDS)


# AI 专属新闻源（落盘子目录名，须与 fetch_news.py 的 AI_SOURCES 对齐）。
# 这些源天然 AI/科技向，自动选题兜底与实时选题清单优先采用。
AI_NEWS_SOURCES = ("qbitai", "infoq")

# 人设注入使用的个人背景分区（相对 work/docs），与 resume-tailor 的
# RESUME/MARKET_PARTITIONS 对齐：简历 + 工作经历 + 技术背景；不含 jobs 市场情报。
PERSONA_SUBDIRS = (
    "resume2026ppcnlean-v2",
    "resume-fragments",
    "technical",
    "paypal",
    "work",
    "resume-story",
)

# 非文章的静态/工具页 slug（与 wordpress-tools/gen_keywords_index.py 的 SKIP_SLUGS 对齐）：
# 首页/关于/简历/隐私等页面不是可引流的原创文章，索引时跳过。
SKIP_ARTICLE_SLUGS = {
    "home", "about", "resume", "resume-pe",
    "privacy-policy", "tags", "architecture", "keywords", "links",
}

# 用户确认不参与引流的文章文件名（中文/乱码 slug，可能是草稿或测试文；文件保留，仅不进索引）。
EXCLUDE_ARTICLE_FILES = {
    "china-developer-survey-2024",
    "draw-angle-with-triangle-ruler",
    "electron-react-desktop-app",
    "finding-opportunities-beyond-zhangyiming",
    "math-fraction-arithmetic",
    "mickey-mouse",
    "nginx-configuration-guide",
    "node-typescript-express-setup",
    "nowcoder-prime-factor",
    "nowcoder-string-sort",
    "roman-empire",
    "styled-components-clock-demo",
}

# 真实发布文章（zh URL）的权威来源：crewai-pse 的发布登记表。
# 文件名与该 JSON 中项目名不一致的少数文章，在此显式覆盖（文件名 -> zh URL）。
ARTICLE_PUBLISHED_OVERRIDES = {
    "ai-analyze-design-decisions": "https://erishen.cn/ai-analyze-five-design-decisions-cn/",
    "ai-tool-server-lobster-architecture": "https://erishen.cn/building-ai-tool-server-lobster-architecture-cn/",
    "monorepo-nextjs-fastapi-knowledge-base": "https://erishen.cn/building-fullstack-knowledge-base-platform-cn/",
    "production-react-ssr-framework": "https://erishen.cn/building-production-react-ssr-framework-cn/",
    "rag-smart-chat-app": "https://erishen.cn/building-rag-smart-chat-app-cn/",
    "shadcn-bun-component-library": "https://erishen.cn/building-shadcn-bun-component-library-cn/",
}


def _load_published_article_urls() -> dict[str, str]:
    """从 crewai-pse 发布登记表加载「zh 文章文件名 -> 真实 URL」映射。

    发布登记表是真实已发布文章的唯一事实来源（含 wp_id/中英双链接），
    避免从文件名/frontmatter 猜 slug 拼出无效或未发布链接。文件名归一化
    （去 -zh、下划线转连字符）后与登记表项目名匹配；特殊 slug 文章用显式覆盖。
    无法读取登记表时降级为空映射（不产生引流链接）。
    """
    urls: dict[str, str] = {}
    registry_path = (
        PROJECT_ROOT.parent.parent
        / "frameworks" / "crewai-pse" / "tasks" / "project-articles" / "projects-published.json"
    )
    try:
        import json as _json

        data = _json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"   ⚠️ 无法读取发布登记表 {registry_path}: {e}，引流素材降级为空")
        return {}
    for _name, info in data.items():
        published = (info or {}).get("published") or {}
        zh_link = (published.get("zh") or {}).get("link", "")
        if not zh_link:
            continue
        urls[_name] = zh_link
    return urls


def _norm_slug(name: str) -> str:
    """归一化文章名/项目名：小写、下划线/空格/连字符统一为连字符。"""
    return re.sub(r"[-_ ]+", "-", name.lower().strip("-"))


def _resolve_article_url(stem: str, published_urls: dict[str, str]) -> str:
    """按文件名解析真实发布 URL：显式覆盖优先，其次归一化匹配发布登记表。

    未在登记表中确认发布（或文件名无法匹配）时返回空串，调用方据此跳过引流索引。
    """
    if stem in ARTICLE_PUBLISHED_OVERRIDES:
        return ARTICLE_PUBLISHED_OVERRIDES[stem]
    return published_urls.get(_norm_slug(re.sub(r"-zh$", "", stem)), "")


def _article_is_indexable(stem: str, published_urls: dict[str, str]) -> bool:
    """文章是否进引流索引：仅发布登记表确认已发布（有真实 URL）且不在黑名单。"""
    return bool(_resolve_article_url(stem, published_urls)) and stem not in EXCLUDE_ARTICLE_FILES


def _dedup_article_context(context: str) -> str:
    """对引流检索上下文按「文章标题」去重，每篇文章只保留首个 chunk。

    同一篇文章可能被切分命中多个 chunk（引流头重复），注入前合并避免正文重复引流同一篇。
    """
    if not context:
        return context
    # 按 _retrieve_context 的块分隔符切分（每条以 [n] (score=...) 开头）
    parts = re.split(r"\n(?=\[\d+\] )", context)
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        m = re.search(r"我的原创文章《(.+?)》：", part)
        if m and m.group(1) in seen:
            continue
        if m:
            seen.add(m.group(1))
        kept.append(part)
    return "\n\n".join(kept)


def _pick_news_headline(news_dir: Path) -> tuple[str, str]:
    """AI 选题兜底：优先 AI 专属源(qbitai/infoq)中**最新发布**的一条，其次 kr36/sspai 最新一条。

    按 published_at（缺失时回退为早期时间戳）从新到旧取，避免反复命中同一篇旧闻；
    来源优先级高于新旧（qbitai/infoq 永远先于 kr36/sspai）。
    返回 (title, source_url)；无可用标题时返回 ("", "")。source_url 落入成稿 frontmatter，
    供发布端「内容来源」标注追溯（非热搜兜底时正文不得写「登上热搜」）。
    """
    best: tuple[tuple[int, float], str, str] | None = None
    for sub in (*AI_NEWS_SOURCES, "kr36", "sspai"):
        d = news_dir / sub
        if not d.is_dir():
            continue
        for p in d.glob("*.md"):
            if p.name.startswith("."):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            m = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
            if not (m and m.group(1).strip()):
                continue
            title = m.group(1).strip()
            if _is_excluded(title):
                continue
            pub = re.search(r"^published_at:\s*(.+)$", text, re.MULTILINE)
            ts = 0.0
            if pub and pub.group(1).strip():
                try:
                    ts = datetime.strptime(pub.group(1).strip()[:16], "%Y-%m-%d %H:%M").timestamp()
                except ValueError:
                    ts = 0.0
            um = re.search(r"^url:\s*(.+)$", text, re.MULTILINE)
            url = (um.group(1).strip() if um and um.group(1).strip() else "").strip()
            prio = 0 if sub in AI_NEWS_SOURCES else 1
            key = (prio, -ts)  # 越小越优：AI 源优先；同源里发布时间越新越优
            if best is None or key < best[0]:
                best = (key, title, url)
    return (best[1], best[2]) if best else ("", "")


def _index_fingerprint(
    src_dir: Path,
    skip_dirs: tuple[str, ...] = (),
    suffixes: tuple[str, ...] = (".md", ".txt", ".pdf"),
    subdirs: tuple[str, ...] = (),
) -> str:
    """索引指纹：可索引文件集合（路径 + mtime）的哈希。

    目录新增/删除/更新文件时指纹变化，驱动 RAG 索引自动重建，
    避免缓存索引遗漏新抓取的文件（如 qbitai/infoq、个人人设库更新）。
    subdirs：仅索引指定子目录（相对于 src_dir）；空则索引整个目录。
    """
    parts = []
    if src_dir.exists():
        roots = [src_dir / s for s in subdirs] if subdirs else [src_dir]
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if (
                    p.is_file()
                    and not p.name.startswith(".")
                    and p.suffix in suffixes
                    and (not skip_dirs or p.parent.name not in skip_dirs)
                ):
                    # 毫秒级 mtime：同秒内多次写入也能触发指纹变化（避免缓存漏更新）
                    parts.append(f"{p}:{int(p.stat().st_mtime_ns // 1_000_000)}")
    return hashlib.md5("\n".join(sorted(parts)).encode("utf-8")).hexdigest()


def _build_news_index(
    news_dir: Path, index_cache_dir: Path, top_k: int, rebuild: bool = False
):
    """为新闻目录构建 RAG 索引（简化版，无隐私脱敏）。返回 retriever 或 None。"""
    # weibo 热榜仅作选题清单（见 _collect_titles），不进 grounding 索引，
    # 避免无正文的标题被当作事实源检索命中
    return _build_index(
        news_dir,
        index_cache_dir,
        top_k,
        rebuild,
        name="新闻",
        skip_dirs=("weibo",),
    )


def _build_index(
    src_dir: Path,
    index_cache_dir: Path,
    top_k: int,
    rebuild: bool = False,
    name: str = "文档",
    skip_dirs: tuple[str, ...] = (),
    suffixes: tuple[str, ...] = (".md", ".txt", ".pdf"),
    subdirs: tuple[str, ...] = (),
    headline_fn: Callable[[str], str] | None = None,
    file_filter: Callable[[str], bool] | None = None,
):
    """为任意目录构建 RAG 索引（新闻 / 个人人设库 / 原创文章库复用）。

    skip_dirs：排除的子目录名（如新闻目录的 weibo 纯标题）；suffixes：参与索引的文件后缀。
    subdirs：仅索引指定子目录（相对于 src_dir，可多分区，如 work/docs 下的简历/工作经历分区）；
             空则索引整个目录。headline_fn：给定文件绝对路径，返回要在该文档正文前注入的
             引导头（如原创文章的标题+URL），用于检索命中时携带「延伸阅读/引流」信息。
    file_filter：给定文件绝对路径，返回 False 则跳过该文件（如无有效 URL 的未发布草稿）。
    返回 LlamaIndex retriever 或 None。按指纹缓存，文件变化自动重建。
    """
    from llama_index.core import (
        Document,
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
        load_index_from_storage,
    )
    from llama_index.core.node_parser import SentenceSplitter

    all_files = []
    if src_dir.exists():
        roots = [src_dir / s for s in subdirs] if subdirs else [src_dir]
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if (
                    p.is_file()
                    and not p.name.startswith(".")
                    and p.suffix in suffixes
                ):
                    if skip_dirs and p.parent.name in skip_dirs:
                        continue
                    if file_filter and not file_filter(str(p)):
                        continue
                    all_files.append(str(p))
    if not all_files:
        print(f"   ⚠️ {name}目录无可索引文件，RAG 缺失")
        return None

    print(f"📚 构建{name}索引: {len(all_files)} 个文件{subdirs and '（分区: %s）' % '+'.join(subdirs) or ''}")

    fp_path = index_cache_dir / ".index_fp"
    cur_fp = _index_fingerprint(src_dir, skip_dirs, suffixes, subdirs)

    if not rebuild and index_cache_dir.exists():
        try:
            cached_fp = fp_path.read_text(encoding="utf-8") if fp_path.exists() else ""
            if cached_fp == cur_fp:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(index_cache_dir)
                )
                index = load_index_from_storage(storage_context)
                retriever = index.as_retriever(similarity_top_k=top_k)
                print(f"   索引缓存命中，retriever top_k={top_k}")
                return retriever
            else:
                print(f"   🔄 {name}文件集变化，索引自动重建")
        except Exception as e:
            print(f"   ⚠️ 缓存加载失败: {e}，将重新构建")

    try:
        documents = SimpleDirectoryReader(input_files=all_files).load_data()
        if not documents:
            print("   ⚠️ 加载 0 个文档片段")
            return None
        if headline_fn:
            # 在正文前注入「标题+URL」引流引导头，使检索命中时携带作者原创文章信息；
            # Document.text 为只读属性，需重建 Document 对象（元数据一并保留）
            documents = [
                Document(
                    text=(f"{headline_fn(str(d.metadata.get('file_path', '')))}\n\n{d.text}"),
                    metadata=dict(d.metadata) if getattr(d, "metadata", None) else None,
                )
                for d in documents
            ]
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        if headline_fn:
            # 引流头必须出现在每个检索 chunk 里：先切块，再给每个 node 前插引流头，
            # 避免命中正文中部 chunk 时 URL 丢失（仅放文档头只进第一个 chunk）。
            nodes = splitter.get_nodes_from_documents(documents)
            for node in nodes:
                src = str((node.metadata or {}).get("file_path", ""))
                node.text = f"{headline_fn(src)}\n\n{node.text}"
            from llama_index.core import VectorStoreIndex as _VSI

            index = _VSI(nodes)
        else:
            index = VectorStoreIndex.from_documents(
                documents, transformations=[splitter]
            )
        retriever = index.as_retriever(similarity_top_k=top_k)
        print(f"   索引构建完成: {len(documents)} 个文档片段, top_k={top_k}")
        index.storage_context.persist(persist_dir=str(index_cache_dir))
        fp_path.write_text(cur_fp, encoding="utf-8")
        print(f"   索引已缓存 → {index_cache_dir}")
        return retriever
    except Exception as e:
        print(f"❌ 构建{name}索引失败: {e}")
        return None


def _parse_article_frontmatter(filepath: str) -> tuple[str, str, str]:
    """解析原创文章 frontmatter，返回 (title, slug, url)。

    仅当 slug 合法（小写英文数字连字符）时生成 URL；缺 slug / 中文 slug /
    工具页等未发布草稿返回空 url，调用方据此跳过该文件，避免无效引流链接。
    """
    title = ""
    slug = ""
    url = ""
    try:
        text = Path(filepath).read_text(encoding="utf-8")
    except Exception:
        text = ""
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if m:
        fm = m.group(1)
        for line in fm.splitlines():
            line = line.strip()
            if line.startswith("title:"):
                title = line.split(":", 1)[1].strip().strip("\"' ")
            elif line.startswith("slug:"):
                slug = line.split(":", 1)[1].strip().strip("\"' ")
    if not title:
        title = Path(filepath).stem
    if re.fullmatch(r"[a-z0-9][a-z0-9\-]*", slug or ""):
        url = f"https://erishen.cn/{slug}/"
    else:
        slug = ""
        url = ""
    return title, slug, url


def _build_articles_index(
    articles_dir: Path,
    index_cache_dir: Path,
    top_k: int,
    rebuild: bool = False,
    langs: tuple[str, ...] = ("zh",),
):
    """为个人原创文章构建 RAG 索引，检索命中自带标题+URL 引流头。

    以 crewai-pse 的发布登记表（projects-published.json）为唯一事实来源：
    文件名经归一化（去 -zh、下划线转连字符）匹配登记表项目名，特殊 slug 用显式覆盖；
    只有登记表确认已发布的文章才进引流索引，URL 一律用登记表中的真实链接，
    不再从 frontmatter 猜 slug（避免无效/未发布链接）。langs 默认仅 zh。
    """
    # 文件名 -> 真实 URL 映射（基于发布登记表 + 显式覆盖）
    published_urls = _load_published_article_urls()
    for _file, _url in ARTICLE_PUBLISHED_OVERRIDES.items():
        published_urls[_file] = _url

    def file_url(filepath: str) -> str:
        return _resolve_article_url(Path(filepath).stem, published_urls)

    def headline(filepath: str) -> str:
        url = file_url(filepath)
        title, _slug, _u = _parse_article_frontmatter(filepath)
        return f"📌 我的原创文章《{title}》：{url}（可作延伸阅读 / 站内引流）"

    def has_valid_url(filepath: str) -> bool:
        # 仅索引发布登记表确认已发布的文章（能拿到真实 URL）；
        # 同时排除用户确认不引流的文件名黑名单。
        return _article_is_indexable(Path(filepath).stem, published_urls)

    return _build_index(
        articles_dir,
        index_cache_dir,
        top_k,
        rebuild,
        name="原创文章",
        suffixes=(".md",),
        subdirs=langs,
        headline_fn=headline,
        file_filter=has_valid_url,
    )


def _read_news_files(news_dir: Path) -> dict[Path, str]:
    """读取 news 目录下所有 md/txt 到 dict（单次遍历读盘，供新闻语料与选题 grounding 复用）。"""
    if not news_dir.exists():
        return {}
    files: dict[Path, str] = {}
    for p in sorted(news_dir.rglob("*")):
        if (
            p.is_file()
            and not p.name.startswith(".")
            and p.suffix in (".md", ".txt")
        ):
            try:
                files[p] = p.read_text(encoding="utf-8")
            except Exception:
                continue
    return files


def _read_news_corpus(
    news_dir: Path,
    skip_subdirs: tuple[str, ...] = (),
    files: dict[Path, str] | None = None,
) -> str:
    """合并新闻目录下所有 md/txt 为事实核查语料（供 verify_fn 事实对照）。

    skip_subdirs 用于排除仅作选题清单的子目录（如 weibo 纯标题，正文语料不含）。
    files：传入 _read_news_files 的复用结果，避免与选题 grounding 重复读盘。
    """
    if files is None:
        files = _read_news_files(news_dir)
    chunks = [
        text
        for p, text in files.items()
        if not skip_subdirs or p.parent.name not in skip_subdirs
    ]
    corpus = "\n\n".join(chunks)
    print(f"   📚 新闻核查语料: {len(chunks)} 个文件, {len(corpus)} 字")
    return corpus


def _norm(text: str) -> str:
    """归一化文本：去空白、字母小写，供子串/令牌匹配。"""
    return re.sub(r"\s", "", text or "").lower()


def _collect_titles(news_dir: Path, subdir: str) -> list[str]:
    """收集指定子目录（如 weibo）新闻文件 frontmatter 的 title，作为实时选题清单。"""
    titles: list[str] = []
    d = news_dir / subdir
    if not d.is_dir():
        return titles
    for p in sorted(d.glob("*.md")):
        if p.name.startswith("."):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        m = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
        if m and m.group(1).strip():
            titles.append(m.group(1).strip())
    return titles


def _list_ai_news_candidates(news_dir: Path, top_n: int = 20) -> list[tuple[str, str, str]]:
    """收集 AI 专属源（qbitai/infoq）新闻标题作为选题候选，按发布时间从新到旧。

    与微博热搜候选互补：微博热榜未必有 AI 话题，但 qbitai/infoq 天然 AI/科技向，
    供 tech_ai 品类选材时优先看这里。返回 (title, source, published_at)。
    """
    rows: list[tuple[float, str, str, str]] = []  # (published_ts, source, title, published_at)
    for sub in AI_NEWS_SOURCES:
        d = news_dir / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            if p.name.startswith("."):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            m = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
            if not (m and m.group(1).strip()):
                continue
            title = m.group(1).strip()
            if _is_excluded(title):
                continue
            pub = re.search(r"^published_at:\s*(.+)$", text, re.MULTILINE)
            published_at = pub.group(1).strip() if pub and pub.group(1).strip() else ""
            ts = 0.0
            if published_at:
                try:
                    ts = datetime.strptime(published_at[:16], "%Y-%m-%d %H:%M").timestamp()
                except ValueError:
                    ts = 0.0
            rows.append((ts, sub, title, published_at))
    # 同一篇文章可能被抓成多个文件（fetch_news.py 编号前缀重复），按标题去重，
    # 保留发布时间最新的一条，避免候选清单里同一话题反复出现。
    rows.sort(key=lambda x: (x[2], x[0]), reverse=True)
    deduped: list[tuple[float, str, str, str]] = []
    seen_titles: set[str] = set()
    for row in rows:
        if row[2] in seen_titles:
            continue
        seen_titles.add(row[2])
        deduped.append(row)
    deduped.sort(key=lambda x: x[0], reverse=True)
    return [(t, s, p) for (_ts, s, t, p) in deduped[:top_n]]


# 话题在新闻语料中「有据」的最低命中字符量：至少能说明语料能覆盖该话题的实体
GROUNDED_MIN_CHARS = 4


def _topic_grounded(title: str, corpus_norm: str) -> bool:
    """热点主题是否在新闻语料中有原文支撑。

    判据（归一化后）：
      - 全串精确出现在语料中 → 有据；
      - 否则按「分隔段 + AI 关键词」拆令牌，令牌命中语料的字符长度占主题字符比
        ≥ max(4, 40%) 即视为有据。只命中通用词（如「鸿蒙」之于音乐适配话题）不达标。
    """
    if not corpus_norm or not title:
        return False
    t = _norm(title)
    if t in corpus_norm:
        return True
    segments = [
        s
        for s in re.split(r"[\s·、/,+：:&]", title.lower())
        if len(s.strip()) >= 2
    ]
    kws = [k.lower() for k in AI_TOPIC_KEYWORDS if k.lower() in title.lower()]
    tokens = set(segments + kws)
    matched = sum(len(tk) for tk in tokens if _norm(tk) in corpus_norm)
    need = max(GROUNDED_MIN_CHARS, int(len(t) * 0.4))
    return matched >= need


def _pick_hot_topic(
    news_dir: Path,
    top_n: int = 1,
    ai_only: bool = True,
    corpus_norm: str = "",
    selection: str = "random",
) -> list[tuple[str, int, str, str]]:
    """从微博热搜落盘文件按 hot 热度降序取 Top-N。

    返回 (title, hot, source, source_url)；source ∈ {weibo, news}。
    ai_only=True 时优先取命中 AI/科技关键词的热搜；在此基础上优先「语料有据」的热搜
    （corpus_norm 来自非 weibo 新闻原文，weibo 纯标题不构成事实依据），
    即在有原文支撑的最高热度热搜间选题，避免选中语料无从谈起的冷话题。
    selection="top" 取有据候选人里热度最高的；"random" 在全部有据候选人里随机
    （避免同一批新闻快照反复命中同一选题）。无有据候选时回退到新闻标题。
    """
    d = news_dir / "weibo"
    if not d.is_dir():
        return []
    items: list[tuple[str, int]] = []
    for p in sorted(d.glob("*.md")):
        if p.name.startswith("."):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        tm = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
        if not tm:
            continue
        hm = re.search(r"^hot:\s*(\d+)$", text, re.MULTILINE)
        hot = int(hm.group(1)) if hm else 0
        items.append((tm.group(1).strip(), hot))
    items.sort(key=lambda x: x[1], reverse=True)
    if ai_only:
        ai_items = [
            (t, h, "weibo", "")
            for (t, h) in items
            if _is_ai_topic(t) and not _is_excluded(t)
        ]
        if ai_items:
            grounded_ai = [
                it for it in ai_items if _topic_grounded(it[0], corpus_norm)
            ]
            if grounded_ai:
                if selection == "random":
                    pool = list(grounded_ai)
                    random.shuffle(pool)
                    return pool[:top_n]
                return grounded_ai[:top_n]
        # 无 AI 热搜、或 AI 热搜均无据：回退到新闻标题（天然有据、AI 向）
        news_topic, news_url = _pick_news_headline(news_dir)
        if news_topic:
            return [(news_topic, 0, "news", news_url)]
        return ai_items[:top_n]
    return [(t, h, "weibo", "") for (t, h) in items[:top_n]]


def _postprocess(artifact: str) -> str:
    """确定性后处理（占位）。

    AI 标注「AI 辅助创作」不再写入草稿正文——按 2026 平台新规，由发布端
    浏览器自动化层在发布时自动勾选（见 README「发布端」章节），故此处透传。
    """
    return artifact


def _render_frontmatter(args, generated_at: str, actual_topic: str = "", source_url: str = "") -> str:
    """成稿 YAML frontmatter：记录选题/平台/品类/模型/生成时间与来源链接。

    actual_topic 非空（成稿实际标题）时以其为准，避免「热搜话题」与正文选题错位；
    输入的热搜话题记为 heat_topic，仅在两者不同时写入，供追溯。
    source_url 仅当自动选题来自新闻标题兜底（或 --source-url 显式给出）时写入，
    供发布端「内容来源」标注使用。
    """
    topic = (actual_topic or args.topic).strip().replace('"', "'")
    heat = args.topic.strip().replace('"', "'")
    lines = [
        "---",
        f'topic: "{topic}"',
        f"platform: {args.platform}",
        f"category: {args.category}",
        f"provider: {args.provider}",
        f'generated_at: "{generated_at}"',
    ]
    if source_url:
        src = source_url.strip().replace('"', "'")
        lines.append(f"source_url: {src}")
    if heat != topic:
        lines.append(f'heat_topic: "{heat}"')
    lines.append("---\n")
    return "\n".join(lines) + "\n"


async def main():
    ap = argparse.ArgumentParser(description="RAG 加持的热点营销内容生成 (llamaindex-pse)")
    ap.add_argument("--topic", type=str, default="", help="热点主题（留空则自动从微博热搜 Top1 选题）")
    ap.add_argument("--list-topics", action="store_true",
                    help="仅列出选题候选（微博热搜 + qbitai/infoq AI 科技源）后退出，不生成")
    ap.add_argument("--news-dir", type=str, default="", help="已抓取新闻目录（RAG grounding 源）")
    ap.add_argument(
        "--persona-dir",
        type=str,
        default="",
        help="个人背景资料目录（RAG 人设注入源）。缺省自动用 work/docs 的简历/工作/技术分区；显式指定则视为单目录",
    )
    ap.add_argument(
        "--persona-top-k",
        type=int,
        default=3,
        help="人设 RAG 检索返回的文档数（默认 3）",
    )
    ap.add_argument(
        "--articles-dir",
        type=str,
        default="",
        help="个人原创文章目录（zh/en 双语，RAG 引流素材源）。"
        "缺省自动用 personal/personal-site/wordpress-tools/articles",
    )
    ap.add_argument(
        "--articles-top-k",
        type=int,
        default=2,
        help="原创文章 RAG 检索返回数（用于延伸阅读/引流，默认 2）",
    )
    ap.add_argument(
        "--articles-langs",
        type=str,
        default="zh",
        help="原创文章索引语言子目录，逗号分隔（默认 zh；含英文旧文可传 zh,en）",
    )
    ap.add_argument(
        "--no-articles",
        action="store_true",
        help="禁用原创文章 RAG 引流注入（如给外部客户生成不带个人站链接）",
    )
    ap.add_argument(
        "--provider",
        choices=["deepseek", "agnes", "scnet-kimi", "scnet-minimax"],
        default="agnes",
        help="LLM 网关（默认 agnes）",
    )
    ap.add_argument("--platform", choices=PLATFORMS, default="xiaohongshu", help="目标发布平台")
    ap.add_argument("--category", choices=CATEGORIES, default="tech_ai", help="内容品类（决定违禁词表）")
    ap.add_argument("--top-k", type=int, default=8, help="RAG 检索 top-k")
    ap.add_argument(
        "--selection",
        choices=["top", "random"],
        default="random",
        help="自动选题策略：top=有据热搜中最高热度；random=有据热搜中随机，避免同快照重复选题（默认 random）",
    )
    ap.add_argument("--rebuild", action="store_true", help="强制重建索引")
    ap.add_argument("--out-dir", type=str, default="", help="成稿输出目录（默认 tasks/hot-news/articles）")
    ap.add_argument("--source-url", type=str, default="",
                    help="显式写入成稿 source_url frontmatter（缺省由新闻兜底选题带出）")
    args = ap.parse_args()

    platform_label = PLATFORM_LABELS.get(args.platform, args.platform)
    category_label = CATEGORY_LABELS.get(args.category, args.category)

    news_dir = Path(args.news_dir) if args.news_dir else None

    # 人设注入源：缺省用 workspace 下 work/docs 的个人简历分区（与 resume-tailor 的
    # RESUME/MARKET_PARTITIONS 对齐：简历 + 工作经历 + 技术背景，不含 jobs 市场情报）
    persona_dir = Path(args.persona_dir) if args.persona_dir else (
        PROJECT_ROOT.parent.parent / "work" / "docs"
    )
    # 显式指定 persona-dir 时视为单目录（只索引该目录本身）；缺省为 work/docs 多分区
    persona_subdirs = () if args.persona_dir else PERSONA_SUBDIRS
    if not persona_dir.exists():
        persona_dir = None
        if args.persona_dir:
            print(f"⚠️ 人设目录不存在: {args.persona_dir}，跳过人设注入")

    # 原创文章引流源：缺省用 workspace 下 personal-site 的 zh/en 双语文章库
    articles_dir = Path(args.articles_dir) if args.articles_dir else (
        PROJECT_ROOT.parent.parent / "personal" / "personal-site" / "wordpress-tools" / "articles"
    )
    # 索引语言子目录（默认仅 zh，中文稿件不混英文旧文）
    articles_langs = tuple(
        s.strip() for s in args.articles_langs.split(",") if s.strip()
    )
    if not articles_dir.exists():
        articles_dir = None
        if args.articles_dir:
            print(f"⚠️ 原创文章目录不存在: {args.articles_dir}，跳过引流素材注入")

    # 仅列候选模式：不加载 LLM 环境，直接打印微博热搜候选 + AI 科技源候选
    if args.list_topics:
        if not news_dir or not news_dir.exists():
            print("❌ --list-topics 需先抓取新闻（make hot-news-fetch）")
            return 1
        picks = _pick_hot_topic(news_dir, top_n=30, ai_only=False)
        ai_picks = _list_ai_news_candidates(news_dir, top_n=20)
        if not picks and not ai_picks:
            print("⚠️ 无候选（news 目录为空，请先 make hot-news-fetch）")
            return 0
        if picks:
            print("📋 微博热搜候选（按热度降序）：")
            for i, (t, h, _, _) in enumerate(picks, 1):
                print(f"  {i:2d}. {t}  (热度 {h})")
        if ai_picks:
            if picks:
                print()
            print("📋 AI 科技源候选（qbitai/infoq，按发布时间降序）：")
            for i, (t, src, published_at) in enumerate(ai_picks, 1):
                tag = f"[{src}] {published_at}" if published_at else f"[{src}]"
                print(f"  {i:2d}. {t}  ({tag})")
        return 0

    # 加载 LlamaIndex 运行环境
    try:
        from llamaindex_pse.config import settings
        from llamaindex_pse.model import create_embedding, create_llm
        from llamaindex_pse.workflow import build_workflow
    except Exception as e:
        print(f"❌ 无法加载 llamaindex 运行环境: {e}\n（请先 `uv sync`）")
        return 1

    llm = create_llm(args.provider)
    print(f"   LLM: {llm.model_name}")
    try:
        from llama_index.core import Settings
        Settings.embed_model = create_embedding()
        print(f"   Embedding: {settings.EMBEDDING_PROVIDER}/{settings.EMBEDDING_MODEL}")
    except RuntimeError as e:
        print(f"   ⚠️ {e}，将使用 LlamaIndex 默认 embedding")

    # ── RAG 索引（新闻为 grounding 源）──
    news_retriever = None
    news_corpus = ""
    grounded_corpus_norm = ""
    weibo_titles: list[str] = []
    if news_dir:
        if not news_dir.exists():
            print(f"❌ 新闻目录不存在: {news_dir}")
            return 1
        news_retriever = _build_news_index(
            news_dir, BASE / ".index_cache" / "news", args.top_k, args.rebuild
        )
        # 单遍读盘复用：事实对照语料（含 weibo）+ 选题 grounding（不含 weibo 纯标题）
        news_files = _read_news_files(news_dir)
        news_corpus = _read_news_corpus(news_dir, files=news_files)
        grounded_corpus_norm = _norm(
            _read_news_corpus(news_dir, skip_subdirs=("weibo",), files=news_files)
        )
        weibo_titles = _collect_titles(news_dir, "weibo")
        if weibo_titles:
            print(f"   🔥 实时选题清单（微博热搜）: {len(weibo_titles)} 条")
    else:
        print("   ⚠️ 未提供 --news-dir，纯 topic 降级生成（事实对照缺失）")

    # ── 个人人设 RAG（可选注入）：从简历/自我介绍检索与当前话题相关的个人背景，
    #    让文章不只是转述新闻，而是带上"技术人设"的第一人称观点 ──
    persona_retriever = None
    persona_context = ""
    if persona_dir:
        persona_retriever = _build_index(
            persona_dir,
            BASE / ".index_cache" / "persona",
            args.persona_top_k,
            args.rebuild,
            name="人设",
            suffixes=(".md", ".txt"),
            subdirs=persona_subdirs,
        )
        if persona_retriever:
            # 用最终选题做检索词：让注入的背景与文章主题对齐
            query = args.topic or "AI Agent 工程化 全栈 技术"
            from llamaindex_pse.workflow import _retrieve_context

            persona_context = await _retrieve_context(
                persona_retriever, query, args.persona_top_k
            )
            if persona_context:
                print(f"   🧑 人设注入: 检索到 {len(persona_context)} 字个人背景（top_k={args.persona_top_k}）")
            else:
                print("   ⚠️ 人设检索为空，跳过注入")
        else:
            print("   ⚠️ 人设目录无可索引文件，跳过注入")

    # ── 自动选题：优先选「语料有据」的 AI 热搜，其次回退新闻标题，再兜底热搜 ──
    topic_source = "manual"  # weibo(微博热搜) | news(新闻标题兜底) | manual(人工指定)
    article_source_url = args.source_url.strip() or ""
    if not args.topic:
        if news_dir:
            picks = _pick_hot_topic(
                news_dir,
                top_n=1,
                ai_only=True,
                corpus_norm=grounded_corpus_norm,
                selection=args.selection,
            )
            if picks:
                args.topic, hot, src, src_url = picks[0]
                topic_source = src
                if src == "weibo":
                    tag = f"微博热度(语料有据, {args.selection})"
                else:
                    tag = "新闻标题兜底(AI 选题)"
                    article_source_url = article_source_url or src_url
                print(f"🎯 自动选题：{args.topic}（{tag} {hot}）")
        if not args.topic:
            print("❌ 未提供 --topic 且无可用热榜候选（请先 make hot-news-fetch 或显式 --topic）")
            return 1

    # ── 个人原创文章 RAG（可选引流素材）：检索与最终主题最相关的作者旧文，
    #    命中后注入正文作「延伸阅读 / 站内引流」，把读者导流到 erishen.cn ──
    articles_context = ""
    if articles_dir and not args.no_articles:
        try:
            from llamaindex_pse.workflow import _retrieve_context
        except Exception:
            _retrieve_context = None
        if _retrieve_context:
            articles_retriever = _build_articles_index(
                articles_dir,
                BASE / ".index_cache" / "articles",
                args.articles_top_k,
                args.rebuild,
                langs=articles_langs,
            )
            if articles_retriever:
                articles_context = await _retrieve_context(
                    articles_retriever, args.topic, args.articles_top_k
                )
                # 同一篇可能命中多个 chunk，按文章标题去重，避免正文重复引流同一篇
                articles_context = _dedup_article_context(articles_context)
                if articles_context:
                    print(
                        "   🔗 引流素材注入: 检索到 "
                        f"{len(articles_context)} 字相关原创文章（top_k={args.articles_top_k}）"
                    )
                else:
                    print("   ⚠️ 原创文章检索为空，跳过引流注入")
            else:
                print("   ⚠️ 原创文章目录无可索引文件，跳过引流注入")

    # 合规核查闭包（按品类+平台）
    verify_fn = make_compliance_verifier(category=args.category, platform=args.platform)
    max_retries = settings.PSE_MAX_RETRIES or 3

    workflow = build_workflow(
        llm=llm,
        task="hot-news",
        verify_fn=verify_fn,
        use_planner=True,
        max_retries=max_retries,
        provider=args.provider,
        retriever=news_retriever,
        planner_retriever=news_retriever,
        rag_top_k=args.top_k,
    )

    source_label = {
        "weibo": "微博热搜（在下方实时热榜中，允许用「登上热搜/冲上热搜」表述）",
        "news": "新闻头条兜底（来自新闻源标题，**不是**实时热搜，**禁止**写「登上热搜/冲上热搜」等热搜表述）",
        "manual": "人工指定",
    }.get(topic_source, "人工指定")
    ban_words = load_banned_words(args.category)
    task_input = (
        f"请基于检索到的新闻原文，撰写一篇面向**{platform_label}**的"
        f"**{category_label}**向热点内容。\n\n"
        f"## 热点主题\n{args.topic}（来源：{source_label}）\n\n"
        f"## {category_label}品类违禁词（全文禁止出现，含标题/正文/标签）\n"
        f"{'、'.join(ban_words)}\n\n"
    )
    if persona_context:
        task_input += (
            "## 我的个人背景（RAG 注入的人设，仅用于主观观点/经历/口吻，**不是**新闻事实）\n"
            f"{persona_context}\n\n"
            "**人设要求**：请以第一人称「我」，结合上述个人背景里真实的技术经历、技术栈、行业判断，"
            "围绕热点主题写出**有技术人设的观点**（个人解读/踩坑经验/工程视角），而不是干巴巴转述新闻。"
            "但所有客观事实（数字/人名/机构/事件）仍必须来自检索到的新闻原文；个人背景只提供「观点和经历」，"
            "不得把简历里的公司/项目经历说成是新闻事件的一部分，也不得编造新闻中不存在的个人经历。\n\n"
        )
    if articles_context:
        task_input += (
            "## 我的相关原创文章（RAG 检索到的作者旧文，用于观点呼应 / 延伸阅读 / 站内引流）\n"
            f"{articles_context}\n\n"
            "**引流要求（可选，相关才引）**：若上述原创文章中确有与本文主题相关的，可在文末"
            "「延伸阅读」区列出 1–2 条，URL **原样照抄**任务给定的完整链接，不得改写/拼接/杜撰；"
            "正文可配一句第一人称呼应。"
            "**相关性优先于引流**：检索素材与主题契合才引；相关性弱则宁可不引，"
            "把精力放在内容质量上，绝不生硬硬塞。"
            "**格式严格**：每篇独立成行、**不加任何列表符号**（无 `-`/`*`/序号），"
            "格式为 `《标题》：https://erishen.cn/…/`（中文书名号 + 全角冒号 + 完整 URL）。\n\n"
        )
    if weibo_titles:
        topic_list = "\n".join(f"- {t}" for t in weibo_titles[:30])
        task_input += (
            "## 当前实时热榜（来自微博热搜，优先从中选择热点切入）\n"
            f"{topic_list}\n\n"
            "**选题要求**：内容选题应贴合上述实时热点；检索到的新闻原文（36氪/少数派等）"
            "仅作事实依据与背景素材，不要把其中陈旧的行业旧闻或已有概念包装成"
            "「刚刚提出的全新爆点」。\n"
            "**素材否决权**：若给定热点主题在检索新闻中找不到任何可核实的原文支撑，"
            "**禁止**写「暂无消息 / 存疑 / 证伪 / 请提供链接」类元叙述，"
            "而应从实时热榜中另选一个有新闻原文支撑的热点、或从检索新闻中换一个"
            "有事实依据的角度切入。\n\n"
        )
    if topic_source == "manual":
        task_input += (
            f"**人工指定主题铁律**：本主题「{args.topic}」由用户人工指定，且新闻语料中有其原文依据。"
            "正文与标题必须围绕该主题展开，**不得**用其他热榜话题（如微博热搜）替换或另起炉灶；"
            "实时热榜话题仅可作为承接句的背景素材，不能成为文章主角或标题主体。\n\n"
        )
    task_input += (
        "目标平台与品类已在系统提示中说明，请严格遵循其格式与合规要求。\n\n"
        "**铁律**：成稿中每个事实断言必须能在检索到的新闻原文（36氪/InfoQ/量子位等）中"
        "找到出处；若所选主题无原文支撑，宁可换一个受支撑的角度，也不要写「查无实据」"
        "类元叙述。"
    )

    # 供 evaluator / fix 核对的「真实数据」：除新闻语料外，把实时热榜清单与承接的
    # 热搜话题也带入，防止 LLM 评审把正文中引用的热榜话题误判为不可核实
    verify_scan = {}
    if weibo_titles:
        verify_scan["weibo_hot"] = weibo_titles[:30]
    verify_scan["heat_topic"] = args.topic
    verify_scan["heat_topic_source"] = topic_source
    # 品类违禁词全表：供 evaluator / fix 精确核对清理，避免违规词漏网反复
    verify_scan["banned_words"] = load_banned_words(args.category)

    print(
        f"\n🚀 hot-news (platform={args.platform}, category={args.category}, "
        f"provider={args.provider}, grounding={'news' if news_dir else 'none'})"
    )
    handler = workflow.run(
        task_input=task_input,
        task_data={
            "news_corpus": news_corpus,
            "scan_result": verify_scan,
            "articles_context": articles_context,
        },
        max_retries=max_retries,
    )
    result = await handler

    artifact = result.get("artifact", "")
    artifact = _postprocess(artifact)

    # 成稿实际标题：frontmatter topic 跟随正文，避免热搜话题与内容错位
    title_m = re.search(r"(?m)^#\s+(.+?)\s*$", artifact)
    actual_topic = title_m.group(1).strip() if title_m else ""

    # 成稿落盘：默认进 articles/ 子目录，文件名带生成时间戳，避免覆盖历史成稿
    out_dir = Path(args.out_dir) if args.out_dir else (BASE / "articles")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    artifact = _render_frontmatter(args, ts_human, actual_topic, article_source_url) + artifact

    out_path = out_dir / f"hot_news_{args.platform}_{args.provider}_{ts_file}.md"
    out_path.write_text(artifact, encoding="utf-8")
    print(f"\n✅ 热点内容已保存 → {out_path}")

    from llamaindex_pse.model import token_stats
    print(f"\n{token_stats.summary()}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
