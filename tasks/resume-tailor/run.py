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


def _verify_resume(resume: str, rag_context: str) -> tuple[list, list]:
    """程序化验证：简历中的关键声明应能在 RAG 上下文中找到出处。"""
    bad: list[str] = []
    ok: list[str] = []

    if not rag_context:
        bad.append("无 RAG 上下文可供核查，简历可能包含未验证内容")
        return bad, ok

    # 1. 检查简历中的年份/年限是否在上下文中出现
    year_pattern = r"\b(20\d{2})\b"
    resume_years = set(re.findall(year_pattern, resume))
    context_years = set(re.findall(year_pattern, rag_context))
    for y in resume_years:
        if y in context_years:
            ok.append(f"年份 {y} 在源文档中存在")
        else:
            bad.append(f"年份 {y} 未在源文档中找到，可能虚构")

    # 2. 检查量化数据（数字 + 单位）是否在上下文中出现
    quant_pattern = r"(\d+(?:\.\d+)?%|\d+(?:,\d{3})+(?:\+)?|\d+\+?\s*(?:人|万|倍|ms|GB|TB|个))"
    resume_quants = set(re.findall(quant_pattern, resume))
    context_quants = set(re.findall(quant_pattern, rag_context))
    for q in resume_quants:
        if q in context_quants:
            ok.append(f"量化数据 {q} 在源文档中存在")
        else:
            bad.append(f"量化数据 {q} 未在源文档中找到，可能编造")

    # 3. 基本长度检查
    if len(resume) < 200:
        bad.append("简历内容过短（< 200 字），可能不完整")
    elif len(resume) > 10000:
        bad.append("简历内容过长（> 10000 字），可能包含冗余")

    return bad, ok


def _verify_state(state: dict) -> tuple[list, list]:
    rag_context = state.get("task_data", {}).get("rag_context", "") or state.get("rag_context", "")
    return _verify_resume(state.get("artifact", ""), rag_context)


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
    docs_dir = args.docs
    if not docs_dir:
        pse_root = os.getenv("PSE_ROOT", str(Path.cwd()))
        docs_dir = str(Path(pse_root) / "work" / "docs")
    if not Path(docs_dir).exists():
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

    # 构建 RAG index（优先从本地持久化加载，避免重复 embedding）
    index_dir = BASE / ".index_cache"
    retriever = None

    if not args.rebuild and index_dir.exists():
        print(f"📚 从本地缓存加载索引: {index_dir}")
        try:
            from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

            storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
            index = load_index_from_storage(storage_context)
            retriever = index.as_retriever(similarity_top_k=args.top_k)
            print(f"   索引加载完成（缓存命中），retriever top_k={args.top_k}")
        except Exception as e:
            print(f"   ⚠️ 缓存加载失败: {e}，将重新构建")
            retriever = None

    if retriever is None:
        print(f"📚 加载文档: {docs_dir}")
        try:
            from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
            from llama_index.core.node_parser import SentenceSplitter

            documents = SimpleDirectoryReader(docs_dir, recursive=True).load_data()
            if not documents:
                print("❌ 未加载到任何文档")
                sys.exit(1)
            print(f"   加载了 {len(documents)} 个文档片段")

            splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
            retriever = index.as_retriever(similarity_top_k=args.top_k)
            print(f"   索引构建完成，retriever top_k={args.top_k}")

            index.storage_context.persist(persist_dir=str(index_dir))
            print(f"   索引已缓存 → {index_dir}")
        except Exception as e:
            print(f"❌ 构建索引失败: {e}")
            sys.exit(1)

    # 构建 PSE workflow
    max_retries = settings.PSE_MAX_RETRIES or 3

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
            retriever=retriever,
            rag_top_k=args.top_k,
        )
        # 覆盖提示词为推荐模式专用
        workflow._planner_prompt = planner_prompt
        workflow._specialist_prompt = specialist_prompt
        workflow._evaluator_prompt = evaluator_prompt

        # 推荐模式：多维度检索，覆盖技能/经历/项目
        task_input = (
            "请分析我的职业背景，结合当前国内招聘市场行情，"
            "推荐最适合我的岗位方向，并为排名第一的岗位定制简历。\n\n"
            "检索关键词：技术栈、项目经验、工作经历、架构设计、AI 工程化、团队管理"
        )

        print(f"\n🚀 自由推荐模式 (provider={args.provider}, max_retries={max_retries})")
        handler = workflow.run(
            task_input=task_input,
            max_retries=max_retries,
        )
        result = await handler

        artifact = result.get("artifact", "")
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
            retriever=retriever,
            rag_top_k=args.top_k,
        )

        task_input = (
            f"请根据以下 JD 定制简历：\n\n## 岗位描述\n{jd_text}"
        )

        print(f"\n🚀 JD 定制模式 (provider={args.provider}, max_retries={max_retries})")
        handler = workflow.run(
            task_input=task_input,
            max_retries=max_retries,
        )
        result = await handler

        resume = result.get("artifact", "")
        out_path = BASE / "tailored_resume.md"
        out_path.write_text(resume, encoding="utf-8")
        print(f"\n✅ 定制简历已保存 → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
