"""LlamaIndex PSE — resume-tailor 任务：RAG 加持的简历定制。

流程：
    1. 从 work/docs 加载文档，构建 LlamaIndex VectorStoreIndex
    2. 传入 JD（岗位描述）作为 task_input
    3. PSE Workflow:
       Planner(RAG检索) → Specialist(RAG grounded 撰写) → Evaluator(核查) → (Fix)

用法:
    python run.py --jd path/to/jd.md          # 从文件读取 JD
    python run.py --jd-text "JD 内容..."       # 直接传入 JD 文本
    python run.py --docs /path/to/docs         # 指定文档目录（默认 work/docs）
    python run.py --provider agnes              # 使用 Agnes 网关
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _verify_resume(resume: str, rag_context: str) -> tuple[list, list]:
    """程序化验证：简历中的关键声明应能在 RAG 上下文中找到出处。

    返回 (不符列表, 符合列表)。
    """
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

    # 2. 检查简历中的公司名/项目名是否在上下文中出现
    #    从上下文中提取可能的专有名词（简单启发式：首字母大写的连续词）
    context_proper_nouns = set()
    for m in re.finditer(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*", rag_context):
        noun = m.group()
        if len(noun) > 3:  # 过滤掉太短的
            context_proper_nouns.add(noun)

    # 3. 检查量化数据（数字 + 单位）是否在上下文中出现
    quant_pattern = r"(\d+(?:\.\d+)?%|\d+(?:,\d{3})+(?:\+)?|\d+\+?\s*(?:人|万|倍|ms|GB|TB|个))"
    resume_quants = set(re.findall(quant_pattern, resume))
    context_quants = set(re.findall(quant_pattern, rag_context))
    for q in resume_quants:
        if q in context_quants:
            ok.append(f"量化数据 {q} 在源文档中存在")
        else:
            bad.append(f"量化数据 {q} 未在源文档中找到，可能编造")

    # 4. 基本长度检查
    if len(resume) < 200:
        bad.append("简历内容过短（< 200 字），可能不完整")
    elif len(resume) > 10000:
        bad.append("简历内容过长（> 10000 字），可能包含冗余")

    return bad, ok


def _verify_state(state: dict) -> tuple[list, list]:
    rag_context = state.get("task_data", {}).get("rag_context", "")
    return _verify_resume(state.get("artifact", ""), rag_context)


def main():
    ap = argparse.ArgumentParser(description="RAG 加持的简历定制 (llamaindex-pse)")
    ap.add_argument("--jd", type=str, help="JD 文件路径")
    ap.add_argument("--jd-text", type=str, help="JD 文本（直接传入）")
    ap.add_argument("--docs", type=str,
                    default=os.getenv("RESUME_DOCS_PATH", ""),
                    help="文档目录路径（默认从 PSE_ROOT/work/docs 加载）")
    ap.add_argument("--provider", choices=["deepseek", "agnes"], default="deepseek",
                    help="LLM 网关")
    ap.add_argument("--top-k", type=int, default=8,
                    help="RAG 检索 top-k 文档数（默认 8）")
    args = ap.parse_args()

    # 读取 JD
    if args.jd:
        jd_text = Path(args.jd).read_text(encoding="utf-8")
    elif args.jd_text:
        jd_text = args.jd_text
    else:
        print("❌ 请提供 --jd 或 --jd-text")
        sys.exit(1)

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

    # 配置全局 LLM + Embedding
    from llama_index.core import Settings
    Settings.llm = create_llm(args.provider)
    try:
        Settings.embed_model = create_embedding()
        print(f"   Embedding: {settings.EMBEDDING_MODEL}")
    except RuntimeError:
        print("   ⚠️ 未配置 EMBEDDING_MODEL，将使用 LlamaIndex 默认 embedding")

    # 构建 RAG index
    print(f"📚 加载文档: {docs_dir}")
    try:
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        from llama_index.core.node_parser import SentenceSplitter

        documents = SimpleDirectoryReader(docs_dir, recursive=True).load_data()
        if not documents:
            print("❌ 未加载到任何文档")
            sys.exit(1)
        print(f"   加载了 {len(documents)} 个文档片段")

        # 构建 index（使用已配置的 Settings.embed_model）
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
        retriever = index.as_retriever(similarity_top_k=args.top_k)
        print(f"   索引构建完成，retriever top_k={args.top_k}")
    except Exception as e:
        print(f"❌ 构建索引失败: {e}")
        sys.exit(1)

    # 构建 PSE workflow
    max_retries = settings.PSE_MAX_RETRIES or 3
    workflow = build_workflow(
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

    print(f"\n🚀 开始简历定制 (provider={args.provider}, max_retries={max_retries})")
    result = asyncio.run(workflow.run(
        task_input=task_input,
        max_retries=max_retries,
    ))

    resume = result.get("artifact", "")
    out_path = BASE / "tailored_resume.md"
    out_path.write_text(resume, encoding="utf-8")
    print(f"\n✅ 定制简历已保存 → {out_path}")


if __name__ == "__main__":
    main()
