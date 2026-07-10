"""LLM / Embedding 客户端 — LlamaIndex OpenAI 兼容 + Ollama。

LLM 支持两种 provider（均 OpenAI 兼容协议）：
  - "deepseek"（默认）：用 OPENAI_* 变量
  - "agnes"：用 AGNES_* 变量

Embedding 支持两种 provider（由 EMBEDDING_PROVIDER 控制）：
  - "openai"（默认）：用 EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL
  - "ollama"：用 OLLAMA_BASE_URL / EMBEDDING_MODEL
"""

from llama_index.llms.openai import OpenAI

from .config import settings


def create_llm(provider: str = "deepseek") -> OpenAI:
    """创建 OpenAI 兼容 LLM（LlamaIndex 封装）。

    provider: "deepseek" | "agnes"
    """
    if provider == "agnes":
        api_key = settings.AGNES_KEY
        base_url = settings.AGNES_BASE_URL
        model = settings.AGNES_MODEL
        label = "AGNES"
    else:
        api_key = settings.OPENAI_API_KEY
        base_url = settings.OPENAI_BASE_URL
        model = settings.OPENAI_MODEL
        label = "OPENAI"

    if not api_key:
        raise RuntimeError(
            f"未设置 {label}_API_KEY / {label}_KEY。请在 .env 中配置（参考 .env.example）。"
        )
    if not model:
        raise RuntimeError(
            f"未设置 {label}_MODEL。请在 .env 中补充模型名（例如 AGNES_MODEL）。"
        )
    return OpenAI(
        model=model,
        api_key=api_key,
        api_base=base_url or None,
        timeout=180,
        max_retries=settings.PSE_MAX_RETRIES or 6,
    )


def create_embedding():
    """创建 Embedding 模型（LlamaIndex 封装）。

    根据 EMBEDDING_PROVIDER 选择后端：
      - "openai"：OpenAIEmbedding，复用 OPENAI_API_KEY/BASE_URL（可单独覆盖）
      - "ollama"：OllamaEmbedding，用 OLLAMA_BASE_URL

    EMBEDDING_MODEL 必填（如 deepseek-embedding, snowflake-arctic-embed2 等）。
    """
    provider = settings.EMBEDDING_PROVIDER.lower()
    model = settings.EMBEDDING_MODEL

    if not model:
        raise RuntimeError(
            "未设置 EMBEDDING_MODEL。请在 .env 中补充"
            "（如 deepseek-embedding, snowflake-arctic-embed2）。"
        )

    if provider == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding

        base_url = settings.OLLAMA_BASE_URL
        return OllamaEmbedding(
            model_name=model,
            base_url=base_url,
        )

    # 默认: openai (OpenAI 兼容，含 DeepSeek / 阿里 DashScope)
    from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingMode

    api_key = settings.EMBEDDING_API_KEY
    base_url = settings.EMBEDDING_BASE_URL

    if not api_key:
        raise RuntimeError(
            "未设置 EMBEDDING_API_KEY（也不存在 OPENAI_API_KEY 兜底）。"
            "请在 .env 中配置。"
        )

    # 非 OpenAI 官方模型（如 deepseek-embedding, text-embedding-v4）
    # 不在 OpenAIEmbeddingModelType 枚举中，需指定 mode 绕过枚举校验
    return OpenAIEmbedding(
        model=model,
        api_key=api_key,
        api_base=base_url or None,
        mode=OpenAIEmbeddingMode.TEXT_MODE,
    )
