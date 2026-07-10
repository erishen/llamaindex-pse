"""LLM / Embedding 客户端 — 直接用 openai SDK 绕开 LlamaIndex 枚举校验。

LLM 支持两种 provider（均 OpenAI 兼容协议）：
  - "deepseek"（默认）：用 OPENAI_* 变量
  - "agnes"：用 AGNES_* 变量

LlamaIndex 的 OpenAI LLM 封装在 metadata 属性中调用
openai_modelname_to_contextsize()，对非 OpenAI 官方模型（deepseek-chat、
agnes-2.0-flash）会报 ValueError。因此直接用 openai SDK 实现 chat/complete，
完全绕开 LlamaIndex 的 OpenAI 封装。

Embedding 同理：CustomOpenAIEmbedding 基于 openai SDK 实现。
"""

import openai

from .config import settings


class SimpleLLM:
    """基于 openai SDK 的轻量 LLM（绕开 LlamaIndex OpenAI 校验）。

    仅提供 PSE workflow 需要的 chat() 和 complete() 接口。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self._model = model
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: list[dict]) -> str:
        """Chat completion，返回 assistant 内容。"""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            timeout=180,
        )
        return resp.choices[0].message.content or ""

    def complete(self, prompt: str) -> str:
        """Text completion（内部转 chat）。"""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            timeout=180,
        )
        return resp.choices[0].message.content or ""

    @property
    def model_name(self) -> str:
        return self._model


def create_llm(provider: str = "deepseek") -> SimpleLLM:
    """创建 OpenAI 兼容 LLM。

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
    return SimpleLLM(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
    )


class CustomOpenAIEmbedding:
    """基于 openai SDK 的自定义 Embedding（绕开 LlamaIndex 枚举校验）。

    实现 LlamaIndex 的 BaseEmbedding 接口，但直接调用 openai SDK，
    不经过 OpenAIEmbeddingModelType 枚举校验。
    适用于所有 OpenAI 兼容 API（DeepSeek、阿里 DashScope 等）。
    """

    def __init__(self, model: str, api_key: str, api_base: str):
        self._model = model
        self._client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self._async_client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)

    def _get_text_embedding(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=text, model=self._model)
        return resp.data[0].embedding

    async def _aget_text_embedding(self, text: str) -> list[float]:
        resp = await self._async_client.embeddings.create(input=text, model=self._model)
        return resp.data[0].embedding

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in resp.data]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        resp = await self._async_client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in resp.data]


def create_embedding():
    """创建 Embedding 模型。

    根据 EMBEDDING_PROVIDER 选择后端：
      - "openai"（默认）：CustomOpenAIEmbedding，直接用 openai SDK
      - "ollama"：OllamaEmbedding，用 OLLAMA_BASE_URL

    EMBEDDING_MODEL 必填（如 deepseek-embedding, text-embedding-v4, snowflake-arctic-embed2 等）。
    """
    from llama_index.core.embeddings import BaseEmbedding

    provider = settings.EMBEDDING_PROVIDER.lower()
    model = settings.EMBEDDING_MODEL

    if not model:
        raise RuntimeError(
            "未设置 EMBEDDING_MODEL。请在 .env 中补充"
            "（如 deepseek-embedding, text-embedding-v4）。"
        )

    if provider == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding

        base_url = settings.OLLAMA_BASE_URL
        return OllamaEmbedding(
            model_name=model,
            base_url=base_url,
        )

    # 默认: openai 兼容（DeepSeek / 阿里 DashScope / OpenAI）
    # 使用自定义实现，绕开 LlamaIndex OpenAIEmbedding 的枚举校验
    api_key = settings.EMBEDDING_API_KEY
    base_url = settings.EMBEDDING_BASE_URL

    if not api_key:
        raise RuntimeError(
            "未设置 EMBEDDING_API_KEY（也不存在 OPENAI_API_KEY 兜底）。"
            "请在 .env 中配置。"
        )

    custom = CustomOpenAIEmbedding(
        model=model,
        api_key=api_key,
        api_base=base_url or None,
    )

    # 包装为 LlamaIndex BaseEmbedding 兼容对象
    class _EmbeddingWrapper(BaseEmbedding):
        def __init__(self, custom_emb):
            super().__init__(model_name=custom_emb._model)
            self._custom = custom_emb

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._custom._get_text_embedding(text)

        async def _aget_text_embedding(self, text: str) -> list[float]:
            return await self._custom._aget_text_embedding(text)

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return self._custom._get_text_embeddings(texts)

        async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return await self._custom._aget_text_embeddings(texts)

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._custom._get_text_embedding(query)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return await self._custom._aget_text_embedding(query)

    return _EmbeddingWrapper(custom)
