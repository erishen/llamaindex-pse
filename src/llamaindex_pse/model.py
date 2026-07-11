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

import time

import openai

from .config import settings


class TokenStats:
    """Token 消耗统计收集器，含价格计算。"""

    # 模型定价表：元 / 百万 tokens（人民币）
    PRICING = {
        "deepseek-chat": {"input": 1.0, "output": 2.0},       # DeepSeek V3 官方价
        "deepseek-reasoner": {"input": 4.0, "output": 16.0},  # DeepSeek R1
        "agnes-2.0-flash": {"input": 0.5, "output": 1.5},     # Agnes 网关（估算）
    }

    def __init__(self):
        self.calls: list[dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost_cny = 0.0

    def record(self, stage: str, prompt_tokens: int, completion_tokens: int, model: str):
        total = prompt_tokens + completion_tokens
        # 计算费用
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000
        self.calls.append({
            "stage": stage,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "model": model,
            "cost_cny": cost,
        })
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total
        self.total_cost_cny += cost

    def summary(self) -> str:
        if not self.calls:
            return "无 LLM 调用"
        has_pricing = any(c["cost_cny"] > 0 for c in self.calls)
        lines = [
            f"📊 Token 消耗统计（{len(self.calls)} 次调用）",
            f"   {'阶段':<12} {'输入':>8} {'输出':>8} {'合计':>8}" + (f" {'费用(¥)':>10}" if has_pricing else ""),
            f"   {'─'*12} {'─'*8} {'─'*8} {'─'*8}" + (f" {'─'*10}" if has_pricing else ""),
        ]
        for c in self.calls:
            row = f"   {c['stage']:<12} {c['prompt_tokens']:>8} {c['completion_tokens']:>8} {c['total_tokens']:>8}"
            if has_pricing:
                row += f" {c['cost_cny']:>10.4f}"
            lines.append(row)
        lines.append(
            f"   {'─'*12} {'─'*8} {'─'*8} {'─'*8}" + (f" {'─'*10}" if has_pricing else "")
        )
        total_row = f"   {'合计':<12} {self.total_prompt_tokens:>8} {self.total_completion_tokens:>8} {self.total_tokens:>8}"
        if has_pricing:
            total_row += f" {self.total_cost_cny:>10.4f}"
        lines.append(total_row)
        if has_pricing:
            lines.append(f"   💰 总费用: ¥{self.total_cost_cny:.4f}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_cny": self.total_cost_cny,
        }


# 全局 token 统计实例
token_stats = TokenStats()


class SimpleLLM:
    """基于 openai SDK 的轻量 LLM（绕开 LlamaIndex OpenAI 校验）。

    仅提供 PSE workflow 需要的 chat() 和 complete() 接口。
    内置连接重试（最多 3 次，间隔 5 秒），应对网关偶发断连。
    自动记录 token 消耗到全局 token_stats。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self._model = model
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def _call_with_retry(self, fn, max_retries=3, delay=5):
        """带重试的 API 调用，应对网关偶发断连。"""
        for attempt in range(max_retries):
            try:
                return fn()
            except openai.APIConnectionError as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠️ 连接失败 ({attempt+1}/{max_retries})，{delay}s 后重试...")
                    time.sleep(delay)
                else:
                    raise

    def chat(self, messages: list[dict], stage: str = "chat") -> str:
        """Chat completion，返回 assistant 内容。"""
        resp = self._call_with_retry(lambda: self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            timeout=180,
        ))
        # 记录 token 消耗
        if resp.usage:
            token_stats.record(
                stage=stage,
                prompt_tokens=resp.usage.prompt_tokens or 0,
                completion_tokens=resp.usage.completion_tokens or 0,
                model=self._model,
            )
        return resp.choices[0].message.content or ""

    def complete(self, prompt: str, stage: str = "complete") -> str:
        """Text completion（内部转 chat）。"""
        resp = self._call_with_retry(lambda: self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            timeout=180,
        ))
        # 记录 token 消耗
        if resp.usage:
            token_stats.record(
                stage=stage,
                prompt_tokens=resp.usage.prompt_tokens or 0,
                completion_tokens=resp.usage.completion_tokens or 0,
                model=self._model,
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
