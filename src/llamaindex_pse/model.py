"""LLM 客户端 — LlamaIndex OpenAI 兼容 LLM。

支持两种 provider（均 OpenAI 兼容协议）：
  - "deepseek"（默认）：用 OPENAI_* 变量
  - "agnes"：用 AGNES_* 变量
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
