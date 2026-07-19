"""配置管理 — 从环境变量和 .env 加载。

所有凭证（API key 等）一律走环境变量，绝不硬编码。
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
    PSE_MAX_RETRIES: int = int(os.getenv("PSE_MAX_RETRIES", "3"))
    # Agnes 网关（与 DeepSeek 同 OpenAI 兼容协议，独立 key / base_url / 模型）
    AGNES_KEY: str = os.getenv("AGNES_KEY", "")
    AGNES_BASE_URL: str = os.getenv("AGNES_BASE_URL", "")
    AGNES_MODEL: str = os.getenv("AGNES_MODEL", "")
    # SCNet 网关（Kimi / MiniMax，OpenAI 兼容协议，独立 key / base_url / 模型）
    SCNET_KEY: str = os.getenv("SCNET_KEY", "")
    SCNET_BASE_URL: str = os.getenv("SCNET_BASE_URL", "")
    SCNET_KIMI_MODEL: str = os.getenv("SCNET_KIMI_MODEL", "")
    SCNET_MINIMAX_MODEL: str = os.getenv("SCNET_MINIMAX_MODEL", "")
    # Embedding 模型（RAG 索引用）
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")  # "openai" | "ollama"
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "")
    # Ollama Embedding（EMBEDDING_PROVIDER=ollama 时使用）
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ── 简历任务个人配置（从 .env 注入，避免硬编码入库）──
    # 公司任职期间：JSON 格式 {"公司": "入职~离职"}（用 ~ 分隔入职/离职）
    RESUME_COMPANY_PERIODS: str = os.getenv("RESUME_COMPANY_PERIODS", "{}")
    # 必须有项目覆盖的公司：逗号分隔
    RESUME_REQUIRED_COMPANIES: str = os.getenv("RESUME_REQUIRED_COMPANIES", "")
    # 核心简历文件名（在 resume2026ppcnlean-v2/ 目录下）
    RESUME_SOURCE_FILE: str = os.getenv("RESUME_SOURCE_FILE", "ai-engineering.md")
    # 当前/最近公司离职时间（用于替换"至今"）
    RESUME_LATEST_COMPANY_END: str = os.getenv("RESUME_LATEST_COMPANY_END", "")
    # 当前/最近公司名及入职时间（用于后处理"至今"替换）
    RESUME_LATEST_COMPANY: str = os.getenv("RESUME_LATEST_COMPANY", "")
    RESUME_LATEST_COMPANY_START: str = os.getenv("RESUME_LATEST_COMPANY_START", "")
    # 项目影响力关键词：JSON [["pattern", score], ...]
    RESUME_IMPACT_KEYWORDS: str = os.getenv("RESUME_IMPACT_KEYWORDS", "")
    # 求职意向默认值（后处理自动填充用）
    RESUME_DEFAULT_POSITION: str = os.getenv("RESUME_DEFAULT_POSITION", "")
    RESUME_DEFAULT_LOCATION: str = os.getenv("RESUME_DEFAULT_LOCATION", "")
    RESUME_DEFAULT_DIRECTION: str = os.getenv("RESUME_DEFAULT_DIRECTION", "")
    # RAG 检索关键词（Planner 市场情报检索用）
    RESUME_RAG_KEYWORDS: str = os.getenv("RESUME_RAG_KEYWORDS", "")
    # 固定目标岗位（自由推荐模式）：设置后大标题与求职意向严格使用该称谓，跳过 LLM 自由选岗
    RESUME_TARGET_ROLE: str = os.getenv("RESUME_TARGET_ROLE", "")
    # 禁止出现的年限表述（正则，用 | 分隔多个模式）
    RESUME_BANNED_YEARS: str = os.getenv("RESUME_BANNED_YEARS", "")
    # 项目时间修正：格式 "起始时间:内容关键词正则"（如 "2026.01:测试体系|工程化"）
    RESUME_LATE_START_FIX: str = os.getenv("RESUME_LATE_START_FIX", "")
    # 职位修正：LLM 常写错职位标题
    RESUME_WRONG_TITLE: str = os.getenv("RESUME_WRONG_TITLE", "")
    RESUME_CORRECT_TITLE: str = os.getenv("RESUME_CORRECT_TITLE", "")
    # 幻觉数字清理：| 分隔的字符串列表
    RESUME_HALLUCINATIONS: str = os.getenv("RESUME_HALLUCINATIONS", "")
    # GitHub 用户名（链接归一化用）
    GITHUB_USERNAME: str = os.getenv("GITHUB_USERNAME", "")


settings = Settings()
