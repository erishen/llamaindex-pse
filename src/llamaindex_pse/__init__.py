"""LlamaIndex PSE — Planner-Specialist-Evaluator 三角色 Agent 框架（LlamaIndex Workflow 实现）。

公开 API：
- build_workflow(llm=None, max_retries=3) — 构建 PSE Workflow
- create_llm() — 创建带重试的 OpenAI 兼容 LLM
- settings — 配置（环境变量）
"""

from .config import settings
from .model import create_llm
from .workflow import build_workflow, PSEState

__all__ = ["build_workflow", "create_llm", "settings", "PSEState"]
