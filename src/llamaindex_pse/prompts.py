"""提示词加载 — 从任务目录的 prompts/*.md 加载。

任务目录约定：tasks/<task>/prompts/{planner,specialist,evaluator}.md
任务目录由使用者自行创建，框架本身不内置任何任务。

模板变量：prompt 文件中可使用 {VAR} 占位符，load_prompt() 会从环境变量
替换为真实值，避免隐私数据硬编码在模板中。
"""

import os
import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent.parent / "tasks"

# 模板变量映射：{GITHUB_USERNAME} → os.getenv("GITHUB_USERNAME", "")
_TEMPLATE_VARS = {
    "GITHUB_USERNAME": lambda: os.getenv("GITHUB_USERNAME", ""),
}


def _substitute(text: str) -> str:
    """替换模板变量 {VAR} 为环境变量值。"""
    for var, getter in _TEMPLATE_VARS.items():
        placeholder = "{" + var + "}"
        if placeholder in text:
            text = text.replace(placeholder, getter())
    return text


def load_prompt(name: str, task: str | None = None) -> str:
    """加载指定角色的系统提示词。

    优先读 tasks/<task>/prompts/<name>.md，找不到返回空串。
    加载后自动替换模板变量（如 {GITHUB_USERNAME}）。
    """
    if task:
        prompt_path = PROMPTS_DIR / task / "prompts" / f"{name}.md"
        if prompt_path.exists():
            return _substitute(prompt_path.read_text(encoding="utf-8"))
    return ""
