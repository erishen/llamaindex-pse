"""提示词加载 — 从任务目录的 prompts/*.md 加载。"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent.parent / "tasks"


def load_prompt(name: str, task: str | None = None) -> str:
    """加载指定角色的系统提示词。

    优先读 tasks/<task>/prompts/<name>.md，找不到返回空串。
    """
    if task:
        prompt_path = PROMPTS_DIR / task / "prompts" / f"{name}.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
    return ""
