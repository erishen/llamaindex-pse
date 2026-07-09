"""工具函数 — 文件读取（沙箱）和 bash 执行（沙箱）。

read_file 限制只能读取项目根目录内的文件；run_bash 禁止破坏性命令，
工作目录限定在项目根内。

使用 LlamaIndex FunctionTool 封装。
"""

import os
import re
import subprocess
from pathlib import Path

from llama_index.core.tools import FunctionTool

# 项目根目录，限制文件访问范围（可由 PSE_ROOT 覆盖）
_PROJECT_ROOT = Path(os.getenv("PSE_ROOT", Path.cwd())).resolve()


def read_file(path: str) -> str:
    """读取文件内容。参数 path 为文件路径（限定在项目目录内）。"""
    p = Path(path).resolve()
    if not str(p).startswith(str(_PROJECT_ROOT)):
        return f"[错误] 路径超出项目范围: {path}"
    if not p.exists():
        return f"[错误] 文件不存在: {path}"
    if not p.is_file():
        return f"[错误] 不是文件: {path}"
    return p.read_text(encoding="utf-8")


# 危险命令片段黑名单（命中即拒绝，降低被诱导执行破坏命令的风险）
_DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\b", r"\brm\s+-fr\b", r"\brm\s+-r\b", r"\brm\s+-R\b",
    r"\bmkfs\b", r"\bdd\b\s+if=", r":\(\)\s*\{", r"\bshutdown\b",
    r"\breboot\b", r"\bhalt\b", r"\bpoweroff\b", r">\s*/dev/sd",
    r"\bchmod\b\s+-R\s+777\s+/", r"\bchown\b\s+-R\s+.*\s+/",
    r"curl\b[^\n]*\|\s*(sh|bash)", r"wget\b[^\n]*\|\s*(sh|bash)",
    r"\bnc\b[^\n]*-e\b",
]


def run_bash(command: str) -> str:
    """执行 bash 命令并返回输出（受限沙箱：禁止破坏性命令，工作目录限定在项目根内）。"""
    for pat in _DANGEROUS_PATTERNS:
        if re.search(pat, command):
            return (
                f"[拒绝] 命令命中危险模式（{pat}），已被沙箱拦截。"
                "如需执行破坏性操作请人工进行。"
            )
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=30, cwd=str(_PROJECT_ROOT),
        )
        return result.stdout + "\n" + result.stderr
    except Exception as e:
        return f"[错误] {e}"


# ── LlamaIndex FunctionTool 封装 ──
read_file_tool = FunctionTool.from_defaults(fn=read_file)
run_bash_tool = FunctionTool.from_defaults(fn=run_bash)

TOOLS = [read_file_tool, run_bash_tool]
