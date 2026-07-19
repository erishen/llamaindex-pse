"""运行时脱敏：发往外部 LLM / Embedding API 前，遮蔽可直接定位个人的标识符。

设计原则（手术式脱敏）：
- 只遮蔽「能直接定位到个人」的标识符：姓名、企业邮箱、个人邮箱、手机号、个人站点、GitHub。
- 保留雇主、项目、技术栈、日期等简历实质内容——这些是简历生成所必需的，且不属于直接标识符。
- 源文件不被修改，仅在运行时（构建索引 / 拼装发给 LLM 的 prompt）脱敏。
- 本地 verify_fn 仍使用原始简历做事实核查，不受影响。

⚠️ 安全：本文件**不含任何真实 PII**。具体标识符规则来自本地、已 gitignore 的
`privacy_patterns.json`（或环境变量 `RESUME_REDACT`），切勿把真实标识符提交进仓库。

切换：设环境变量 RESUME_DESENSITIZE=false 可关闭（默认开启）。
失败保护：脱敏已启用却加载不到任何规则时（如新克隆 / CI 缺失
`privacy_patterns.json` 且未设 `RESUME_REDACT`），会向 stderr 发出**一次**显式告警，
避免「静默明文外发」这一最危险的泄露路径；设 `RESUME_REDACT_STRICT=true` 则直接抛错
fail-fast。结构模板见 `privacy_patterns.example.json`（可入库，不含真实 PII）。
"""

import json
import os
import re
import sys
from pathlib import Path

# 本地规则文件（gitignore，含真实标识符，不提交）
_PATTERNS_FILE = Path(__file__).resolve().parent / "privacy_patterns.json"


def _load_raw() -> object:
    """从环境变量 RESUME_REDACT(JSON) 或本地 privacy_patterns.json 加载原始配置。"""
    raw = os.getenv("RESUME_REDACT")
    if not raw:
        p = _PATTERNS_FILE
        if p.exists():
            try:
                raw = p.read_text(encoding="utf-8")
            except Exception:
                raw = None
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


def _load_patterns() -> list[tuple[re.Pattern, str]]:
    """脱敏规则：兼容 list[[pattern, repl]] 或 dict{"patterns": [...]} 两种格式。"""
    data = _load_raw()
    items = data["patterns"] if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    return [(re.compile(pat, re.I), repl) for pat, repl in items]


def _load_local_config() -> dict:
    """本地回填配置：restore 映射 + opensource 列表（均来自 gitignore 文件，不外发）。"""
    data = _load_raw()
    return data if isinstance(data, dict) else {}


_PATTERNS: list[tuple[re.Pattern, str]] = _load_patterns()
_WARNED_NO_PATTERNS = False


def _resolve_patterns() -> list[tuple[re.Pattern, str]]:
    """返回脱敏规则；允许运行时文件出现后重新加载。

    若脱敏已启用（RESUME_DESENSITIZE=true）却加载不到任何规则，发出**一次**显式告警
    到 stderr，避免「静默明文外发」这一最危险的隐私泄露路径。设 RESUME_REDACT_STRICT=true
    时直接抛错，强制在 CI / 新克隆环境 fail-fast。
    """
    global _PATTERNS, _WARNED_NO_PATTERNS
    if not _PATTERNS:
        _PATTERNS = _load_patterns()  # 文件可能在 import 后才生成
    if not _PATTERNS and is_enabled() and not _WARNED_NO_PATTERNS:
        _WARNED_NO_PATTERNS = True
        strict = os.getenv("RESUME_REDACT_STRICT", "false").lower() == "true"
        msg = (
            "⚠️ [privacy] 脱敏已启用（RESUME_DESENSITIZE=true）但加载不到任何规则："
            "privacy_patterns.json 缺失且未设置 RESUME_REDACT 环境变量。"
            "文本将按原样发往外部 LLM / Embedding API，存在 PII 泄露风险！"
            "请放置 privacy_patterns.json（参考 privacy_patterns.example.json），"
            "或显式设置 RESUME_DESENSITIZE=false 关闭脱敏。"
        )
        print(msg, file=sys.stderr)
        if strict:
            raise RuntimeError(msg)
    return _PATTERNS


def desensitize_text(text: str) -> str:
    """遮蔽文本中的直接个人标识符。无规则或空输入原样返回。"""
    if not text:
        return text
    patterns = _resolve_patterns()
    if not patterns:
        return text
    for pat, repl in patterns:
        text = pat.sub(repl, text)
    return text


def is_enabled() -> bool:
    """是否启用脱敏（默认开启，RESUME_DESENSITIZE=false 关闭）。"""
    return os.getenv("RESUME_DESENSITIZE", "true").lower() != "false"


def redact(text: str) -> str:
    """按开关与本地规则脱敏：关闭或无规则时原样返回。"""
    return desensitize_text(text) if is_enabled() else text


def _build_opensource_section(items: list, github_base: str) -> str:
    """由本地配置构建「开源项目」段落（GitHub 用户名取自本地，不外发）。"""
    lines = ["## 开源项目", ""]
    for it in items:
        name = it.get("name", "")
        desc = it.get("desc", "")
        stack = it.get("stack", "")
        url = f"https://{github_base}/{name}" if github_base else f"https://github.com/{name}"
        lines.append(f"- **{name}** — {desc} | [{url}]({url}) | {stack}")
    return "\n".join(lines)


def finalize(text: str) -> str:
    """本地回填：还原脱敏占位符为真实信息，并保证「开源项目」段落存在。

    仅作用于最终落盘文本，绝不回传 API——这是「外发脱敏、本地还原」的闭环。
    开关关闭（RESUME_DESENSITIZE=false）或无本地配置时，原样返回。
    """
    if not text or not is_enabled():
        return text
    cfg = _load_local_config()
    restore_map = cfg.get("restore", {})
    # 同时支持半角括号占位符（LLM 偶尔回显半角 [x] 而非全角 ［x］）
    norm_map = dict(restore_map)
    for k, v in restore_map.items():
        norm_map[k.replace("［", "[").replace("］", "]")] = v
    for placeholder, value in norm_map.items():
        if placeholder in text:
            text = text.replace(placeholder, value)

    # 兜底：LLM 偶发把大标题里的姓名占位符写成裸「姓名」（源自结构模板字面、
    # 未带括号，故未命中上面的 ［姓名］ 还原规则）。在标题位置补还原为真实姓名。
    # 数据驱动：姓名真实值取自本地 restore_map，不在此硬编码任何 PII。
    name_val = restore_map.get("［姓名］") or restore_map.get("[姓名]")
    if name_val:
        text = re.sub(
            r"^#\s*姓名\s*([|\-])",
            lambda m: f"# {name_val} {m.group(1)}",
            text,
            flags=re.MULTILINE,
        )

    # 开源项目：仅当文本中「不存在任何开源相关章节」时才从本地权威清单注入
    # （GitHub 用户名本地补全，不外发）。基准简历的开源章节标题为
    # 「## 个人开源与 AI 实验」，必须一并识别，否则会与注入的「## 开源项目」
    # 并存，造成重复章节。
    _oss_markers = ("## 开源项目", "## 开源经历", "## 个人开源与 AI 实验", "## 开源贡献")
    if not any(m in text for m in _oss_markers):
        oss = cfg.get("opensource")
        if oss:
            section = _build_opensource_section(oss, restore_map.get("［GitHub］", ""))
            if "## 教育背景" in text:
                text = text.replace("## 教育背景", section + "\n\n## 教育背景", 1)
            else:
                text = text.rstrip() + "\n\n" + section
    return text
