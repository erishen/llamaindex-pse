"""LlamaIndex PSE — 通用 Planner → Specialist → Evaluator → Fix Workflow 核心。

任务无关：通过 task 参数加载 tasks/<task>/prompts/{planner,specialist,evaluator}.md，
通过 verify_fn 注入任务专属的程序化核查。

Workflow 结构:
    START → [planner] → specialist → evaluator ─┬─(通过)─▶ END
                                              └─(仍有问题)─▶ fix → evaluator (循环)
- planner / specialist：LLM 两角色（规划 / 执行）
- evaluator：合并闸门 = LLM 评审(仅首轮) + 程序化 verify_fn 硬核查（每轮，防编造）
- fix：LLM 按核查出的问题修正产物

LlamaIndex 独特优势：
- RAG 增强：planner / specialist 可选注入 retriever，从知识库检索真实上下文，
  产物从源头 grounded，减少幻觉（而非仅在 evaluator 检测幻觉后修补）。
- Workflow + @step + Event：事件驱动的控制流，Evaluator 返回 FixEvent 或 StopEvent。
"""

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from .config import settings
from .model import create_llm
from .prompts import load_prompt
from .tools import TOOLS


# ─────────────────────── 状态模型 ───────────────────────

@dataclass
class PSEState:
    """PSE Workflow 状态（dataclass，通过 Context 传递）。"""
    task_input: str = ""
    task_data: dict = field(default_factory=dict)
    plan: str = ""
    artifact: str = ""
    attempts: int = 0
    fictitious: list = field(default_factory=list)
    verified: list = field(default_factory=list)
    eval_issues: list = field(default_factory=list)
    max_retries: int = 3
    # RAG 检索上下文（planner / specialist 自动填充）
    rag_context: str = ""


# ─────────────────────── 事件 ───────────────────────

class PlannerEvent(Event):
    """触发 Planner 步骤。"""
    task_input: str = ""


class SpecialistEvent(Event):
    """触发 Specialist 步骤。"""
    task_input: str = ""
    plan: str = ""


class EvaluatorEvent(Event):
    """触发 Evaluator 步骤。"""
    artifact: str = ""
    attempts: int = 0


class FixEvent(Event):
    """触发 Fix 步骤。"""
    artifact: str = ""
    issues: list = field(default_factory=list)


# ─────────────────────── RAG 辅助 ───────────────────────

async def _retrieve_context(retriever, query: str, top_k: int = 5) -> str:
    """用 LlamaIndex retriever 检索相关文档，拼接为上下文字符串。

    query 过长时会超出 embedding 模型上下文长度，因此截取前 200 字作为检索关键词。
    """
    # 截取短查询：embedding 模型上下文有限（如 Ollama snowflake-arctic-embed2 仅 8K tokens）
    short_query = query[:200] if len(query) > 200 else query
    nodes = retriever.retrieve(short_query)
    if not nodes:
        return ""
    # 取 top_k 个节点，按 score 降序
    sorted_nodes = sorted(nodes, key=lambda n: n.score or 0, reverse=True)[:top_k]
    parts = []
    for i, node in enumerate(sorted_nodes, 1):
        score = f" (score={node.score:.2f})" if node.score is not None else ""
        parts.append(f"[{i}]{score}\n{node.get_content()}")
    return "\n\n".join(parts)


# ─────────────────────── Workflow ───────────────────────

class PSEWorkflow(Workflow):
    """Planner-Specialist-Evaluator 三角色 Workflow。

    通过 @step 装饰器定义节点，通过返回事件控制流转。

    LlamaIndex 独特能力：
    - retriever：可选的 LlamaIndex Retriever，为 planner / specialist 提供
      RAG 增强上下文。这是 llamaindex-pse 区别于 langgraph-pse 的核心——
      不靠 evaluator 事后检测幻觉，而是从源头就用检索到的真实文档 grounding。
    """

    def __init__(
        self,
        llm=None,
        task: Optional[str] = None,
        tools=None,
        verify_fn: Optional[Callable] = None,
        max_retries: int = 3,
        use_planner: bool = True,
        provider: str = "deepseek",
        retriever=None,
        rag_top_k: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._llm = llm or create_llm(provider)
        self._task = task
        self._tools = tools or TOOLS
        self._verify_fn = verify_fn
        self._max_retries = max_retries
        self._use_planner = use_planner
        self._provider = provider
        self._retriever = retriever
        self._rag_top_k = rag_top_k

        self._planner_prompt = load_prompt("planner", task) if use_planner else ""
        self._specialist_prompt = load_prompt("specialist", task)
        self._evaluator_prompt = load_prompt("evaluator", task)

    @step
    async def start_node(self, ctx: Context, ev: StartEvent) -> PlannerEvent | SpecialistEvent:
        """入口节点：从 StartEvent 读取初始数据，存入 Context，然后分流。"""
        state = PSEState(
            task_input=ev.get("task_input", ""),
            task_data=ev.get("task_data", {}),
            max_retries=ev.get("max_retries", self._max_retries),
        )
        await ctx.store.set("state", state)

        if self._use_planner:
            return PlannerEvent(task_input=state.task_input)
        return SpecialistEvent(task_input=state.task_input, plan="")

    @step
    async def planner(self, ctx: Context, ev: PlannerEvent) -> SpecialistEvent:
        """Planner：RAG 检索 + 读取上下文 → 产出执行规划。"""
        # RAG 增强：检索与任务相关的文档
        rag_ctx = ""
        if self._retriever:
            rag_ctx = await _retrieve_context(self._retriever, ev.task_input, self._rag_top_k)
            if rag_ctx:
                print(f"  📚 RAG 检索到 {len(rag_ctx)} 字上下文")

        full_input = ev.task_input
        if rag_ctx:
            full_input += f"\n\n## 检索到的参考文档\n{rag_ctx}"

        messages = []
        if self._planner_prompt:
            messages.append({"role": "system", "content": self._planner_prompt})
        messages.append({"role": "user", "content": full_input})
        plan = self._llm.chat(messages)
        print(f"✅ 规划已完成 ({len(plan)} 字)")

        state: PSEState = await ctx.store.get("state")
        state.plan = plan
        state.rag_context = rag_ctx
        await ctx.store.set("state", state)

        return SpecialistEvent(task_input=ev.task_input, plan=plan)

    @step
    async def specialist(self, ctx: Context, ev: SpecialistEvent) -> EvaluatorEvent:
        """Specialist：RAG 检索 + 把规划展开为最终产物。"""
        full = (ev.task_input + "\n\n## 执行规划\n" + ev.plan) if ev.plan else ev.task_input

        # RAG 增强：如果 planner 没跑（use_planner=False），这里补检索
        state: PSEState = await ctx.store.get("state")
        rag_ctx = state.rag_context
        if not rag_ctx and self._retriever:
            rag_ctx = await _retrieve_context(self._retriever, ev.task_input, self._rag_top_k)
            if rag_ctx:
                print(f"  📚 RAG 检索到 {len(rag_ctx)} 字上下文")
                state.rag_context = rag_ctx
                await ctx.store.set("state", state)

        if rag_ctx:
            full += f"\n\n## 检索到的参考文档（产物必须基于此，禁止编造）\n{rag_ctx}"

        messages = []
        if self._specialist_prompt:
            messages.append({"role": "system", "content": self._specialist_prompt})
        messages.append({"role": "user", "content": full})
        artifact = self._llm.chat(messages)
        if not artifact:
            raise RuntimeError("Specialist 未输出任何内容")

        state.artifact = artifact
        await ctx.store.set("state", state)

        return EvaluatorEvent(artifact=artifact, attempts=state.attempts)

    @step
    async def evaluator(self, ctx: Context, ev: EvaluatorEvent) -> FixEvent | StopEvent:
        """Evaluator（合并闸门）：LLM 评审(仅首轮) + 程序化 verify_fn 硬核查(每轮)。"""
        state: PSEState = await ctx.store.get("state")

        # 1) LLM 评审（仅首轮）
        eval_issues: list = []
        if ev.attempts == 0 and self._evaluator_prompt:
            scan = state.task_data.get("scan_result", {})
            scan_str = json.dumps(scan, ensure_ascii=False, indent=2)

            # RAG 交叉验证：用检索到的文档补充评审依据
            rag_section = ""
            if state.rag_context:
                rag_section = f"\n\n## RAG 检索到的参考文档（评审依据）\n{state.rag_context}"

            full = (
                f"## 待评估的产物\n{ev.artifact}\n\n"
                f"## 真实数据（供核对，禁止以产物之外的内容为依据）\n{scan_str}"
                f"{rag_section}"
            )
            resp = self._llm.complete(
                self._evaluator_prompt + "\n\n" + full
            )
            text = str(resp)
            eval_issues = _parse_eval_issues(text)
            print(f"  🔍 评审员发现问题 {len(eval_issues)} 项")

        # 2) 程序化核查（每轮）
        if self._verify_fn is not None:
            # 构造类 dict state 给 verify_fn（兼容 langgraph-pse 签名）
            state_dict = {
                "task_input": state.task_input,
                "task_data": state.task_data,
                "plan": state.plan,
                "artifact": state.artifact,
                "attempts": state.attempts,
                "fictitious": state.fictitious,
                "verified": state.verified,
                "eval_issues": state.eval_issues,
                "max_retries": state.max_retries,
                "rag_context": state.rag_context,
            }
            prog_bad, ok = self._verify_fn(state_dict)
        else:
            prog_bad, ok = [], []

        all_bad = list(prog_bad) + list(eval_issues)
        state.attempts += 1
        state.fictitious = all_bad
        state.verified = ok
        state.eval_issues = eval_issues
        await ctx.store.set("state", state)

        print(f"\n{'=' * 60}")
        print(f"  核查 (第{state.attempts}次) — 程序化通过 {len(ok)} 项, "
              f"程序化问题 {len(prog_bad)} 项, 评审问题 {len(eval_issues)} 项")
        for b in all_bad:
            print(f"    ❌ {b}")

        # 决定：通过 → Stop，否则 → Fix
        if not all_bad or state.attempts > state.max_retries:
            if not all_bad:
                print("  ✅ 核查通过")
            else:
                print("  ⚠️ 达到最大重试次数，停止修正")
            return StopEvent(result={"artifact": state.artifact, "state": state})

        return FixEvent(artifact=state.artifact, issues=all_bad)

    @step
    async def fix(self, ctx: Context, ev: FixEvent) -> EvaluatorEvent:
        """Fix：按核查出的问题修正产物。RAG 上下文注入修正提示，防止凭空编造。"""
        state: PSEState = await ctx.store.get("state")
        scan = state.task_data.get("scan_result", {})
        scan_str = json.dumps(scan, ensure_ascii=False, indent=2)

        # RAG 上下文：修正时也基于检索到的真实文档
        rag_section = ""
        if state.rag_context:
            rag_section = (
                "\n\n**RAG 参考文档（修正时必须以此为准）**:\n"
                f"{state.rag_context}\n"
            )

        print("  🔄 自动修正中...")
        prompt = (
            "以下产物被程序化核查发现问题，请修正。\n\n"
            f"**问题清单（必须修复）**:\n" + "\n".join(f"- {i}" for i in ev.issues) + "\n\n"
            "**真实数据（修正时必须以此为准，把错误数字改为真实值，"
            "不得编造也不得删除数字）**:\n"
            f"{scan_str}\n"
            f"{rag_section}\n"
            "**规则**:\n"
            "1. 仅修正问题清单中指出的错误，将错误数字改为真实数据中的正确值\n"
            "2. 不要删除任何正确的数字或内容，保持其余部分不变\n"
            "3. 输出修正后的完整产物，不输出解释\n\n"
            f"## 当前产物\n{ev.artifact}"
        )
        resp = self._llm.complete(prompt)
        fixed = str(resp)

        state.artifact = fixed
        state.eval_issues = []
        await ctx.store.set("state", state)

        return EvaluatorEvent(artifact=fixed, attempts=state.attempts)


# ─────────────────────── 辅助函数 ───────────────────────

def _parse_eval_issues(text: str) -> list[str]:
    """解析评审员(LLM)输出：PASS/无问题 → []；否则收集 '- ' 开头的行。"""
    if not text:
        return []
    if "PASS" in text.upper() and "-" not in text:
        return []
    issues = [ln[2:].strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
    return issues


# ─────────────────────── 构建入口 ───────────────────────

def build_workflow(
    llm=None,
    task: Optional[str] = None,
    tools=None,
    verify_fn: Optional[Callable] = None,
    max_retries: int = 3,
    use_planner: bool = True,
    provider: str = "deepseek",
    retriever=None,
    rag_top_k: int = 5,
) -> PSEWorkflow:
    """构建通用 PSE Workflow。

    llm:         LlamaIndex LLM（缺省按 provider 创建）。
    task:        任务名，用于加载 tasks/<task>/prompts/{planner,specialist,evaluator}.md。
    tools:       注入 agent 的工具列表（默认 read_file + run_bash）。
    verify_fn:   程序化核查函数，签名 (state) -> (bad: list, ok: list)；不传则默认通过。
    use_planner: 是否包含 planner 节点（无规划需求的任务可关掉，从 specialist 起步）。
    provider:    "deepseek" | "agnes"，决定 LLM 网关。
    retriever:   LlamaIndex Retriever（可选）。传入后 planner/specialist 自动 RAG 增强：
                 检索与任务相关的文档，产物从源头 grounded，减少幻觉。
    rag_top_k:   RAG 检索返回的最大文档数（默认 5）。
    返回 PSEWorkflow 实例，用 await workflow.run(task_input=..., task_data=..., max_retries=...) 调用。
    """
    return PSEWorkflow(
        llm=llm,
        task=task,
        tools=tools,
        verify_fn=verify_fn,
        max_retries=max_retries,
        use_planner=use_planner,
        provider=provider,
        retriever=retriever,
        rag_top_k=rag_top_k,
    )
