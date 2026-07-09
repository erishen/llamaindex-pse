"""LlamaIndex PSE — 通用 Planner → Specialist → Evaluator → Fix Workflow 核心。

任务无关：通过 task 参数加载 tasks/<task>/prompts/{planner,specialist,evaluator}.md，
通过 verify_fn 注入任务专属的程序化核查。

Workflow 结构:
    START → [planner] → specialist → evaluator ─┬─(通过)─▶ END
                                              └─(仍有问题)─▶ fix → evaluator (循环)
- planner / specialist：LLM 两角色（规划 / 执行）
- evaluator：合并闸门 = LLM 评审(仅首轮) + 程序化 verify_fn 硬核查（每轮，防编造）
- fix：LLM 按核查出的问题修正产物

使用 LlamaIndex Workflow + Step + Event 实现控制流，
与 langgraph-pse 的 StateGraph + 条件边方案殊途同归。
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


# ─────────────────────── Workflow ───────────────────────

class PSEWorkflow(Workflow):
    """Planner-Specialist-Evaluator 三角色 Workflow。

    通过 @step 装饰器定义节点，通过 emit 事件控制流转。
    与 langgraph-pse 的 StateGraph 方案等价，但使用 LlamaIndex 的
    Workflow 原语（Step + Event + Context）。
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
        await ctx.set("state", state)

        if self._use_planner:
            return PlannerEvent(task_input=state.task_input)
        return SpecialistEvent(task_input=state.task_input, plan="")

    @step
    async def planner(self, ctx: Context, ev: PlannerEvent) -> SpecialistEvent:
        """Planner：读取上下文，产出执行规划。"""
        from llama_index.core.agent import FunctionCallingAgent

        agent = FunctionCallingAgent.from_tools(
            self._tools,
            llm=self._llm,
            system_prompt=self._planner_prompt or None,
            verbose=False,
        )
        response = agent.chat(ev.task_input)
        plan = str(response)
        print(f"✅ 规划已完成 ({len(plan)} 字)")

        state: PSEState = await ctx.get("state")
        state.plan = plan
        await ctx.set("state", state)

        return SpecialistEvent(task_input=ev.task_input, plan=plan)

    @step
    async def specialist(self, ctx: Context, ev: SpecialistEvent) -> EvaluatorEvent:
        """Specialist：把规划（或原始任务）展开为最终产物。"""
        from llama_index.core.agent import FunctionCallingAgent

        full = (ev.task_input + "\n\n## 执行规划\n" + ev.plan) if ev.plan else ev.task_input

        agent = FunctionCallingAgent.from_tools(
            self._tools,
            llm=self._llm,
            system_prompt=self._specialist_prompt or None,
            verbose=False,
        )
        response = agent.chat(full)
        artifact = str(response)
        if not artifact:
            raise RuntimeError("Specialist 未输出任何内容")

        state: PSEState = await ctx.get("state")
        state.artifact = artifact
        await ctx.set("state", state)

        return EvaluatorEvent(artifact=artifact, attempts=state.attempts)

    @step
    async def evaluator(self, ctx: Context, ev: EvaluatorEvent) -> FixEvent | StopEvent:
        """Evaluator（合并闸门）：LLM 评审(仅首轮) + 程序化 verify_fn 硬核查(每轮)。"""
        state: PSEState = await ctx.get("state")

        # 1) LLM 评审（仅首轮）
        eval_issues: list = []
        if ev.attempts == 0 and self._evaluator_prompt:
            scan = state.task_data.get("scan_result", {})
            scan_str = json.dumps(scan, ensure_ascii=False, indent=2)
            full = (
                f"## 待评估的产物\n{ev.artifact}\n\n"
                f"## 真实数据（供核对，禁止以产物之外的内容为依据）\n{scan_str}"
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
            }
            prog_bad, ok = self._verify_fn(state_dict)
        else:
            prog_bad, ok = [], []

        all_bad = list(prog_bad) + list(eval_issues)
        state.attempts += 1
        state.fictitious = all_bad
        state.verified = ok
        state.eval_issues = eval_issues
        await ctx.set("state", state)

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
        """Fix：按核查出的问题修正产物。"""
        state: PSEState = await ctx.get("state")
        scan = state.task_data.get("scan_result", {})
        scan_str = json.dumps(scan, ensure_ascii=False, indent=2)

        print("  🔄 自动修正中...")
        prompt = (
            "以下产物被程序化核查发现问题，请修正。\n\n"
            f"**问题清单（必须修复）**:\n" + "\n".join(f"- {i}" for i in ev.issues) + "\n\n"
            "**真实数据（修正时必须以此为准，把错误数字改为真实值，"
            "不得编造也不得删除数字）**:\n"
            f"{scan_str}\n\n"
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
        await ctx.set("state", state)

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
) -> PSEWorkflow:
    """构建通用 PSE Workflow。

    llm:         LlamaIndex LLM（缺省按 provider 创建）。
    task:        任务名，用于加载 tasks/<task>/prompts/{planner,specialist,evaluator}.md。
    tools:       注入 agent 的工具列表（默认 read_file + run_bash + query_crm + crm_qa_scan）。
    verify_fn:   程序化核查函数，签名 (state) -> (bad: list, ok: list)；不传则默认通过。
    use_planner: 是否包含 planner 节点（无规划需求的任务可关掉，从 specialist 起步）。
    provider:    "deepseek" | "agnes"，决定 LLM 网关。
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
    )
