<div align="right">
  <a href="README.zh.md">🇨🇳 中文</a>
</div>

# LlamaIndex PSE

A **Planner-Specialist-Evaluator** multi-agent framework built on [LlamaIndex](https://github.com/run-llama/llama_index) Workflow. It models a generic *generate → verify → fix* loop as an explicit **Workflow** with typed events and step decorators, so that any "produce something, then programmatically check it, and auto-fix if it fails" workflow can be wired up by supplying a task name and a verification function.

This is the LlamaIndex sibling of [`langgraph-pse`](../langgraph-pse) and [`crewai-pse`](../crewai-pse) — same PSE philosophy, different orchestration primitive: a **Workflow with `@step` + Event + Context** instead of a StateGraph or a Crew.

## How It Works

```
START → [planner] → specialist → evaluator ─┬─(pass)─▶ END
                                          └─(issues)─▶ fix → evaluator (loop, max N)
```

1. **Planner (optional)** — a `FunctionCallingAgent` reads context via sandboxed tools and produces an execution plan.
2. **Specialist** — expands the plan (or the raw task input) into the final artifact (report, …).
3. **Evaluator (merged gate)** — runs every round and combines two checks:
   - **Programmatic verification** via a task-supplied `verify_fn(state) -> (bad, ok)`. This is *not* an LLM judge — deterministic checks are far more reliable than asking a model to grade its own output (e.g. it guarantees every number in the report matches the scan).
   - **LLM review** (first round only): an independent reviewer inspects the artifact against the real data and flags hallucinations, fabricated samples, or weak suggestions.
4. **Fix → Evaluator loop** — the evaluator step returns either a `FixEvent` (triggering the fix step) or a `StopEvent` (terminating the workflow), up to `PSE_MAX_RETRIES` times.

The retry loop is naturally expressed as **step → event → step** — no manual loop counters, no re-invoking a team. The Workflow *is* the control flow.

## Why LlamaIndex Workflow?

| | langgraph-pse | llamaindex-pse |
|---|---|---|
| Orchestration | `StateGraph` + conditional edges | `Workflow` + `@step` + Event |
| Retry loop | `add_conditional_edges("evaluator", should_fix)` | Evaluator returns `FixEvent` or `StopEvent` |
| Tool use | LangGraph `create_agent` (LangChain tools) | LlamaIndex `FunctionCallingAgent` (FunctionTool) |
| State | `TypedDict` on graph edges | `dataclass` via `Context` |
| Verify step | injected `verify_fn` in the graph | injected `verify_fn` in the workflow |
| Review gate | dedicated **Evaluator** node | dedicated **Evaluator** step |

## Project Structure

```
llamaindex-pse/
├── src/llamaindex_pse/     # Core framework (task-agnostic)
│   ├── __init__.py          # Public API: build_workflow(), create_llm()
│   ├── config.py            # Settings from environment / .env
│   ├── model.py             # LlamaIndex OpenAI-compatible LLM (deepseek / agnes)
│   ├── tools.py             # read_file (sandboxed) + run_bash (sandboxed)
│   ├── prompts.py           # Prompt loader (tasks/<task>/prompts/*.md)
│   └── workflow.py          # Workflow: planner → specialist → evaluator → fix
├── tasks/                   # User-created tasks (not bundled)
├── pyproject.toml
├── Makefile
└── .env.example
```

## Installation

```bash
make install        # or: uv sync
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

You need **either** the `OPENAI_*` set (DeepSeek is OpenAI-compatible) **or** the `AGNES_*` set. Both are supported via `provider` parameter.

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅* | LLM API key (OpenAI-compatible, e.g. DeepSeek) |
| `OPENAI_BASE_URL` | ✅* | LLM API base URL |
| `OPENAI_MODEL` | ✅* | Model name (e.g. `deepseek-chat`) |
| `AGNES_KEY` | ✅† | Alternative: Agnes API key (free model) |
| `AGNES_BASE_URL` | ✅† | Alternative: Agnes base URL |
| `AGNES_MODEL` | ✅† | Alternative: Agnes model name (e.g. `agnes-2.0-flash`) |
| `PSE_ROOT` | ✅ | Sandbox root for `read_file` / `run_bash` |
| `PSE_MAX_RETRIES` | | Max evaluator/fix rounds (default: `3`) |

\* required when `provider="deepseek"` (the default).  &nbsp; † required when `provider="agnes"`.

## Building a task

1. Create `tasks/<your-task>/prompts/{planner,specialist,evaluator}.md`.
2. Call `build_workflow(task="<your-task>", verify_fn=..., use_planner=...)`.
3. `verify_fn(state) -> (bad, ok)` is your deterministic check; the workflow loops `fix` until it passes or hits `max_retries`.

### Quick example

```python
import asyncio
from llamaindex_pse import build_workflow

workflow = build_workflow(
    task="my-task",
    verify_fn=my_verify_fn,
    use_planner=True,
    provider="deepseek",
)

result = asyncio.run(workflow.run(
    task_input="Your task description here",
    task_data={"key": "value"},  # optional extra data for verify_fn
    max_retries=3,
))
print(result["artifact"])
```

## Security Notes

- **No hardcoded secrets.** All credentials are read from `.env`, which is gitignored.
- **Sandboxed tools.** `read_file` only reads under `PSE_ROOT`; `run_bash` blocks destructive commands (`rm -rf`, `dd`, `curl|sh`, …) and runs in `PSE_ROOT`.
- **No network-exposed service.** This project runs locally as a CLI.

## License

MIT
