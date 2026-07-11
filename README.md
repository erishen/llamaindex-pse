<div align="right">
  <a href="README.zh.md">🇨🇳 中文</a>
</div>

# LlamaIndex PSE

A **Planner-Specialist-Evaluator** multi-agent framework built on [LlamaIndex](https://github.com/run-llama/llama_index) Workflow. It models a generic *generate → verify → fix* loop as an explicit **Workflow** with typed events and step decorators, so that any "produce something, then programmatically check it, and auto-fix if it fails" workflow can be wired up by supplying a task name and a verification function.

This is the LlamaIndex sibling of [`langgraph-pse`](../langgraph-pse) and [`crewai-pse`](../crewai-pse) — same PSE philosophy, but with two LlamaIndex-native advantages:

1. **Workflow + `@step` + Event + Context** — event-driven control flow instead of StateGraph or Crew.
2. **RAG-grounded generation** — optional `retriever` injects real documents into Planner/Specialist, grounding the artifact at the source (not just detecting hallucinations after the fact).

## How It Works

```
START → [planner] → specialist → evaluator ─┬─(pass)─▶ END
                                          └─(issues)─▶ fix → evaluator (loop, max N)
```

1. **Planner (optional)** — RAG retrieves relevant documents, then an LLM (custom OpenAI-compatible client, see `model.py`) produces an execution plan grounded in real context.
2. **Specialist** — RAG retrieves (if Planner didn't), then expands the plan into the final artifact. The artifact is *source-grounded* — far less likely to hallucinate.
3. **Evaluator (merged gate)** — runs every round and combines two checks:
   - **Programmatic verification** via a task-supplied `verify_fn(state) -> (bad, ok)`. This is *not* an LLM judge — deterministic checks are far more reliable than asking a model to grade its own output.
   - **LLM review** (first round only): an independent reviewer inspects the artifact against the real data **and RAG documents**, flagging hallucinations or weak suggestions.
4. **Fix → Evaluator loop** — Fix also receives RAG context, preventing it from fabricating replacements. The evaluator returns `FixEvent` or `StopEvent`, up to `PSE_MAX_RETRIES` times.

The retry loop is naturally expressed as **step → event → step** — no manual loop counters, no re-invoking a team. The Workflow *is* the control flow.

## Why LlamaIndex Workflow?

| | langgraph-pse | llamaindex-pse |
|---|---|---|
| Orchestration | `StateGraph` + conditional edges | `Workflow` + `@step` + Event |
| Retry loop | `add_conditional_edges("evaluator", should_fix)` | Evaluator returns `FixEvent` or `StopEvent` |
| Tool use | LangGraph `create_agent` (LangChain tools) | LlamaIndex `FunctionTool` (sandboxed `read_file` / `run_bash`) |
| State | `TypedDict` on graph edges | `dataclass` via `Context` |
| RAG | — | **Built-in**: `retriever` parameter, auto-grounded Planner/Specialist/Fix |
| Verify step | injected `verify_fn` in the graph | injected `verify_fn` in the workflow |
| Review gate | dedicated **Evaluator** node | dedicated **Evaluator** step |

### RAG: the LlamaIndex advantage

In langgraph-pse, hallucinations are caught *after* generation by the Evaluator's `verify_fn` and LLM review, then patched by Fix. This works, but it's a **detect-and-repair** loop — the model generates something wrong, then fixes it.

In llamaindex-pse, when a `retriever` is provided, the Planner and Specialist receive **retrieved documents as context** before generating. The artifact is *grounded at the source* — far fewer hallucinations to detect in the first place. The Evaluator's `verify_fn` still runs as a safety net, but the RAG layer shifts the defense left:

```
langgraph-pse:  generate → detect(hallucination) → fix → detect → …  (reactive)
llamaindex-pse: retrieve → generate(grounded) → verify(residual)      (proactive)
```

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
| `EMBEDDING_PROVIDER` | | `openai` (DeepSeek/Ali, etc.) or `ollama` (local). Default `openai` |
| `EMBEDDING_MODEL` | ✅‡ | Embedding model name (e.g. `deepseek-embedding`, `text-embedding-v4`) |
| `EMBEDDING_API_KEY` | ✅‡ | Defaults to `OPENAI_API_KEY` |
| `EMBEDDING_BASE_URL` | | Defaults to `OPENAI_BASE_URL` |
| `OLLAMA_BASE_URL` | | Ollama endpoint (default `http://localhost:11434`) |

\* required when `provider="deepseek"` (the default).  &nbsp; † required when `provider="agnes"`.  &nbsp; ‡ required when `EMBEDDING_PROVIDER=openai`.

> The `resume-tailor` task also reads personal config from `.env` (`RESUME_*`, see `.env.example`). These contain PII (company periods, source file, banned year phrases) and must stay in the gitignored `.env`.

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

### RAG-grounded example

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Build a LlamaIndex index from your documents
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=5)

# Pass retriever to the workflow — Planner/Specialist auto-grounded
workflow = build_workflow(
    task="my-task",
    verify_fn=my_verify_fn,
    retriever=retriever,       # ← RAG: the LlamaIndex advantage
    rag_top_k=5,
    provider="deepseek",
)

result = asyncio.run(workflow.run(
    task_input="Summarize the project's architecture",
    max_retries=3,
))
```

## Data Flow & Privacy

This project runs as a local CLI, but it is **not** air-gapped: it transmits your task data to third-party APIs.

- **LLM API** (`chat.completions`) — the full `task_input`, `task_data`, retrieved RAG documents, and every generated artifact are sent to the configured provider (DeepSeek by default, or Agnes). The provider stores prompts under its own retention policy.
- **Embedding API** (`embeddings.create`) — when RAG is enabled (the `resume-tailor` task uses it), the indexed documents are chunked and sent to the embedding endpoint to build the vector index. With the default `EMBEDDING_PROVIDER=openai` (e.g. DeepSeek embedding) this is a second external transfer. Set `EMBEDDING_PROVIDER=ollama` to keep index building fully local.

If your input contains PII (e.g. a résumé with real employers, dates, contact info, or private repo names), that PII leaves your machine. The only way to avoid third-party transfer is to **self-host the models** (local LLM + local embedding).

Local plaintext residues (present even without any API call):
- Task prompts under `tasks/<task>/prompts/*.md` may contain real PII — `tasks/*/prompts/recommend_specialist.md` is gitignored for this reason.
- Generated outputs (`tailored_resume.md`, `recommended_resume.md`) and `.index_cache/` are gitignored but stored unencrypted on disk.

## Relation to Sibling Frameworks

All four share the **PSE role model** and a **verify→fix loop**, but differ in orchestration:

| | `autogen-pse` | `crewai-pse` | `langgraph-pse` | `llamaindex-pse` |
|---|---|---|---|---|
| Orchestration | AutoGen `RoundRobinGroupChat` | CrewAI `Sequential` | LangGraph `StateGraph` + conditional edges | **LlamaIndex `Workflow` + `@step` + Event** |
| Verify step | grep/pytest/ruff | regex/grep in `run.py` | injected `verify_fn` in the graph | injected `verify_fn` in the workflow |
| RAG | optional | — | — | **built-in** (`retriever`, source-grounded — shifts the defense left) |
| Reference use | asset-lens → next-week investment advice | project code → bilingual article → WordPress | CRM data-quality QA + weekly relationship review | **résumé tailoring (RAG)** |
| Best for | cheap, frequent drafts | richer multi-agent publishing | explicit state control + anti-hallucination gates | **RAG-grounded generation** |

The distinctive edge here is RAG: siblings *detect-and-repair* hallucinations after generation, whereas llamaindex-pse can *ground at the source* before generating — see [RAG: the LlamaIndex advantage](#rag-the-llamaindex-advantage).

## Security Notes

- **No hardcoded secrets.** All credentials are read from `.env`, which is gitignored.
- **Sandboxed tools.** `read_file` only reads under `PSE_ROOT`; `run_bash` blocks destructive commands (`rm -rf`, `dd`, `curl|sh`, …) and runs in `PSE_ROOT`.
- **Local process, external data.** The CLI itself is not network-exposed, but it sends data to the LLM/Embedding providers above — see [Data Flow & Privacy](#data-flow--privacy).

## License

MIT
