<div align="right">
  <a href="README.md">🌐 English</a>
</div>

# LlamaIndex PSE

基于 [LlamaIndex](https://github.com/run-llama/llama_index) Workflow 的 **Planner-Specialist-Evaluator** 多 Agent 框架。它把通用的「生成 → 程序化核查 → 自动修正」循环建模成一个带类型事件和步骤装饰器的显式 **Workflow**：任何「产出某物，再用程序化方式核查，不通过就自动修」的工作流，只要提供任务名 + 一个核查函数即可挂载。

这是 [`langgraph-pse`](../langgraph-pse) 和 [`crewai-pse`](../crewai-pse) 的 LlamaIndex 版本——同样的 PSE 理念，但有两个 LlamaIndex 原生优势：

1. **Workflow + `@step` + Event + Context** — 事件驱动的控制流，而非 StateGraph 或 Crew。
2. **RAG 加持生成** — 可选的 `retriever` 将真实文档注入 Planner/Specialist，产物从源头 grounded（而非事后检测幻觉再修补）。

## 工作原理

```
START → [planner] → specialist → evaluator ─┬─(通过)─▶ END
                                          └─(有问题)─▶ fix → evaluator（循环，最多 N 轮）
```

1. **Planner（可选）** — RAG 检索相关文档，再由 LLM（自研 OpenAI 兼容客户端，见 `model.py`）基于真实上下文产出执行规划。
2. **Specialist** — RAG 检索（Planner 没跑时补检索），把规划展开为最终产物。产物*源头 grounded*——幻觉概率大幅降低。
3. **Evaluator（合并闸门）** — 每轮都跑，融合两道核查：
   - **程序化验证**：由任务注入的 `verify_fn(state) -> (bad, ok)` 做确定性检查。刻意**不做** LLM 裁判——确定性验证比让模型评判自己输出可靠得多。
   - **LLM 评审**（仅首轮）：独立评审员对照真实数据**和 RAG 文档**审查产物，揪出幻觉、空泛建议。
4. **Fix → Evaluator 循环** — Fix 也接收 RAG 上下文，防止凭空编造替代内容。Evaluator 返回 `FixEvent` 或 `StopEvent`，最多重试 `PSE_MAX_RETRIES` 轮。

重试循环（Evaluator → Fix → Evaluator）天生适合**步骤 → 事件 → 步骤**——不用手写循环计数、不用重复调用 team，Workflow 本身就是控制流。

## 为什么用 LlamaIndex Workflow？

| | langgraph-pse | llamaindex-pse |
|---|---|---|
| 编排 | `StateGraph` + 条件边 | `Workflow` + `@step` + Event |
| 重试循环 | `add_conditional_edges("evaluator", should_fix)` | Evaluator 返回 `FixEvent` 或 `StopEvent` |
| 工具调用 | LangGraph `create_agent`（LangChain tools） | LlamaIndex `FunctionTool`（沙箱版 `read_file` / `run_bash`） |
| 状态 | `TypedDict` 在图边上传递 | `dataclass` 通过 `Context` 传递 |
| RAG | — | **内置**：`retriever` 参数，Planner/Specialist/Fix 自动 grounded |
| 验证步骤 | 图内注入的 `verify_fn` | Workflow 内注入的 `verify_fn` |
| 评审闸门 | 独立 **Evaluator** 节点 | 独立 **Evaluator** 步骤 |

### RAG：LlamaIndex 的核心优势

在 langgraph-pse 中，幻觉靠 Evaluator 的 `verify_fn` 和 LLM 评审**事后检测**，再由 Fix 修补。这可行，但本质是 **detect-and-repair** 循环——模型先编造，再修正。

在 llamaindex-pse 中，传入 `retriever` 后，Planner 和 Specialist 在生成前先接收**检索到的真实文档**。产物从源头就是 grounded 的——待检测的幻觉大幅减少。Evaluator 的 `verify_fn` 仍作为安全网运行，但 RAG 层把防线前移了：

```
langgraph-pse:  generate → detect(幻觉) → fix → detect → …  （被动修补）
llamaindex-pse: retrieve → generate(grounded) → verify(残留)  （主动防御）
```

## 项目结构

```
llamaindex-pse/
├── src/llamaindex_pse/     # 核心框架（任务无关）
│   ├── __init__.py          # 公开 API: build_workflow(), create_llm()
│   ├── config.py            # 从环境变量 / .env 读取配置
│   ├── model.py             # LlamaIndex OpenAI 兼容 LLM（deepseek / agnes）
│   ├── tools.py             # read_file（沙箱）+ run_bash（沙箱）
│   ├── prompts.py           # 提示词加载（tasks/<task>/prompts/*.md）
│   └── workflow.py          # Workflow: planner → specialist → evaluator → fix
├── tasks/                   # 使用者自行创建的任务（框架不内置）
├── pyproject.toml
├── Makefile
└── .env.example
```

## 安装

```bash
make install        # 或: uv sync
```

## 配置

把 `.env.example` 复制为 `.env` 并填写：

```bash
cp .env.example .env
```

需要 **`OPENAI_*` 组（DeepSeek，OpenAI 兼容）** 或 **`AGNES_*` 组** 二者之一；通过 `provider` 参数切换。

| 变量 | 必填 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | ✅* | LLM API key（OpenAI 兼容，如 DeepSeek） |
| `OPENAI_BASE_URL` | ✅* | LLM API base URL |
| `OPENAI_MODEL` | ✅* | 模型名（如 `deepseek-chat`） |
| `AGNES_KEY` | ✅† | 备选：Agnes API key（免费模型） |
| `AGNES_BASE_URL` | ✅† | 备选：Agnes base URL |
| `AGNES_MODEL` | ✅† | 备选：Agnes 模型名（如 `agnes-2.0-flash`） |
| `PSE_ROOT` | ✅ | `read_file` / `run_bash` 沙箱根路径 |
| `PSE_MAX_RETRIES` | | 最大验证/修正轮数（默认 `3`） |
| `EMBEDDING_PROVIDER` | | `openai`（DeepSeek/阿里 等）或 `ollama`（本地）。默认 `openai` |
| `EMBEDDING_MODEL` | ✅‡ | Embedding 模型名（如 `deepseek-embedding`、`text-embedding-v4`） |
| `EMBEDDING_API_KEY` | ✅‡ | 默认复用 `OPENAI_API_KEY` |
| `EMBEDDING_BASE_URL` | | 默认复用 `OPENAI_BASE_URL` |
| `OLLAMA_BASE_URL` | | Ollama 端点（默认 `http://localhost:11434`） |

\* 使用 `provider="deepseek"`（默认）时必填。 &nbsp; † 使用 `provider="agnes"` 时必填。 &nbsp; ‡ `EMBEDDING_PROVIDER=openai` 时必填。

> `resume-tailor` 任务还会从 `.env` 读取个人配置（`RESUME_*` 系列，见 `.env.example`）。这些配置含 PII（任职期间、源文件名、禁用语等），必须留在已 gitignore 的 `.env` 中。

## 新建一个任务

1. 创建 `tasks/<your-task>/prompts/{planner,specialist,evaluator}.md`。
2. 调用 `build_workflow(task="<your-task>", verify_fn=..., use_planner=...)`。
3. `verify_fn(state) -> (bad, ok)` 即你的确定性核查；Workflow 会循环 `fix` 直到通过或达到 `max_retries`。

### 快速示例

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
    task_input="你的任务描述",
    task_data={"key": "value"},  # 可选，供 verify_fn 使用
    max_retries=3,
))
print(result["artifact"])
```

### RAG 加持示例

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 从文档构建 LlamaIndex 索引
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=5)

# 传入 retriever — Planner/Specialist 自动 grounded
workflow = build_workflow(
    task="my-task",
    verify_fn=my_verify_fn,
    retriever=retriever,       # ← RAG：LlamaIndex 的核心优势
    rag_top_k=5,
    provider="deepseek",
)

result = asyncio.run(workflow.run(
    task_input="总结项目的架构设计",
    max_retries=3,
))
```

## 数据流向与隐私

本项目以**本地 CLI** 运行，但**并非离线隔离**：它会把你的任务数据发往第三方 API。

- **LLM API**（`chat.completions`）— 完整的 `task_input`、`task_data`、检索到的 RAG 文档，以及每轮生成的产物，都会发往所选 provider（默认 DeepSeek，或 Agnes）。provider 按其自身留存策略存储 prompt。
- **Embedding API**（`embeddings.create`）— 启用 RAG 时（如 `resume-tailor` 任务）会先把待索引文档切块，再发往 embedding 端点构建向量索引。默认 `EMBEDDING_PROVIDER=openai`（如 DeepSeek embedding）意味着**第二次外部传输**。将 `EMBEDDING_PROVIDER=ollama` 可让索引构建完全在本地完成。

若输入含 PII（例如带真实雇主、日期、联系方式、私人仓库名的简历），这些 PII 会离开本机。要避免第三方传输，唯一途径是**自托管模型**（本地 LLM + 本地 embedding）。

本机明文残留（即使完全不发 API 也存在）：
- `tasks/<task>/prompts/*.md` 可能含真实 PII —— `tasks/*/prompts/recommend_specialist.md` 因此被 gitignore。
- 生成产物（`tailored_resume.md`、`recommended_resume.md`）与 `.index_cache/` 虽已 gitignore，但仍以明文存于磁盘。

## 安全说明

- **无硬编码密钥.** 所有凭证均从 `.env` 读取，`.env` 已 gitignore。
- **沙箱工具.** `read_file` 只能读 `PSE_ROOT` 内文件；`run_bash` 拦截破坏性命令（`rm -rf`、`dd`、`curl|sh` 等）并在 `PSE_ROOT` 内运行。
- **进程本地、数据外发.** 本项目本身不暴露网络服务，但会向上述 LLM/Embedding provider 发送数据——详见[数据流向与隐私](#数据流向与隐私)。

## 许可证

MIT
