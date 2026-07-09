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

1. **Planner（可选）** — RAG 检索相关文档，`FunctionCallingAgent` 基于真实上下文产出执行规划。
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
| 工具调用 | LangGraph `create_agent`（LangChain tools） | LlamaIndex `FunctionCallingAgent`（FunctionTool） |
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

\* 使用 `provider="deepseek"`（默认）时必填。 &nbsp; † 使用 `provider="agnes"` 时必填。

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

## 安全说明

- **无硬编码密钥。** 所有凭证均从 `.env` 读取，`.env` 已 gitignore。
- **沙箱工具.** `read_file` 只能读 `PSE_ROOT` 内文件；`run_bash` 拦截破坏性命令（`rm -rf`、`dd`、`curl|sh` 等）并在 `PSE_ROOT` 内运行。
- **无网络暴露服务.** 本项目仅作为本地 CLI 运行。

## 许可证

MIT
