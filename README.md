# 基于LangGraph的企业运维RAG智能Agent 项目（LangGraph + LangSmith + RAG）

面向企业内部IT运维文档场景的私有知识库问答系统，解决人工运维答疑效率低、新人上手慢、文档分散检索难的问题。
> 将原有 RAG 项目升级为基于 **LangGraph** 的智能 **AI Agent** 项目，集成 **LangSmith** 全链路追踪，支持 **Agent 工具调用**自主决策。

---

## 一、项目改造概览

### 改造前（RAG + LCEL）

```
用户问题 → 查询改写 → 混合检索 → 重排 → LLM 生成 → 回答
（固定流水线，每次都执行全部步骤）
```

### 改造后（AI Agent + LangGraph）

```
用户问题 → Agent(LLM 决策) → [工具调用] → Agent(再决策) → ... → 最终回答
（动态流程，LLM 根据意图自主选择工具）
```

### 四大改造点

| 改造点 | 改造前 | 改造后 |
|--------|--------|--------|
| **工作流引擎** | LCEL Chain（固定流水线） | LangGraph StateGraph（状态机 + 条件路由） |
| **决策方式** | 代码硬编码流程 | LLM 自主决策调用哪些工具 |
| **可观测性** | 仅日志 | LangSmith 全链路追踪 |
| **能力扩展** | 仅 RAG 问答 | 8 个工具（RAG + 文档管理 + 评估 + 计算 + 历史） |

---

## 二、项目结构

```
agent_project/
├── config_data.py            # 全局配置（含 LangSmith、Agent 提示词）
├── rag_infra.py              # RAG 基础设施（Embedding/Qdrant/BM25/Rerank/Chat 模型单例）
├── file_history_store.py     # 历史记录存储（Redis + 文件降级）
├── evaluation.py             # RAG 评估模块（检索 + 生成 + 幻觉率）
├── main_agent.py             # FastAPI 服务入口（Agent 端点）
├── run_evaluation.py         # 独立评估脚本
├── tools/                    # Agent 工具包
│   ├── __init__.py           # 工具注册与导出
│   ├── rag_tools.py          # rag_search / rag_answer
│   ├── file_tools.py         # upload_document / list_documents
│   ├── eval_tools.py         # evaluate_rag
│   ├── history_tools.py      # get_chat_history / clear_history
│   └── utility_tools.py      # calculator
├── graph/                    # LangGraph 工作流
│   ├── __init__.py
│   ├── state.py              # AgentState 状态定义
│   ├── nodes.py              # agent_node / tools_node / should_continue
│   └── build.py              # 工作流构建与编译
├── requirements.txt
├── .env.example
└── README.md
```

---

## 三、核心架构

### 3.1 LangGraph 工作流

```
START
  │
  ▼
agent_node (LLM + 工具绑定，决策下一步)
  │
  ▼
should_continue? ──── 有 tool_calls ──→ tools_node (执行工具)
  │                                        │
  │ 无 tool_calls                          │
  ▼                                        │
END  ◄─────────────────────────────────────┘
     (工具执行后回到 agent_node 继续决策)
```


### 3.2 Agent 工具清单

| 工具名 | 功能 | 适用场景 |
|--------|------|----------|
| `rag_search` | 混合检索（向量+BM25+Rerank） | 查看知识库相关资料 |
| `rag_answer` | 检索+生成完整回答 | 基于知识库的问答 |
| `upload_document` | 上传文本到知识库 | 添加新知识 |
| `list_documents` | 列出知识库文档 | 查看已有资料 |
| `evaluate_rag` | RAG 质量评估 | 测试系统效果 |
| `get_chat_history` | 查询会话历史 | 回顾上下文 |
| `clear_history` | 清空会话历史 | 重置对话 |
| `calculator` | 数学计算 | 数值计算需求 |

### 3.3 LangSmith 集成

LangSmith 通过**环境变量**自动启用，无需修改业务代码：

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_xxx
LANGSMITH_PROJECT=rag-agent-demo
```

启用后，以下内容会自动上报到 LangSmith 面板：
- 每次 LLM 调用的 prompt、response、token 用量
- 每次工具调用的输入、输出、耗时
- LangGraph 每个节点的执行轨迹
- 完整的 Agent 决策链路（多轮工具调用可视化）

对于需要额外追踪的自定义函数，使用 `@traceable` 装饰器：

```python
from langsmith import traceable

@traceable(name="rag_search", run_type="retriever")
def _do_rag_search(query: str):
    ...
```

---

## 四、快速开始

### 4.1 环境准备

```bash
# 1. 安装依赖
cd agent_project
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY 和 LANGSMITH_API_KEY
```

### 4.2 启动服务

```bash
python main_agent.py
# 或
uvicorn main_agent:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health
- 工具列表：http://localhost:8000/tools
- LangSmith 面板：https://smith.langchain.com/

### 4.3 调用示例

#### 同步对话

```bash
curl -X POST http://localhost:8000/chat/sync -H "Content-Type: application/json" -d '{"input": "帮我检索一下关于尺码推荐的内容", "session_id": "test_001"}'
```

#### 流式对话（SSE）

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"input\":\"知识库里有哪些文档？\",\"session_id\":\"test_001\"}"
```

#### 上传文档

```bash
curl -X POST http://localhost:8000/upload -F "file=@your_document.txt"
```

#### 上传文本

```bash
curl -X POST http://localhost:8000/upload/text -H "Content-Type: application/json" -d '{"content": "这是要上传的文本内容", "source_name": "manual"}'
```

---

## 五、Agent 行为示例

### 示例 1：知识库问答

**用户**：帮我查一下尺码推荐的相关信息

**Agent 决策链路**（在 LangSmith 中可见）：
1. `agent_node` → LLM 判断需要检索 → 调用 `rag_search`
2. `tools_node` → 执行 `rag_search`，返回文档片段
3. `agent_node` → LLM 基于检索结果生成回答 → 无工具调用 → END

### 示例 2：多工具协作

**用户**：先上传这份资料，然后基于它回答我的问题

**Agent 决策链路**：
1. `agent_node` → 调用 `upload_document`
2. `tools_node` → 上传成功
3. `agent_node` → 调用 `rag_answer`
4. `tools_node` → 检索+生成
5. `agent_node` → 整合结果回答 → END

### 示例 3：系统评估

**用户**：帮我评估一下 RAG 系统的效果，这是评估数据集 [...]

**Agent 决策链路**：
1. `agent_node` → 调用 `evaluate_rag`
2. `tools_node` → 运行评估，返回指标
3. `agent_node` → 解读指标并给出建议 → END

---

## 六、与原 RAG 项目的对比

### 6.1 代码层面

| 模块 | 原 RAG 项目 | 新 Agent 项目 |
|------|------------|--------------|
| 主入口 | `main_qdrant.py` | `main_agent.py` |
| 工作流 | LCEL `conversation_chain` | LangGraph `agent_graph` |
| 检索逻辑 | 内嵌于 chain | 封装为 `rag_search` 工具 |
| 生成逻辑 | 内嵌于 chain | 封装为 `rag_answer` 工具 |
| 决策逻辑 | 无（固定流程） | LLM 自主决策 |
| 可观测性 | 仅日志 | LangSmith 全链路 |

### 6.2 能力层面

| 能力 | 原 RAG 项目 | 新 Agent 项目 |
|------|------------|--------------|
| 知识库问答 | ✅ | ✅ |
| 文档上传 | ✅ | ✅ |
| 系统评估 | ✅（独立脚本） | ✅（Agent 工具，可对话式触发） |
| 历史记录 | ✅ | ✅ |
| 数学计算 | ❌ | ✅ |
| 多工具协作 | ❌ | ✅ |
| 自主决策 | ❌ | ✅ |
| 全链路追踪 | ❌ | ✅（LangSmith） |

---

## 七、扩展指南

### 7.1 添加新工具

1. 在 `tools/` 下创建新文件或编辑现有文件
2. 使用 `@tool` 装饰器定义工具：

```python
from langchain_core.tools import tool

@tool
def my_new_tool(param: str) -> str:
    """工具描述（LLM 会根据此描述决定是否调用）"""
    # 实现逻辑
    return result
```

3. 在 `tools/__init__.py` 中注册：

```python
from .my_tools import my_new_tool
ALL_TOOLS.append(my_new_tool)
```

4. 重启服务，Agent 即可自动使用新工具

### 7.2 自定义 Agent 提示词

编辑 `config_data.py` 中的 `AGENT_SYSTEM_PROMPT`，调整 Agent 的行为策略。

### 7.3 接入语言图表持久化

如需语言图表原生的会话检查点（而非自定义历史记录),修改`graph/build.py`：

```
从langgraph.checkpoint.memory导入内存Saver

graph = workflow . compile(check pointer = memory saver())
```

调用时传入`线程id`：

```
config = { " configurable ":{ " thread _ id ":session _ id } }
graph.invoke(初始状态，配置=配置)
```

