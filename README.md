# 基于LangGraph的企业运维RAG智能Agent项目（LangGraph + LangSmith + RAG）

面向企业内部IT运维文档场景的私有知识库问答系统，解决人工运维答疑效率低、新人上手慢、文档分散检索难的问题。

> 基于 **LangGraph** 的智能 AI Agent 服务，集成 **RAG**、**文档管理**、**系统评估**、**多会话管理**等能力，支持 **LangSmith 全链路追踪**。

---

## 一、项目概览

### 核心架构

```
用户问题 → Agent(LLM 决策) → [工具调用] → Agent(再决策/反思) → ... → 最终回答
```

| 特性 | 说明 |
|------|------|
| **工作流引擎** | LangGraph StateGraph（状态机 + 条件路由 + 反思机制） |
| **决策方式** | LLM 自主决策调用哪些工具，支持多轮协作 |
| **可观测性** | LangSmith 全链路追踪（LLM 调用、工具执行、节点轨迹） |
| **检索能力** | 混合检索（向量 + BM25 + RRF 融合 + DashScope Rerank） |
| **自纠错** | 支持反思/自纠错机制，评估回答质量并迭代优化 |

---

## 二、项目结构

```
.
├── config_data.py            # 全局配置（模型、检索参数、Agent 提示词等）
├── rag_infra.py              # RAG 基础设施（Embedding/Qdrant/BM25/Rerank/Chat 模型单例）
├── file_history_store.py     # 历史记录存储（Redis + 文件降级）
├── evaluation.py             # RAG 评估模块（检索指标 + 生成指标 + 幻觉率）
├── main_agent.py             # FastAPI 服务入口（Agent 端点 + 前端静态托管）
├── run_evaluation.py         # 独立评估脚本
├── eval_dataset.json         # 评估数据集示例
├── qa_dataset.csv            # QA 数据集
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量模板
├── chat_history/             # 本地聊天记录存储目录
├── qdrant_data/              # Qdrant 向量库数据目录
├── static/                   # 前端静态文件
│   └── index.html           # 聊天界面
├── graph/                    # LangGraph 工作流
│   ├── __init__.py          # 导出 agent_graph
│   ├── state.py             # AgentState 状态定义
│   ├── nodes.py             # agent_node / tools_node / should_continue / reflection_node
│   └── build.py             # 工作流构建与编译
├── tools/                    # Agent 工具包
│   ├── __init__.py          # 工具注册与导出
│   ├── rag_tools.py         # rag_search / rag_answer
│   ├── file_tools.py        # upload_document / list_documents
│   ├── history_tools.py     # get_chat_history / clear_history
│   ├── utility_tools.py     # calculator
│   └── eval_tools.py        # evaluate_rag
└── tests/                    # 测试文件
    └── test_api.py          # API 接口测试
```

---

## 三、核心功能

### 3.1 Agent 工具清单

| 工具名 | 功能 | 适用场景 |
|--------|------|----------|
| `rag_search` | 混合检索（向量+BM25+Rerank） | 查看知识库相关资料 |
| `rag_answer` | 检索 + 生成完整回答 | 基于知识库的问答 |
| `upload_document` | 上传文本到知识库 | 添加新知识 |
| `list_documents` | 列出知识库文档 | 查看已有资料 |
| `evaluate_rag` | RAG 质量评估 | 测试系统效果 |
| `get_chat_history` | 查询会话历史 | 回顾上下文 |
| `clear_history` | 清空会话历史 | 重置对话 |
| `calculator` | 数学计算 | 数值计算需求 |

### 3.2 LangGraph 工作流

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
reflection_node (评估回答质量)              │
  │                                        │
  ▼                                        │
should_reflect? ──── 需要改进 ──→ agent (带反馈重试)
  │                                        │
  无需改进                                  │
  ▼                                        │
END ◄──────────────────────────────────——─—┘
```

**关键设计**：
- **工具调用循环**：工具执行后回到 agent_node 继续决策，实现多工具协作
- **反思机制**：当不再调用工具时，进入反思节点评估回答质量；不合格则注入反馈让 LLM 重新尝试
- **防死循环**：超过 `AGENT_MAX_ITERATIONS`（默认 10）强制生成最终回答

### 3.3 RAG 基础设施

| 组件 | 配置 |
|------|------|
| Embedding 模型 | DashScope text-embedding-v3（维度 1024） |
| 向量库 | Qdrant（余弦相似度） |
| 对话模型 | DashScope Qwen3-Max |
| Rerank 模型 | DashScope qwen-rerank |
| 混合检索 | 向量检索 + BM25（jieba 分词）+ RRF 融合 |
| 召回数量 | RECALL_K = 10（向量/BM25 各召回） |
| 重排数量 | SIMILARITY_K = 3（Rerank 后保留） |
| RRF 平滑常数 | RRF_K = 60 |

### 3.4 LangSmith 集成

通过**环境变量**自动启用，无需修改业务代码：

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_xxx
LANGSMITH_PROJECT=rag-agent-demo
```

启用后自动上报：
- 每次 LLM 调用的 prompt、response、token 用量
- 每次工具调用的输入、输出、耗时
- LangGraph 每个节点的执行轨迹
- 完整的 Agent 决策链路（多轮工具调用可视化）

---

## 四、快速开始

### 4.1 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY 和 LANGSMITH_API_KEY
```

### 4.2 上传初始数据

启动前需要先上传知识库文档：

```bash
# 方式 1：通过 API 上传
curl -X POST http://localhost:8000/upload/text -H "Content-Type: application/json" -d '{"content": "你的文档内容", "source_name": "manual"}'

# 方式 2：使用前端界面访问 http://localhost:8000
```

### 4.3 启动服务

```bash
# 方式 1：直接运行
python main_agent.py

# 方式 2：使用 uvicorn
uvicorn main_agent:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：
- **前端界面**：http://localhost:8000
- **API 文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/health
- **工具列表**：http://localhost:8000/tools
- **LangSmith 面板**：https://smith.langchain.com/

---

## 五、API 端点

### 5.1 对话接口

#### 流式对话（SSE）

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"input":"知识库里有哪些文档？","session_id":"test_001"}'
```

#### 同步对话

```bash
curl -X POST http://localhost:8000/chat/sync -H "Content-Type: application/json" -d '{"input":"帮我检索一下相关内容","session_id":"test_001"}'
```

### 5.2 会话管理

```bash
# 创建会话
curl -X POST http://localhost:8000/sessions -H "Content-Type: application/json" -d '{"title":"我的对话"}'

# 列出所有会话
curl http://localhost:8000/sessions

# 获取会话消息
curl http://localhost:8000/sessions/{session_id}/messages

# 删除会话
curl -X DELETE http://localhost:8000/sessions/{session_id}

# 重命名会话
curl -X PUT http://localhost:8000/sessions/{session_id} -H "Content-Type: application/json" -d '{"title":"新标题"}'
```

### 5.3 文档管理

```bash
# 上传文本
curl -X POST http://localhost:8000/upload/text -H "Content-Type: application/json" -d '{"content":"文档内容","source_name":"manual"}'

# 上传文件
curl -X POST http://localhost:8000/upload -F "file=@document.txt"

# 列出文档
curl http://localhost:8000/documents
```

### 5.4 评估接口

```bash
# 使用 evaluate_rag 工具进行端到端评估
curl -X POST http://localhost:8000/chat/sync -H "Content-Type: application/json" -d '{
    "input": "这是评估数据集 [...], 请评估系统效果",
    "session_id": "eval_session"
  }'
```

---

## 六、Agent 行为示例

### 示例 1：知识库问答

**用户**：帮我查一下相关知识

**Agent 决策链路**：
1. `agent_node` → LLM 判断需要检索 → 调用 `rag_search`
2. `tools_node` → 执行 `rag_search`，返回文档片段
3. `agent_node` → LLM 基于检索结果生成回答 → 无工具调用
4. `reflection_node` → 评估回答质量 → END

### 示例 2：多工具协作

**用户**：先看看有什么文档，然后回答我一个问题

**Agent 决策链路**：
1. `agent_node` → 调用 `list_documents`
2. `tools_node` → 返回文档列表
3. `agent_node` → 调用 `rag_answer`
4. `tools_node` → 检索 + 生成
5. `reflection_node` → 评估 → END

### 示例 3：自纠错机制

**用户**：解释一下这个概念

**Agent 决策链路**：
1. `agent_node` → 首次回答
2. `reflection_node` → 评估发现不充分 → 标记需改进
3. `agent_node` → 带反馈重新生成
4. `reflection_node` → 评估合格 → END

---

## 七、配置说明

### 7.1 模型相关 (`config_data.py`)

```python
embedding_model_name = "text-embedding-v3"  # Embedding 模型
chat_model_name = "qwen3-max"               # 对话模型
rerank_model_name = "qwen-rerank"           # Rerank 模型
embedding_dim = 1024                        # 向量维度
```

### 7.2 检索参数

```python
RECALL_K = 10      # 向量 + BM25 各自召回数
SIMILARITY_K = 3   # Rerank 后保留数
RRF_K = 60         # RRF 融合算法平滑常数
```

### 7.3 Agent 行为

```python
AGENT_MAX_ITERATIONS = 10  # 单次会话最大工具调用轮数
MAX_REFLECTIONS = 2        # 最大反思次数
```

### 7.4 历史记录

```python
HISTORY_BACKEND = "redis"  # "redis" 或 "file"
HISTORY_FILE_DIR = "./chat_history"
MAX_HISTORY_MESSAGES = 20  # 单会话语义上限
```

---

## 八、常见问题

### Q1: LangSmith 没有数据？

- 检查 `.env` 中 `LANGSMITH_TRACING=true`
- 确认 `LANGSMITH_API_KEY` 已正确设置
- 访问 https://smith.langchain.com/ 查看对应项目

### Q2: Redis 连接失败？

项目会自动降级为文件存储，不影响使用。如需启用 Redis：
- 确认 Redis 服务已启动
- 检查 `.env` 中的 Redis 配置
- 确保 `HISTORY_BACKEND = "redis"`

### Q3: Agent 一直调用工具不结束？

- 检查 `AGENT_MAX_ITERATIONS`（默认 10，防止死循环）
- 优化 `AGENT_SYSTEM_PROMPT`，明确告知何时应直接回答
- 查看 LangSmith 链路排查卡住的工具

### Q4: 如何扩展新功能？

**添加工具**：
1. 在 `tools/` 下创建新文件或编辑现有文件
2. 使用 `@tool` 装饰器定义工具
3. 在 `tools/__init__.py` 中注册到 `ALL_TOOLS`
4. 重启服务即可

**自定义 Agent 提示词**：
编辑 `config_data.py` 中的 `AGENT_SYSTEM_PROMPT`

---

## 九、技术栈

| 类别 | 技术选型 |
|------|----------|
| **框架** | LangGraph, FastAPI |
| **LLM** | DashScope (通义千问) |
| **Embedding** | DashScope text-embedding-v3 |
| **Rerank** | DashScope qwen-rerank |
| **向量库** | Qdrant |
| **可观测性** | LangSmith |
| **历史存储** | Redis / 文件降级 |
| **前端** | HTML + CSS + JavaScript |

---

## 十、与原项目的区别

| 方面 | 原项目 | 当前项目 |
|------|--------|----------|
| **工作流** | LCEL Chain（固定流水线） | LangGraph StateGraph（动态决策） |
| **决策** | 代码硬编码流程 | LLM 自主决策 |
| **工具** | 无 | 8 个工具（RAG + 管理 + 评估 + 计算） |
| **反思** | 无 | 支持自纠错机制 |
| **追踪** | 仅日志 | LangSmith 全链路 |
| **会话** | 简单历史记录 | 多会话管理（创建/切换/删除） |



