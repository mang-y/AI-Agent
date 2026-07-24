"""
LangGraph 状态定义
==================

定义 Agent 工作流中各节点共享的状态结构。
使用 TypedDict 确保 LangGraph 能正确进行状态合并 (reducer)。

状态字段说明：
- messages:       对话消息列表（含用户、助手、工具调用消息），使用 add_messages reducer
- session_id:     当前会话 ID（用于历史记录管理）
- iteration:      当前迭代轮数（防止死循环）
- retrieved_docs: 检索到的文档（可选，供调试）
"""

from typing import Annotated, Optional, Any
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class AgentState(TypedDict):
    """Agent 工作流状态"""
    # 消息列表：使用 add_messages reducer 自动追加而非覆盖
    messages: Annotated[list[AnyMessage], add_messages]
    # 会话 ID（从 RunnableConfig 注入或由用户传入）
    session_id: str
    # 当前迭代轮数
    iteration: int
    # 检索到的文档（可选，用于调试与追踪）
    retrieved_docs: Optional[list[Document]]
    # 额外元数据（用于 LangSmith 追踪）
    metadata: Optional[dict[str, Any]]
    # 反思/自纠错：已执行的反思次数（0 = 尚未反思）
    reflection_count: int
