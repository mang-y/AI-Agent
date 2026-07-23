"""
LangGraph 工作流构建
=====================

将 agent_node、tools_node、should_continue 组装成完整的 LangGraph 工作流。

工作流结构：
    START → agent → should_continue?
                        ├─ "tools" → tools → agent (循环)
                        └─ "end"   → END

构建后的 graph 是一个标准的 LangChain Runnable，
支持 invoke / stream / astream，并自动被 LangSmith 追踪。
"""

import logging
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from graph.nodes import agent_node, tools_node, should_continue

logger = logging.getLogger(__name__)


def build_agent_graph():
    """
    构建 Agent 工作流图。

    Returns:
        编译后的 LangGraph Runnable，可直接调用 invoke/stream/astream
    """
    # 1. 创建状态图
    workflow = StateGraph(AgentState)

    # 2. 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    # 3. 设置入口
    workflow.set_entry_point("agent")

    # 4. 添加条件边：agent → should_continue → tools / END
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # 5. 添加工具回边：tools → agent（工具执行后回到 agent 决策）
    workflow.add_edge("tools", "agent")

    # 6. 编译图
    # 注意：这里不使用 checkpointer，历史记录由 agent_node 内部管理
    # 如需 LangGraph 原生持久化，可传入 MemorySaver 或 SqliteSaver
    graph = workflow.compile()

    logger.info("✅ LangGraph Agent 工作流构建完成")
    logger.info("   工作流: START → agent → (tools → agent)* → END")
    return graph


# 模块级单例：首次导入时构建，避免重复编译
agent_graph = build_agent_graph()
