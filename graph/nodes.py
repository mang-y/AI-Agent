"""
LangGraph 节点定义
==================

定义 Agent 工作流中的核心节点：
- agent_node:    调用 LLM 决策下一步（直接回答 or 调用工具）
- tools_node:    执行 LLM 选择的工具调用
- should_continue: 路由函数，决定是继续调用工具还是结束

每个节点接收 AgentState，返回需要更新的状态字段。
LangSmith 会自动追踪每个节点的执行（因为底层都是 LangChain Runnable）。
"""

import logging
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from graph.state import AgentState
from rag_infra import get_chat_model
from config_data import AGENT_SYSTEM_PROMPT, AGENT_MAX_ITERATIONS
from tools import ALL_TOOLS
from file_history_store import get_history

logger = logging.getLogger(__name__)


# ============================================================
# 绑定工具到 LLM（懒加载）
# ============================================================
# ChatTongyi 通过 bind_tools 支持函数调用（function calling）
# 绑定后，LLM 的输出可能包含 tool_calls 字段
# 使用懒加载避免模块导入时就初始化 LLM（需要 API Key）
_llm_with_tools = None


def get_llm_with_tools():
    """懒加载：首次调用时初始化 LLM 并绑定工具"""
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = get_chat_model().bind_tools(ALL_TOOLS)
        logger.info("✅ LLM 工具绑定完成")
    return _llm_with_tools


# ============================================================
# Agent 节点
# ============================================================
def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Agent 主节点：调用 LLM 进行决策。

    流程：
    1. 从 state 中获取历史消息 + 当前消息
    2. 注入系统提示词
    3. 调用绑定了工具的 LLM
    4. 返回 LLM 的响应消息（可能包含 tool_calls）

    LangSmith 会自动追踪此次 LLM 调用，
    在面板中可以看到完整的 prompt、tools 定义、response。
    """
    session_id = state.get("session_id", "default")
    iteration = state.get("iteration", 0)

    logger.info(f"[agent_node] session={session_id}, iteration={iteration}")

    # 防止死循环：超过最大迭代数则强制结束
    if iteration >= AGENT_MAX_ITERATIONS:
        logger.warning(f"达到最大迭代数 {AGENT_MAX_ITERATIONS}，强制结束")
        return {
            "messages": [HumanMessage(
                content=f"(系统提示：已达到最大工具调用轮数 {AGENT_MAX_ITERATIONS}，"
                        f"请基于已有信息给出最终回答。)"
            )],
            "iteration": iteration,
        }

    # 加载历史记录并拼接到消息列表前部
    history = get_history(session_id)
    history_msgs = history.messages[-20:] if history.messages else []  # 最近 20 条

    # 构建完整消息列表：系统提示 + 历史 + 当前消息
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + history_msgs + state["messages"]

    # 调用 LLM（LangSmith 自动追踪，懒加载 LLM）
    response = get_llm_with_tools().invoke(messages, config=config)

    logger.info(
        f"[agent_node] LLM 响应: "
        f"tool_calls={len(response.tool_calls) if hasattr(response, 'tool_calls') else 0}, "
        f"content_len={len(response.content) if response.content else 0}"
    )

    return {
        "messages": [response],
        "iteration": iteration + 1,
    }


# ============================================================
# 工具节点
# ============================================================
# 使用 LangGraph 内置的 ToolNode，自动处理 tool_calls 并执行对应工具
# ToolNode 会将每个 tool_call 的结果包装为 ToolMessage 返回
tool_node = ToolNode(ALL_TOOLS)


def tools_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    工具执行节点：执行 LLM 选择的工具调用。

    使用 LangGraph 内置的 ToolNode，它会：
    1. 从最后一条 AI 消息中提取 tool_calls
    2. 并行执行对应的工具函数
    3. 将每个工具的返回值包装为 ToolMessage

    LangSmith 会自动追踪每个工具的执行。
    """
    logger.info(f"[tools_node] 执行工具调用，当前消息数: {len(state['messages'])}")

    # 对于需要 session_id 的工具（get_chat_history, clear_history），
    # 我们在工具调用前注入 session_id
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        session_id = state.get("session_id", "default")
        # ToolNode 会自动将 tool_call 的 args 传给工具
        # 如果工具需要 session_id 但 LLM 没传，我们在这里补上
        for tc in last_msg.tool_calls:
            tool_name = tc.get("name", "")
            if tool_name in ("get_chat_history", "clear_history") and "session_id" not in tc.get("args", {}):
                tc["args"]["session_id"] = session_id
                logger.info(f"[tools_node] 为工具 {tool_name} 注入 session_id={session_id}")

    return tool_node.invoke(state, config=config)


# ============================================================
# 路由函数
# ============================================================
def should_continue(state: AgentState) -> str:
    """
    条件路由：决定下一步是调用工具还是结束。

    判断逻辑：
    - 如果最后一条消息包含 tool_calls → 返回 "tools"（继续执行工具）
    - 否则 → 返回 "end"（LLM 已给出最终回答）

    Returns:
        "tools" 或 "end"
    """
    last_message = state["messages"][-1]

    # 检查是否有工具调用
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"[should_continue] → tools (待调用 {len(last_message.tool_calls)} 个工具)")
        return "tools"

    logger.info("[should_continue] → end (无工具调用，结束)")
    return "end"
