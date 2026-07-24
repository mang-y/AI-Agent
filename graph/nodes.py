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

import json
import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from graph.state import AgentState
from rag_infra import get_chat_model
from config_data import (
    AGENT_SYSTEM_PROMPT,
    AGENT_MAX_ITERATIONS,
    MAX_REFLECTIONS,
    REFLECTION_SYSTEM_PROMPT,
)
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

    # 防止死循环：超过最大迭代数则强制让 LLM 基于已有信息给出最终回答
    if iteration >= AGENT_MAX_ITERATIONS:
        logger.warning(f"达到最大迭代数 {AGENT_MAX_ITERATIONS}，强制生成最终回答")
        history = get_history(session_id)
        history_msgs = history.messages[-20:] if history.messages else []
        force_messages = (
            [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
            + history_msgs
            + state["messages"]
            + [SystemMessage(
                content=f"(系统提示：已达到最大工具调用轮数 {AGENT_MAX_ITERATIONS}，"
                        f"请勿再调用任何工具，直接基于已有信息给出最终回答。)"
            )]
        )
        response = get_llm_with_tools().invoke(force_messages, config=config)
        # 强制清除可能残留的 tool_calls，确保工作流能正常结束
        if hasattr(response, "tool_calls"):
            response.tool_calls = []
        return {
            "messages": [response],
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


# ============================================================
# 反思/自纠错节点
# ============================================================
def reflection_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    反思节点：评估 Agent 的回答质量，不合格时注入反馈让其重新尝试。

    流程：
    1. 从 state 中获取最后一条 AI 回答
    2. 用 LLM 评估回答质量（结构化 JSON 输出）
    3. 如果合格 → 仅递增 reflection_count
    4. 如果不合格 → 注入带反馈的 AIMessage，让 agent_node 重新尝试

    Returns:
        更新后的状态（messages + reflection_count）
    """
    reflection_count = state.get("reflection_count", 0) + 1
    logger.info(f"[reflection_node] 第 {reflection_count} 次反思评估")

    # 获取最近的 AI 回答和用户问题
    messages = state["messages"]
    last_ai_msg = messages[-1]
    ai_content = last_ai_msg.content if hasattr(last_ai_msg, "content") else str(last_ai_msg)

    # 从消息列表中找到最近的用户问题
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not msg.additional_kwargs.get("is_reflection_feedback"):
            user_query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    try:
        # 用 LLM 评估回答质量
        eval_messages = [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题：{user_query}\n\nAI 回答：{ai_content}"),
        ]
        eval_response = get_chat_model().invoke(eval_messages, config=config)
        eval_text = eval_response.content if hasattr(eval_response, "content") else str(eval_response)

        # 解析 JSON 评估结果
        eval_result = _parse_reflection_result(eval_text)
        is_satisfactory = eval_result.get("is_satisfactory", True)
        issues = eval_result.get("issues", [])
        feedback = eval_result.get("feedback", "")

        logger.info(
            f"[reflection_node] 评估结果: "
            f"合格={is_satisfactory}, 问题数={len(issues)}, "
            f"反馈={feedback[:100] if feedback else '无'}"
        )

        if is_satisfactory:
            # 回答合格，无需反馈
            return {"reflection_count": reflection_count}

        # 如果已达到最大反思次数，不再注入反馈消息（否则用户会看到内部反馈）
        # 直接保留当前的 AI 回答作为最终输出
        if reflection_count >= MAX_REFLECTIONS:
            logger.info(
                f"[reflection_node] 已达最大反思次数 {MAX_REFLECTIONS}，"
                f"不再重试，保留当前回答"
            )
            return {"reflection_count": reflection_count}

        # 回答不合格：注入反馈消息，让 agent 重新尝试
        feedback_msg = AIMessage(
            content=(
                f"[反思反馈] 你的回答存在以下问题：\n"
                + "\n".join(f"- {issue}" for issue in issues)
                + f"\n改进建议：{feedback}\n"
                + "请根据以上反馈重新回答用户的问题。"
            ),
            additional_kwargs={"is_reflection_feedback": True},
        )
        return {
            "messages": [feedback_msg],
            "reflection_count": reflection_count,
        }

    except Exception as e:
        logger.error(f"[reflection_node] 反思评估失败: {e}，跳过反思")
        # 反思失败不影响正常流程，直接通过
        return {"reflection_count": reflection_count}


def _parse_reflection_result(text: str) -> dict:
    """
    解析反思 LLM 的 JSON 输出。

    尝试多种解析策略：
    1. 直接 JSON 解析
    2. 从文本中提取 JSON 块（```json ... ```）
    3. 回退为默认"合格"
    """
    # 策略 1：直接解析
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 策略 2：提取 JSON 块
    import re
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # 策略 3：尝试找到任何 JSON 对象
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    # 回退：视为合格（避免解析失败导致不必要的重试）
    logger.warning(f"[reflection_node] 无法解析评估结果，默认视为合格: {text[:200]}")
    return {"is_satisfactory": True, "issues": [], "feedback": ""}


def should_reflect(state: AgentState) -> str:
    """
    反思路由：决定是否需要回到 agent 重新尝试。

    判断逻辑：
    1. 如果已超过最大反思次数 → "end"
    2. 如果最后一条消息是反思反馈 → "reflect"（回到 agent）
    3. 否则 → "end"（回答合格，结束）

    Returns:
        "reflect" 或 "end"
    """
    reflection_count = state.get("reflection_count", 0)

    # 超过最大反思次数，强制结束
    if reflection_count >= MAX_REFLECTIONS:
        logger.info(f"[should_reflect] → end (已达最大反思次数 {MAX_REFLECTIONS})")
        return "end"

    # 检查最后一条消息是否包含反思反馈
    messages = state["messages"]
    if messages:
        last_msg = messages[-1]
        if (
            hasattr(last_msg, "additional_kwargs")
            and last_msg.additional_kwargs.get("is_reflection_feedback")
        ):
            logger.info(f"[should_reflect] → reflect (第 {reflection_count} 次反思，重新尝试)")
            return "reflect"

    # 回答合格，无需反思
    logger.info("[should_reflect] → end (回答质量合格)")
    return "end"
