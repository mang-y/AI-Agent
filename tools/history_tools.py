"""
历史记录工具
=============

提供会话历史记录管理能力：
- get_chat_history: 查询当前会话的历史对话
- clear_history:    清空当前会话的历史记录

注意：这两个工具需要从 RunnableConfig 中获取 session_id，
因此使用 InjectedToolArg 注入配置。
"""

import logging
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.runnables import RunnableConfig

from file_history_store import get_history

logger = logging.getLogger(__name__)


@tool
def get_chat_history(session_id: str, limit: int = 10) -> str:
    """
    获取指定会话的历史对话记录。

    从历史记录存储（Redis 或文件）中读取该会话的过往对话，
    帮助回顾上下文或确认对话状态。

    适用场景：
    - 用户想回顾之前聊过什么
    - Agent 需要确认上下文以理解当前问题
    - 用户想查看历史记录是否完整

    Args:
        session_id: 会话 ID（用于定位历史记录）
        limit: 返回的最近消息条数，默认 10

    Returns:
        历史对话记录（按时间从旧到新排列）
    """
    try:
        history = get_history(session_id)
        messages = history.messages

        if not messages:
            return f"会话 {session_id} 暂无历史记录。"

        # 取最近 limit 条
        recent = messages[-limit:] if len(messages) > limit else messages

        lines = [f"会话 {session_id} 的历史记录（最近 {len(recent)} 条）：\n"]
        for msg in recent:
            role = "用户" if msg.type == "human" else "助手"
            content = str(msg.content)[:200]
            lines.append(f"【{role}】{content}")
            lines.append("-" * 40)
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"get_chat_history 执行失败: {e}", exc_info=True)
        return f"获取历史记录失败: {str(e)}"


@tool
def clear_history(session_id: str) -> str:
    """
    清空指定会话的历史对话记录。

    删除该会话在存储中的所有历史消息，不可恢复。

    适用场景：
    - 用户想重新开始对话
    - 用户想清除敏感的对话内容
    - 测试时需要重置会话状态

    Args:
        session_id: 要清空的会话 ID

    Returns:
        操作结果
    """
    try:
        history = get_history(session_id)
        history.clear()
        logger.info(f"[clear_history] 已清空会话 {session_id} 的历史记录")
        return f"✅ 已成功清空会话 {session_id} 的历史记录。"
    except Exception as e:
        logger.error(f"clear_history 执行失败: {e}", exc_info=True)
        return f"清空历史记录失败: {str(e)}"
