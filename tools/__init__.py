"""
工具集合
========

汇总所有 Agent 可用工具，并导出 ALL_TOOLS 列表。
"""

from tools.file_tools import upload_document, list_documents
from tools.history_tools import get_chat_history, clear_history
from tools.rag_tools import rag_search, rag_answer
from tools.utility_tools import calculator

# Agent 可调用的全部工具
ALL_TOOLS = [
    rag_search,
    rag_answer,
    upload_document,
    list_documents,
    get_chat_history,
    clear_history,
    calculator,
]

__all__ = [
    "ALL_TOOLS",
    "rag_search",
    "rag_answer",
    "upload_document",
    "list_documents",
    "get_chat_history",
    "clear_history",
    "calculator",
]
