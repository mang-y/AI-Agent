"""
基于文件的多会话历史记录存储
============================

为前端 ChatGPT 风格的多会话体验提供后端支持：
- 每个会话以独立 JSON 文件存储于 chat_history/ 目录
- 支持会话标题、创建时间、更新时间
- 支持列出全部会话、删除会话、重命名会话
- 兼容 LangChain BaseChatMessageHistory 接口（messages / add_messages / clear）

文件格式（每个 session 一个 JSON）:
{
    "session_id": "xxx",
    "title": "会话标题",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
    "messages": [
        {"type": "human", "content": "..."},
        {"type": "ai", "content": "..."},
        {"type": "tool", "content": "..."}
    ]
}
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Optional

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
#from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

import config_data as config

logger = logging.getLogger(__name__)

# 历史记录存储目录
HISTORY_DIR = config.HISTORY_FILE_DIR
os.makedirs(HISTORY_DIR, exist_ok=True)


# ============================================================
# 消息序列化 / 反序列化
# ============================================================
def _serialize_message(msg: BaseMessage) -> dict:
    """将 LangChain 消息对象序列化为 JSON 可存储的字典"""
    msg_type = "unknown"
    if isinstance(msg, HumanMessage):
        msg_type = "human"
    elif isinstance(msg, AIMessage):
        msg_type = "ai"
    elif isinstance(msg, SystemMessage):
        msg_type = "system"
    elif isinstance(msg, ToolMessage):
        msg_type = "tool"

    return {
        "type": msg_type,
        "content": msg.content if isinstance(msg.content, str) else str(msg.content),
    }


def _deserialize_message(data: dict) -> BaseMessage:
    """将字典反序列化为 LangChain 消息对象"""
    msg_type = data.get("type", "human")
    content = data.get("content", "")
    if msg_type == "human":
        return HumanMessage(content=content)
    elif msg_type == "ai":
        return AIMessage(content=content)
    elif msg_type == "system":
        return SystemMessage(content=content)
    elif msg_type == "tool":
        return ToolMessage(content=content, tool_call_id=data.get("tool_call_id", ""))
    return HumanMessage(content=content)


# ============================================================
# 会话文件路径
# ============================================================
def _session_file_path(session_id: str) -> str:
    """获取会话 JSON 文件路径"""
    # 防止路径穿越：只允许字母数字下划线减号
    safe_id = "".join(c for c in session_id if c.isalnum() or c in ("_", "-"))
    if not safe_id:
        safe_id = "default"
    return os.path.join(HISTORY_DIR, f"{safe_id}.json")


# ============================================================
# 兼容 LangChain ChatMessageHistory 的会话历史类
# ============================================================
#class FileChatMessageHistory(ChatMessageHistory):
class FileChatMessageHistory(BaseChatMessageHistory):
    """
    单会话历史记录，兼容 LangChain 接口。
    在 ChatMessageHistory 基础上增加 JSON 持久化 + 标题管理。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.file_path = _session_file_path(session_id)
        self._load()

    def _load(self):
        """从文件加载会话数据"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._messages = [_deserialize_message(m) for m in data.get("messages", [])]
                self._title = data.get("title", "")
                self._created_at = data.get("created_at", datetime.now().isoformat())
                self._updated_at = data.get("updated_at", self._created_at)
            except Exception as e:
                logger.error(f"加载会话 {self.session_id} 失败: {e}")
                self._messages = []
                self._title = ""
                self._created_at = datetime.now().isoformat()
                self._updated_at = self._created_at
        else:
            self._messages = []
            self._title = ""
            self._created_at = datetime.now().isoformat()
            self._updated_at = self._created_at

    def _save(self):
        """保存会话数据到文件"""
        self._updated_at = datetime.now().isoformat()
        data = {
            "session_id": self.session_id,
            "title": self._title,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "messages": [_serialize_message(m) for m in self._messages],
        }
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存会话 {self.session_id} 失败: {e}")

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """追加消息（兼容 LangChain 接口）"""
        self._messages.extend(messages)
        # 如果还没有标题，用第一条用户消息生成
        if not self._title:
            for m in messages:
                if isinstance(m, HumanMessage):
                    self._title = (m.content[:30] if isinstance(m.content, str) else "新会话")
                    break
            if not self._title:
                self._title = "新会话"
        self._save()

    def clear(self) -> None:
        """清空会话消息（保留会话文件本身）"""
        self._messages = []
        self._save()

    def delete(self) -> None:
        """彻底删除会话文件"""
        if os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
                logger.info(f"已删除会话文件: {self.file_path}")
            except Exception as e:
                logger.error(f"删除会话 {self.session_id} 失败: {e}")

    def rename(self, new_title: str) -> None:
        """重命名会话"""
        self._title = new_title.strip() or "未命名会话"
        self._save()

    @property
    def title(self) -> str:
        return self._title

    @property
    def created_at(self) -> str:
        return self._created_at

    @property
    def updated_at(self) -> str:
        return self._updated_at


# ============================================================
# 全局接口
# ============================================================
def get_history(session_id: str) -> FileChatMessageHistory:
    """获取指定会话的历史记录对象（单例式：每次返回新实例以读取最新文件）"""
    return FileChatMessageHistory(session_id)


def create_session(title: str = "") -> dict:
    """创建新会话，返回会话信息"""
    session_id = f"session_{uuid.uuid4().hex[:12]}"
    history = get_history(session_id)
    if title:
        history.rename(title)
    else:
        # 写入空标题占位，确保文件存在
        history._title = "新会话"
        history._save()
    return {
        "session_id": session_id,
        "title": history.title,
        "created_at": history.created_at,
        "updated_at": history.updated_at,
        "message_count": 0,
    }


def list_sessions() -> list[dict]:
    """列出全部会话（按更新时间倒序）"""
    sessions = []
    if not os.path.exists(HISTORY_DIR):
        return sessions

    for filename in os.listdir(HISTORY_DIR):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(HISTORY_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "session_id": data.get("session_id", filename[:-5]),
                "title": data.get("title", "未命名会话"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except Exception as e:
            logger.error(f"读取会话文件 {filename} 失败: {e}")
            continue

    # 按更新时间倒序
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions


def delete_session(session_id: str) -> bool:
    """删除指定会话"""
    history = get_history(session_id)
    history.delete()
    return True


def rename_session(session_id: str, new_title: str) -> dict:
    """重命名会话"""
    history = get_history(session_id)
    history.rename(new_title)
    return {
        "session_id": session_id,
        "title": history.title,
        "updated_at": history.updated_at,
    }


def get_session_messages(session_id: str, limit: int = 100) -> list[dict]:
    """获取会话消息列表（按时间正序）"""
    history = get_history(session_id)
    messages = history.messages
    if limit and len(messages) > limit:
        messages = messages[-limit:]
    return [_serialize_message(m) for m in messages]


def auto_title_if_needed(session_id: str):
    """如果会话还没有有意义的标题，根据首条用户消息自动生成"""
    history = get_history(session_id)
    if history.title and history.title != "新会话":
        return
    for m in history.messages:
        if isinstance(m, HumanMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            title = content.strip().replace("\n", " ")[:30]
            if title:
                history.rename(title)
            return
