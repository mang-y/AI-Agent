"""
pytest 共享 fixtures
===================
提供跨测试文件的通用 mock 对象和临时目录配置。
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Mock LLM 相关 fixtures
# ============================================================
@pytest.fixture
def mock_llm():
    """模拟一个 LLM 对象，返回预设响应"""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="这是模拟的 LLM 回答")
    llm.bind_tools.return_value = llm
    return llm


@pytest.fixture
def mock_llm_with_tool_calls():
    """模拟一个 LLM，返回包含 tool_calls 的响应"""
    llm = MagicMock()
    response = AIMessage(content="")
    response.tool_calls = [
        {
            "name": "rag_search",
            "args": {"query": "公司考勤制度"},
            "id": "call_1",
        }
    ]
    llm.invoke.return_value = response
    llm.bind_tools.return_value = llm
    return llm


# ============================================================
# Mock History fixtures
# ============================================================
@pytest.fixture
def mock_history():
    """模拟一个 FileChatMessageHistory 对象"""
    history = MagicMock()
    history.messages = [
        HumanMessage(content="你好"),
        AIMessage(content="你好，有什么可以帮你的？"),
    ]
    history.title = "测试会话"
    history.clear = MagicMock()
    return history


@pytest.fixture
def empty_mock_history():
    """模拟一个空的 FileChatMessageHistory 对象"""
    history = MagicMock()
    history.messages = []
    history.title = "新会话"
    return history


# ============================================================
# Mock Vector Store fixtures
# ============================================================
@pytest.fixture
def mock_vector_store():
    """模拟 Qdrant VectorStore"""
    store = MagicMock()
    store.add_documents = MagicMock()
    store.collection_name = "rag"

    # 模拟 scroll 返回的 records
    mock_record = MagicMock()
    mock_record.payload = {
        "page_content": "这是一段测试文档内容",
        "metadata": {"source": "test_doc.txt"},
    }
    store.client.scroll.return_value = ([mock_record], None)
    return store


@pytest.fixture
def mock_hybrid_retriever():
    """模拟 HybridRetriever"""
    retriever = MagicMock()
    retriever.update_bm25_index = MagicMock()
    retriever.rrf_fuse.return_value = [
        Document(page_content="检索到的文档片段", metadata={"source": "test.txt"})
    ]
    return retriever


# ============================================================
# Mock Documents fixtures
# ============================================================
@pytest.fixture
def sample_documents():
    """提供一组测试用的 Document 对象"""
    return [
        Document(page_content="公司员工手册第一章：总则", metadata={"source": "01_公司介绍.txt"}),
        Document(page_content="考勤制度：每日工作时间为 9:00-18:00", metadata={"source": "03_考勤与休假制度.txt"}),
        Document(page_content="财务报销需填写报销申请单", metadata={"source": "04_财务报销流程.txt"}),
    ]
