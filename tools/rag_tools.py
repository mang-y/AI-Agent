"""
RAG 相关工具
=============

提供两个核心 RAG 工具供 Agent 调用：
- rag_search: 仅检索，返回文档片段（适合"查资料"类需求）
- rag_answer: 检索 + 生成完整回答（适合"问答"类需求）

两者都使用 LangSmith 的 @traceable 装饰器，
确保工具内部调用链路在 LangSmith 面板中可见。
"""

import logging
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from rag_infra import (
    hybrid_retrieve_with_rerank,
    rewrite_query,
    format_document,
    get_chat_model,
    SIMILARITY_K,
)

logger = logging.getLogger(__name__)


@traceable(name="rag_search", run_type="retriever")
def _do_rag_search(query: str, history_text: str = "无历史对话") -> list[Document]:
    """内部检索实现（被 @traceable 包裹，便于 LangSmith 追踪）"""
    # 1. 查询改写（消除代词指代）
    rewritten = rewrite_query(query, _parse_history_text(history_text))
    logger.info(f"[rag_search] 原始: {query} → 改写: {rewritten}")

    # 2. 混合检索 + 重排
    docs = hybrid_retrieve_with_rerank(rewritten)
    logger.info(f"[rag_search] 召回 {len(docs)} 篇文档")
    return docs


def _parse_history_text(history_text: str):
    """将简单的历史文本解析回消息对象（用于查询改写）"""
    if not history_text or history_text == "无历史对话":
        return []
    # 简化处理：直接返回空列表，让 rewrite_query 使用原问题
    return []


@tool
def rag_search(query: str) -> str:
    """
    在知识库中检索与查询相关的文档片段。

    使用混合检索策略（向量检索 + BM25 关键词检索 + RRF 融合 + DashScope Rerank 重排），
    返回最相关的文档片段及其元数据。

    适用场景：
    - 用户想查看知识库中有哪些相关资料
    - 用户想了解某主题在知识库中的覆盖情况
    - 作为 rag_answer 的前置步骤查看检索结果

    Args:
        query: 用户的查询问题（自然语言）

    Returns:
        检索到的文档片段列表（含内容与来源元数据），若无结果则返回提示信息
    """
    try:
        docs = _do_rag_search(query)
        if not docs:
            return "未在知识库中检索到相关文档。建议：1) 检查问题表述；2) 使用 upload_document 工具补充资料。"

        lines = [f"共检索到 {len(docs)} 篇相关文档：\n"]
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源") if doc.metadata else "未知来源"
            content = doc.page_content[:500]
            lines.append(f"【文档 {i}】来源: {source}")
            lines.append(f"内容: {content}")
            lines.append("-" * 60)
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"rag_search 执行失败: {e}", exc_info=True)
        return f"检索失败: {str(e)}"


@traceable(name="rag_answer", run_type="chain")
def _do_rag_answer(query: str) -> tuple[str, str]:
    """内部 RAG 问答实现（检索 + 生成）"""
    # 1. 检索
    docs = _do_rag_search(query)
    context = format_document(docs)

    # 2. 为空时直接拒绝，不经过 LLM
    if context == "无相关参考资料":
        return ("知识库中未找到与问题相关的资料。请补充上传相关文档后重试。", context)

    # 3. 生成（硬约束提示词，禁止添加参考资料以外的内容）
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个严格的RAG问答系统。请严格遵守以下规则：

规则1：严格基于参考资料回答，不得添加任何参考资料中没有的信息。
规则2：如果参考资料不足以完整回答问题，请明确指出哪些信息在资料中缺失。
规则3：如果参考资料完全不相关，请回答"根据现有资料，无法回答该问题"。

参考资料：{context}"""),
        ("user", "请回答用户提问：{input}"),
    ])
    chain = prompt | get_chat_model() | StrOutputParser()
    answer = chain.invoke({"context": context, "input": query})
    return answer, context


@tool
def rag_answer(query: str) -> str:
    """
    基于知识库内容生成完整回答（检索增强生成 RAG）。

    流程：查询改写 → 混合检索 → 重排 → LLM 基于检索结果生成回答。
    回答严格基于知识库内容，避免幻觉。

    适用场景：
    - 用户希望基于已上传的文档资料获得有据可依的回答
    - 用户询问知识库中可能包含的具体信息（产品规格、政策、FAQ 等）

    Args:
        query: 用户的查询问题（自然语言）

    Returns:
        基于知识库内容的回答。若知识库无相关内容，会明确告知。
    """
    try:
        answer, context = _do_rag_answer(query)
        if context == "无相关参考资料":
            return ("知识库中未找到相关资料，无法基于知识库回答。"
                    "建议使用 upload_document 工具上传相关文档，或尝试 rag_search 查看知识库现有内容。")
        return f"{answer}\n\n---\n(本回答基于知识库检索结果生成)"
    except Exception as e:
        logger.error(f"rag_answer 执行失败: {e}", exc_info=True)
        return f"问答失败: {str(e)}"
