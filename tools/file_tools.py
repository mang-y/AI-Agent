"""
文件管理工具
=============

提供知识库文档管理能力：
- upload_document: 上传文本内容到知识库（自动分块 + 向量化 + BM25 索引更新）
- list_documents:  列出知识库中已有的文档
"""

import logging
from langchain_core.tools import tool
from langchain_core.documents import Document

from rag_infra import get_vector_store, get_hybrid_retriever

logger = logging.getLogger(__name__)


@tool
def upload_document(content: str, source_name: str = "user_upload") -> str:
    """
    将文本内容上传到知识库。

    流程：按双换行分块 → 向量化 → 存入 Qdrant → 更新 BM25 索引。
    上传后即可被 rag_search / rag_answer 检索到。

    适用场景：
    - 用户想要添加新知识到知识库
    - 用户提供了新的文档内容希望被记住
    - 知识库需要补充资料以回答后续问题

    Args:
        content: 要上传的文本内容（支持多段落，会自动按空行分块）
        source_name: 文档来源标识（用于元数据，便于后续追溯），默认 "user_upload"

    Returns:
        上传结果，包含成功添加的块数
    """
    try:
        if not content or not content.strip():
            return "上传失败：内容为空"

        # 按双换行分块（与原项目保持一致）
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        if not chunks:
            return "上传失败：分块后无有效内容"

        documents = [
            Document(page_content=chunk, metadata={"source": source_name})
            for chunk in chunks
        ]

        # 存入 Qdrant
        vector_store = get_vector_store()
        vector_store.add_documents(documents)

        # 更新 BM25 索引
        hybrid_retriever = get_hybrid_retriever()
        hybrid_retriever.update_bm25_index(documents)

        logger.info(f"[upload_document] 成功上传 {len(chunks)} 个块，来源: {source_name}")
        return (f"✅ 上传成功！共添加 {len(chunks)} 个文档块到知识库。\n"
                f"来源标识: {source_name}\n"
                f"现在可以使用 rag_search 或 rag_answer 检索这些内容了。")
    except Exception as e:
        logger.error(f"upload_document 执行失败: {e}", exc_info=True)
        return f"上传失败: {str(e)}"


@tool
def list_documents(limit: int = 20) -> str:
    """
    列出知识库中已有的文档片段。

    从 Qdrant 向量库中读取并展示文档片段的预览内容与来源元数据，
    帮助用户了解知识库的当前覆盖范围。

    适用场景：
    - 用户想了解知识库里有哪些资料
    - 用户想确认之前上传的文档是否成功入库
    - 用户想检查知识库内容是否需要补充

    Args:
        limit: 返回的文档片段数量上限，默认 20

    Returns:
        文档片段列表（含来源与内容预览）
    """
    try:
        vector_store = get_vector_store()
        docs = []
        offset = None
        count = 0
        while count < limit:
            records, offset = vector_store.client.scroll(
                collection_name=vector_store.collection_name,
                limit=min(100, limit - count),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            for record in records:
                content = record.payload.get("page_content", "").strip()
                metadata = record.payload.get("metadata", {})
                if content:
                    docs.append(Document(page_content=content, metadata=metadata))
                    count += 1
                    if count >= limit:
                        break
            if offset is None:
                break

        if not docs:
            return "知识库当前为空。可以使用 upload_document 工具上传文档。"

        # 按来源分组统计
        source_count = {}
        for doc in docs:
            src = doc.metadata.get("source", "未知来源") if doc.metadata else "未知来源"
            source_count[src] = source_count.get(src, 0) + 1

        lines = [f"知识库中共有 {len(docs)} 个文档片段（仅展示前 {limit} 个）：\n"]
        lines.append("【按来源统计】")
        for src, cnt in source_count.items():
            lines.append(f"  - {src}: {cnt} 个片段")
        lines.append("\n【文档预览】")
        for i, doc in enumerate(docs[:limit], 1):
            source = doc.metadata.get("source", "未知来源") if doc.metadata else "未知来源"
            preview = doc.page_content[:100].replace("\n", " ")
            lines.append(f"{i}. [{source}] {preview}...")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"list_documents 执行失败: {e}", exc_info=True)
        return f"列出文档失败: {str(e)}"
