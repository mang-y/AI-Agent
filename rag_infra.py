"""
RAG 基础设施模块（懒加载版）
============================

集中管理所有 RAG 相关组件，供 Agent 工具复用：
- DashScope Embedding
- Qdrant 向量库
- BM25 混合检索器（RRF 融合）
- DashScope Rerank 重排器
- ChatTongyi 对话模型
- 查询改写链

设计要点：
- 采用懒加载（Lazy Initialization）模式，组件在首次访问时才初始化
- 避免模块导入时就连接外部服务，提升启动速度与容错性
- 所有组件作为单例存在，全局共享同一实例
"""

import os
import logging
import functools
import jieba

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from rank_bm25 import BM25Okapi

import config_data as config
from dotenv import load_dotenv
import dashscope

load_dotenv()

# ============================================================
# 环境变量与全局常量
# ============================================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

COLLECTION_NAME = config.collection_name
PERSIST_DIRECTORY = config.persist_directory
EMBEDDING_MODEL = config.embedding_model_name
CHAT_MODEL = config.chat_model_name
RERANK_MODEL = config.rerank_model_name
RECALL_K = config.RECALL_K
SIMILARITY_K = config.SIMILARITY_K
RRF_K = config.RRF_K

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 懒加载单例缓存
# ============================================================
_instances = {}


def _get_or_create(key: str, factory):
    """获取或创建单例实例"""
    if key not in _instances:
        _instances[key] = factory()
    return _instances[key]


# ============================================================
# Embedding（懒加载）
# ============================================================
def get_embedding():
    return _get_or_create("embedding", lambda: (
        logger.info("正在初始化 DashScope Embedding...") or
        DashScopeEmbeddings(model=EMBEDDING_MODEL)
    ))


# ============================================================
# Qdrant 向量库（懒加载）
# ============================================================
def get_qdrant_client():
    def _create():
        logger.info("正在初始化 Qdrant 向量库...")
        client = QdrantClient(path=PERSIST_DIRECTORY)
        try:
            client.get_collection(collection_name=COLLECTION_NAME)
        except Exception:
            logger.info(f"集合 {COLLECTION_NAME} 不存在，正在创建...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=config.embedding_dim, distance=Distance.COSINE
                ),
            )
        return client
    return _get_or_create("qdrant_client", _create)


def get_vector_store():
    def _create():
        return QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name=COLLECTION_NAME,
            embedding=get_embedding(),
        )
    return _get_or_create("vector_store", _create)


def get_base_retriever():
    def _create():
        return get_vector_store().as_retriever(search_kwargs={"k": RECALL_K})
    return _get_or_create("base_retriever", _create)


# ============================================================
# Rerank 重排器（懒加载）
# ============================================================
def get_rerank_compressor():
    return _get_or_create("rerank_compressor", lambda: (
        DashScopeRerank(model=RERANK_MODEL, top_n=SIMILARITY_K)
    ))


def get_compression_retriever():
    def _create():
        return ContextualCompressionRetriever(
            base_retriever=get_base_retriever(),
            base_compressor=get_rerank_compressor(),
        )
    return _get_or_create("compression_retriever", _create)


# ============================================================
# Chat 模型（懒加载）
# ============================================================
def get_chat_model():
    return _get_or_create("chat_model", lambda: ChatTongyi(model=CHAT_MODEL,streaming=True))


# ============================================================
# BM25 混合检索器
# ============================================================
class HybridRetriever:
    """向量检索 + BM25 关键词检索，使用 RRF 算法融合"""

    def __init__(self, vector_retriever, vector_store, k=10):
        self.vector_retriever = vector_retriever
        self.vector_store = vector_store
        self.k = k
        self.bm25 = None
        self.bm25_docs = []
        self._init_bm25_from_qdrant()

    def _init_bm25_from_qdrant(self):
        """从 Qdrant 中加载现有文档初始化 BM25"""
        logger.info("正在从 Qdrant 加载文档初始化 BM25 索引...")
        try:
            docs = []
            offset = None
            while True:
                records, offset = self.vector_store.client.scroll(
                    collection_name=self.vector_store.collection_name,
                    limit=100,
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
                if offset is None:
                    break

            self.bm25_docs = docs
            if docs:
                tokenized = [list(jieba.cut(doc.page_content)) for doc in docs]
                self.bm25 = BM25Okapi(tokenized)
                logger.info(f"BM25 索引初始化完成，共 {len(docs)} 篇文档")
            else:
                self.bm25 = None
                logger.info("Qdrant 中暂无文档，BM25 索引为空")
        except Exception as e:
            logger.error(f"BM25 索引初始化失败: {str(e)}")
            self.bm25 = None
            self.bm25_docs = []

    def update_bm25_index(self, new_docs: list[Document]):
        """增量更新 BM25 索引（上传新文档后调用）"""
        if not new_docs:
            return
        self.bm25_docs.extend(new_docs)
        tokenized = [list(jieba.cut(doc.page_content)) for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 索引已更新，当前共 {len(self.bm25_docs)} 篇文档")

    def bm25_search(self, query: str, k: int = 10) -> list[Document]:
        """BM25 关键词检索"""
        if not self.bm25 or not self.bm25_docs:
            return []
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.bm25_docs[i] for i in top_indices if scores[i] > 0]

    def vector_search(self, query: str, k: int = 10) -> list[Document]:
        """向量语义检索"""
        try:
            return self.vector_retriever.invoke(query)[:k]
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
            return []

    def rrf_fuse(self, query: str, vector_k: int = 10, bm25_k: int = 10) -> list[Document]:
        """
        RRF (Reciprocal Rank Fusion) 融合向量检索与 BM25 检索结果
        RRF_score = Σ 1/(k + rank_i)
        """
        vector_results = self.vector_search(query, k=vector_k)
        bm25_results = self.bm25_search(query, k=bm25_k)

        rrf_scores = {}
        all_docs = {}

        for rank, doc in enumerate(vector_results, 1):
            key = doc.page_content[:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank)
            all_docs[key] = doc

        for rank, doc in enumerate(bm25_results, 1):
            key = doc.page_content[:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank)
            if key not in all_docs:
                all_docs[key] = doc

        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [all_docs[key] for key in sorted_keys]


def get_hybrid_retriever():
    def _create():
        return HybridRetriever(
            get_base_retriever(),
            get_vector_store(),
            k=RECALL_K,
        )
    return _get_or_create("hybrid_retriever", _create)


# ============================================================
# 查询改写链（懒加载）
# ============================================================
def get_rewrite_chain():
    def _create():
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是查询改写专家。根据用户当前提问与完整对话历史，消除代词指代（它、这个、该产品等），生成一条完整、独立、无歧义、适合文档检索的问句。
要求：
1. 补全上下文缺失的主体，不能保留"它、该、这款"等指代；
2. 只输出改写后的完整问句，不要解释、不要多余文字；
3. 如果问题本身完整无指代，直接原样输出。

对话历史：
{history_text}
用户当前提问：{user_question}"""),
        ])
        return rewrite_prompt | get_chat_model() | StrOutputParser()
    return _get_or_create("rewrite_chain", _create)


# ============================================================
# 模块级懒加载属性（Python 3.7+ __getattr__）
# ============================================================
# 当外部代码执行 `from rag_infra import chat_model` 时，
# 会触发此函数，实现首次访问时才初始化对应组件。
_LAZY_NAMES = {
    "chat_model": get_chat_model,
    "embedding": get_embedding,
    "vector_store": get_vector_store,
    "qdrant_client": get_qdrant_client,
    "hybrid_retriever": get_hybrid_retriever,
    "rerank_compressor": get_rerank_compressor,
    "rewrite_chain": get_rewrite_chain,
}


def __getattr__(name):
    """模块级懒加载：首次访问 chat_model 等属性时才初始化"""
    if name in _LAZY_NAMES:
        return _LAZY_NAMES[name]()
    raise AttributeError(f"module 'rag_infra' has no attribute {name!r}")


# ============================================================
# 工具函数
# ============================================================
def format_history_for_rewrite(history_msgs):
    """将消息列表格式化为查询改写所需的文本"""
    lines = []
    for msg in history_msgs:
        if msg.type == "human":
            lines.append(f"用户：{msg.content}")
        else:
            lines.append(f"助手：{msg.content}")
    return "\n".join(lines) if lines else "无历史对话"


def format_document(docs: list[Document]) -> str:
    """将文档列表格式化为 Prompt 上下文文本"""
    if not docs:
        return "无相关参考资料"
    valid_docs = [doc for doc in docs if doc and doc.page_content]
    if not valid_docs:
        return "无相关参考资料"
    return "\n\n".join([
        f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}"
        for doc in valid_docs
    ])


def rewrite_query(question: str, history_msgs=None) -> str:
    """执行查询改写，失败时降级为原问题"""
    history_msgs = history_msgs or []
    history_text = format_history_for_rewrite(history_msgs)
    try:
        return get_rewrite_chain().invoke({
            "history_text": history_text,
            "user_question": question,
        })
    except Exception as e:
        logger.error(f"查询重写失败: {str(e)}")
        return question


def hybrid_retrieve_with_rerank(query: str) -> list[Document]:
    """
    完整的混合检索 + 重排流程：
    1. 向量检索 + BM25 检索
    2. RRF 融合
    3. DashScope Rerank 重排
    4. 降级处理
    """
    try:
        retriever = get_hybrid_retriever()
        raw_docs = retriever.rrf_fuse(query, vector_k=RECALL_K, bm25_k=RECALL_K)
        if not raw_docs:
            return []

        valid_docs = [d for d in raw_docs if d.page_content and len(d.page_content.strip()) > 2]
        if not valid_docs:
            return raw_docs[:SIMILARITY_K]

        try:
            reranked = get_rerank_compressor().compress_documents(valid_docs, query)
            return reranked if reranked else raw_docs[:SIMILARITY_K]
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            return raw_docs[:SIMILARITY_K]
    except Exception as e:
        logger.error(f"混合检索失败: {str(e)}")
        return []
