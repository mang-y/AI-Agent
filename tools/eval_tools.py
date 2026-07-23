"""
评估工具
=========

将 RAG 评估能力封装为 Agent 工具。
Agent 可以接收用户提供的评估数据集，运行端到端评估并返回指标报告。
"""

import json
import logging
from langchain_core.tools import tool

from evaluation import RAGEvaluator, EvalSample
from rag_infra import get_chat_model
from tools.rag_tools import _do_rag_search, _do_rag_answer
from rag_infra import format_document

logger = logging.getLogger(__name__)


def _retrieve_only(question: str):
    """评估用的检索函数：返回 Document 列表"""
    docs = _do_rag_search(question)
    return docs


def _generate_only(question: str):
    """评估用的生成函数：返回 (answer, context)"""
    answer, context = _do_rag_answer(question)
    return answer, context


@tool
def evaluate_rag(dataset_json: str) -> str:
    """
    对 RAG 系统进行端到端质量评估（检索指标 + 生成指标 + 幻觉率）。

    评估指标包括：
    - 检索：Hit Rate@K, MRR, Recall@K, Precision@K, NDCG@K, Context Precision
    - 生成：Faithfulness(忠实度), Answer Relevance(相关性), Answer Correctness(正确性)
    - 幻觉：Hallucination Rate = 1 - Faithfulness

    适用场景：
    - 用户想测试 RAG 系统的检索和生成质量
    - 用户提供了评估数据集（含问题、标准答案、相关文档）
    - 用户想量化系统改进效果

    Args:
        dataset_json: JSON 格式的评估数据集字符串，格式为：
            [
                {
                    "question": "问题文本",
                    "ground_truth_answer": "标准答案（可选）",
                    "relevant_doc_ids": ["相关文档ID（可选）"],
                    "relevant_doc_contents": ["相关文档内容片段（可选）"]
                },
                ...
            ]

    Returns:
        评估结果摘要，包含各项指标的数值
    """
    try:
        # 解析数据集
        try:
            dataset = json.loads(dataset_json) if isinstance(dataset_json, str) else dataset_json
        except json.JSONDecodeError as e:
            return f"数据集 JSON 解析失败: {str(e)}\n请确保传入合法的 JSON 字符串。"

        if not dataset or not isinstance(dataset, list):
            return "数据集为空或格式不正确，应为 JSON 数组。"

        # 构造评估样本
        samples = [
            EvalSample(
                question=item["question"],
                ground_truth_answer=item.get("ground_truth_answer", ""),
                relevant_doc_ids=item.get("relevant_doc_ids", []),
                relevant_doc_contents=item.get("relevant_doc_contents", []),
            )
            for item in dataset
        ]

        logger.info(f"[evaluate_rag] 开始评估，共 {len(samples)} 条样本")

        # 运行评估
        evaluator = RAGEvaluator(chat_model=get_chat_model())
        report = evaluator.evaluate(
            samples=samples,
            retrieve_fn=_retrieve_only,
            generate_fn=_generate_only,
        )

        # 格式化结果
        lines = ["📊 RAG 系统评估报告", "=" * 50, f"评估样本数: {report.total_samples}\n"]

        lines.append("【检索质量指标】")
        for k, v in report.retrieval_metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

        lines.append("\n【生成质量指标】")
        for k, v in report.generation_metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

        # 诊断建议
        lines.append("\n【诊断建议】")
        halluc = report.generation_metrics.get('hallucination_rate', 0)
        hit = report.retrieval_metrics.get('hit_rate_at_3', 0)
        faith = report.generation_metrics.get('faithfulness', 0)

        if halluc > 0.3:
            lines.append("  ⚠️ 幻觉率偏高，建议优化 Prompt 或检查检索质量")
        if hit < 0.6:
            lines.append("  ⚠️ 检索命中率偏低，建议调整 chunk 大小或增加 Rerank top_n")
        if faith > 0.8 and hit > 0.8:
            lines.append("  ✅ 系统整体表现良好")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"evaluate_rag 执行失败: {e}", exc_info=True)
        return f"评估失败: {str(e)}"
