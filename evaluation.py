"""
RAG 系统评估模块
=================
提供三层评估能力：
1. 检索质量评估 (Hit Rate, MRR, Recall@K, Precision@K, NDCG@K, Context Precision)
2. 生成质量评估 (Faithfulness, Answer Relevance, Answer Correctness)
3. 幻觉检测 (Hallucination Rate = 1 - Faithfulness)

本模块在 Agent 项目中作为「评估工具」被调用，逻辑保持与原 RAG 项目一致。
"""

import json
import math
import re
import os
import logging
from typing import Callable, Optional
from dataclasses import dataclass, field, asdict
from statistics import mean

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================
# 数据结构
# ============================================================
@dataclass
class EvalSample:
    """单条评估样本"""
    question: str
    ground_truth_answer: str = ""
    relevant_doc_ids: list = field(default_factory=list)
    relevant_doc_contents: list = field(default_factory=list)


@dataclass
class RetrievalResult:
    """单条检索结果"""
    question: str
    retrieved_docs: list
    relevant_doc_ids: list
    hit_at_k: dict = field(default_factory=dict)
    mrr: float = 0.0
    recall_at_k: dict = field(default_factory=dict)
    precision_at_k: dict = field(default_factory=dict)
    ndcg_at_k: dict = field(default_factory=dict)
    context_precision: float = 0.0


@dataclass
class GenerationResult:
    """单条生成结果"""
    question: str
    answer: str
    ground_truth: str
    context: str
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    answer_correctness: float = 0.0
    hallucination_rate: float = 0.0


@dataclass
class EvalReport:
    """整体评估报告"""
    total_samples: int = 0
    retrieval_metrics: dict = field(default_factory=dict)
    generation_metrics: dict = field(default_factory=dict)
    details: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def to_markdown(self) -> str:
        lines = ["# RAG 系统评估报告\n"]
        lines.append(f"**评估样本数**: {self.total_samples}\n")

        lines.append("## 一、检索质量指标\n")
        lines.append("| 指标 | 值 |")
        lines.append("|------|----|")
        for k, v in self.retrieval_metrics.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")

        lines.append("\n## 二、生成质量指标\n")
        lines.append("| 指标 | 值 | 说明 |")
        lines.append("|------|----|------|")
        lines.append(f"| Faithfulness (忠实度) | {self.generation_metrics.get('faithfulness', 0):.4f} | 越高越好，1.0 表示完全无幻觉 |")
        lines.append(f"| Answer Relevance (答案相关性) | {self.generation_metrics.get('answer_relevance', 0):.4f} | 越高越好 |")
        lines.append(f"| Answer Correctness (答案正确性) | {self.generation_metrics.get('answer_correctness', 0):.4f} | 与标准答案对比 |")
        lines.append(f"| **Hallucination Rate (幻觉率)** | {self.generation_metrics.get('hallucination_rate', 0):.4f} | **越低越好**，= 1 - Faithfulness |")

        lines.append("\n## 三、综合诊断\n")
        faith = self.generation_metrics.get('faithfulness', 0)
        halluc = self.generation_metrics.get('hallucination_rate', 0)
        hit = self.retrieval_metrics.get('hit_rate_at_3', 0)

        if halluc > 0.3:
            lines.append("- ⚠️ **幻觉率偏高**：建议优化 Prompt，强化'仅基于上下文回答'约束，或检查检索质量。")
        if hit < 0.6:
            lines.append("- ⚠️ **检索命中率偏低**：建议调整 chunk 大小、Embedding 模型或增加 Rerank top_n。")
        if faith > 0.8 and hit > 0.8:
            lines.append("- ✅ 系统整体表现良好，检索与生成都较稳定。")

        return "\n".join(lines)


# ============================================================
# 检索质量评估器
# ============================================================
class RetrievalEvaluator:
    """检索质量评估器（传统 IR 指标 + LLM 评判）"""

    def __init__(self, chat_model=None, k_values: list = None):
        self.chat_model = chat_model
        self.k_values = k_values or [1, 3, 5, 10]

    @staticmethod
    def _doc_id(doc) -> str:
        content = (doc.page_content or "")[:50]
        source = doc.metadata.get("source", "") if doc.metadata else ""
        return f"{source}::{content}"

    @staticmethod
    def _is_relevant(doc, relevant_doc_ids: list, relevant_doc_contents: list) -> bool:
        doc_id = RetrievalEvaluator._doc_id(doc)
        if doc_id in relevant_doc_ids:
            return True
        content = (doc.page_content or "").strip()
        for ref in relevant_doc_contents:
            if ref and (ref in content or content in ref):
                return True
        return False

    def evaluate_sample(self, sample: EvalSample, retrieved_docs: list) -> RetrievalResult:
        result = RetrievalResult(
            question=sample.question,
            retrieved_docs=retrieved_docs,
            relevant_doc_ids=sample.relevant_doc_ids,
        )

        relevance_flags = [
            self._is_relevant(doc, sample.relevant_doc_ids, sample.relevant_doc_contents)
            for doc in retrieved_docs
        ]
        total_relevant = max(len(sample.relevant_doc_ids) + len(sample.relevant_doc_contents), 1)

        first_relevant_rank = None
        for k in self.k_values:
            top_k_flags = relevance_flags[:k]
            hit = any(top_k_flags)
            result.hit_at_k[k] = hit

            for i, flag in enumerate(top_k_flags):
                if flag:
                    if first_relevant_rank is None:
                        first_relevant_rank = i + 1
                    break
            if first_relevant_rank:
                result.mrr = 1.0 / first_relevant_rank

            retrieved_relevant = sum(top_k_flags)
            result.recall_at_k[k] = retrieved_relevant / total_relevant
            result.precision_at_k[k] = retrieved_relevant / max(len(top_k_flags), 1)
            result.ndcg_at_k[k] = self._ndcg_at_k(relevance_flags, k)

        if self.chat_model and retrieved_docs:
            result.context_precision = self._llm_context_precision(
                sample.question, retrieved_docs
            )

        return result

    @staticmethod
    def _ndcg_at_k(relevance_flags: list, k: int) -> float:
        def dcg(flags):
            return sum((1 if f else 0) / math.log2(i + 2) for i, f in enumerate(flags))

        dcg_k = dcg(relevance_flags[:k])
        ideal = sorted(relevance_flags, reverse=True)[:k]
        idcg_k = dcg(ideal)
        return dcg_k / idcg_k if idcg_k > 0 else 0.0

    def _llm_context_precision(self, question: str, docs: list) -> float:
        context = "\n\n".join([
            f"[{i+1}] {doc.page_content[:300]}"
            for i, doc in enumerate(docs[:5])
        ])
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个严格的检索质量评估专家。请评估以下检索到的文档片段对回答用户问题的帮助程度。
只输出一个 0 到 1 之间的数字（保留 2 位小数），不要任何解释：
- 1.0 = 完全相关，能直接回答问题
- 0.5 = 部分相关
- 0.0 = 完全不相关"""),
            ("user", f"问题：{question}\n\n检索到的文档：\n{context}")
        ])
        chain = prompt | self.chat_model | StrOutputParser()
        try:
            raw = chain.invoke({})
            score = float(re.search(r"\d+\.?\d*", raw).group())
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Context Precision LLM 评判失败: {e}")
            return 0.0

    def evaluate_batch(self, samples: list, retrieve_fn: Callable) -> list:
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"[检索评估] {i+1}/{len(samples)}: {sample.question[:50]}")
            try:
                docs = retrieve_fn(sample.question)
            except Exception as e:
                logger.error(f"检索失败: {e}")
                docs = []
            results.append(self.evaluate_sample(sample, docs))
        return results

    @staticmethod
    def aggregate(results: list) -> dict:
        n = len(results)
        if n == 0:
            return {}
        agg = {}
        k_values = [1, 3, 5, 10]
        for k in k_values:
            hits = [r.hit_at_k.get(k, False) for r in results]
            agg[f"hit_rate_at_{k}"] = sum(hits) / n
            agg[f"recall_at_{k}"] = mean([r.recall_at_k.get(k, 0) for r in results])
            agg[f"precision_at_{k}"] = mean([r.precision_at_k.get(k, 0) for r in results])
            agg[f"ndcg_at_{k}"] = mean([r.ndcg_at_k.get(k, 0) for r in results])
        agg["mrr"] = mean([r.mrr for r in results])
        agg["context_precision"] = mean([r.context_precision for r in results])
        return agg


# ============================================================
# 生成质量评估器
# ============================================================
class GenerationEvaluator:
    """生成质量评估器（LLM-as-Judge）"""

    def __init__(self, chat_model):
        self.chat_model = chat_model

    def evaluate_sample(self, question: str, answer: str, ground_truth: str, context: str) -> GenerationResult:
        result = GenerationResult(
            question=question, answer=answer,
            ground_truth=ground_truth, context=context,
        )
        result.faithfulness = self._faithfulness(answer, context)
        result.answer_relevance = self._answer_relevance(question, answer)
        if ground_truth:
            result.answer_correctness = self._answer_correctness(answer, ground_truth)
        result.hallucination_rate = 1.0 - result.faithfulness
        return result

    def _faithfulness(self, answer: str, context: str) -> float:
        if not answer.strip():
            return 0.0
        if not context.strip() or context == "无相关参考资料":
            return 0.0

        sentences = [s.strip() for s in re.split(r"[。.!?！？\n]+", answer) if len(s.strip()) > 3]
        if not sentences:
            return 0.0

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是严格的幻觉检测专家。请判断以下每个陈述句是否能从给定的参考资料中直接推导出来。
对每个句子输出 1（可推导）或 0（不可推导，即幻觉）。
输出格式：仅用逗号分隔的 0/1 序列，例如：1,0,1,1
不要任何解释。"""),
            ("user", """参考资料：
{context}

需要判断的陈述句（按顺序）：
{sentences}""")
        ])
        chain = prompt | self.chat_model | StrOutputParser()
        try:
            raw = chain.invoke({"context": context[:3000], "sentences": "\n".join(sentences)})
            flags = [int(x.strip()) for x in re.findall(r"[01]", raw)]
            if len(flags) < len(sentences):
                flags.extend([0] * (len(sentences) - len(flags)))
            supported = sum(flags[:len(sentences)])
            return supported / len(sentences)
        except Exception as e:
            logger.warning(f"Faithfulness 评估失败: {e}")
            return 0.0

    def _answer_relevance(self, question: str, answer: str) -> float:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """评估以下答案对用户问题的相关性。只输出 0 到 1 之间的数字（保留 2 位小数），不要解释：
- 1.0 = 完全切题，直接回答了问题
- 0.5 = 部分切题
- 0.0 = 完全不切题或答非所问"""),
            ("user", f"问题：{question}\n\n答案：{answer}")
        ])
        chain = prompt | self.chat_model | StrOutputParser()
        try:
            raw = chain.invoke({})
            return min(max(float(re.search(r"\d+\.?\d*", raw).group()), 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Answer Relevance 评估失败: {e}")
            return 0.0

    def _answer_correctness(self, answer: str, ground_truth: str) -> float:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是严格的评分专家。请对比学生答案与标准答案，给出语义一致度评分。
只输出 0 到 1 之间的数字（保留 2 位小数），不要解释：
- 1.0 = 完全一致
- 0.7 = 大部分一致，细节有差异
- 0.3 = 部分一致
- 0.0 = 完全错误"""),
            ("user", f"标准答案：{ground_truth}\n\n学生答案：{answer}")
        ])
        chain = prompt | self.chat_model | StrOutputParser()
        try:
            raw = chain.invoke({})
            return min(max(float(re.search(r"\d+\.?\d*", raw).group()), 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Answer Correctness 评估失败: {e}")
            return 0.0

    def evaluate_batch(self, samples: list, generate_fn: Callable) -> list:
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"[生成评估] {i+1}/{len(samples)}: {sample.question[:50]}")
            try:
                answer, context = generate_fn(sample.question)
            except Exception as e:
                logger.error(f"生成失败: {e}")
                answer, context = "", ""
            results.append(self.evaluate_sample(
                sample.question, answer, sample.ground_truth_answer, context
            ))
        return results

    @staticmethod
    def aggregate(results: list) -> dict:
        n = len(results)
        if n == 0:
            return {}
        return {
            "faithfulness": mean([r.faithfulness for r in results]),
            "answer_relevance": mean([r.answer_relevance for r in results]),
            "answer_correctness": mean([r.answer_correctness for r in results]),
            "hallucination_rate": mean([r.hallucination_rate for r in results]),
        }


# ============================================================
# RAG 综合评估器
# ============================================================
class RAGEvaluator:
    """RAG 端到端综合评估器"""

    def __init__(self, chat_model, k_values: list = None):
        self.retrieval_evaluator = RetrievalEvaluator(chat_model=chat_model, k_values=k_values)
        self.generation_evaluator = GenerationEvaluator(chat_model=chat_model)

    def load_dataset(self, dataset_path: str) -> list:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = []
        for item in data:
            samples.append(EvalSample(
                question=item["question"],
                ground_truth_answer=item.get("ground_truth_answer", ""),
                relevant_doc_ids=item.get("relevant_doc_ids", []),
                relevant_doc_contents=item.get("relevant_doc_contents", []),
            ))
        return samples

    def evaluate(
        self,
        samples: list,
        retrieve_fn: Callable,
        generate_fn: Callable,
        save_path: Optional[str] = None,
    ) -> EvalReport:
        report = EvalReport(total_samples=len(samples))
        logger.info(f"========== 开始 RAG 端到端评估，共 {len(samples)} 条样本 ==========")

        logger.info("【步骤 1/2】检索质量评估...")
        retrieval_results = self.retrieval_evaluator.evaluate_batch(samples, retrieve_fn)
        report.retrieval_metrics = RetrievalEvaluator.aggregate(retrieval_results)

        logger.info("【步骤 2/2】生成质量评估...")
        generation_results = self.generation_evaluator.evaluate_batch(samples, generate_fn)
        report.generation_metrics = GenerationEvaluator.aggregate(generation_results)

        for i, sample in enumerate(samples):
            report.details.append({
                "question": sample.question,
                "retrieval": {
                    "hit_at_3": retrieval_results[i].hit_at_k.get(3, False),
                    "mrr": retrieval_results[i].mrr,
                    "context_precision": retrieval_results[i].context_precision,
                },
                "generation": {
                    "answer": generation_results[i].answer[:200],
                    "faithfulness": generation_results[i].faithfulness,
                    "answer_relevance": generation_results[i].answer_relevance,
                    "answer_correctness": generation_results[i].answer_correctness,
                    "hallucination_rate": generation_results[i].hallucination_rate,
                }
            })

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
            md_path = save_path.replace(".json", ".md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())
            logger.info(f"评估报告已保存: {save_path} & {md_path}")

        logger.info("========== 评估完成 ==========")
        logger.info(f"检索 Hit@3: {report.retrieval_metrics.get('hit_rate_at_3', 0):.4f}")
        logger.info(f"幻觉率: {report.generation_metrics.get('hallucination_rate', 0):.4f}")
        return report
