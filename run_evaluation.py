"""
RAG 评估脚本（Agent 版）
========================

独立运行 RAG 评估，无需启动 HTTP 服务。

使用方式：
    python run_evaluation.py --dataset eval_dataset_example.json --report ./eval_reports/report.json --mode full

依赖：
    - 需要与 main_agent.py 在同一目录
    - 需要 .env 中配置 DASHSCOPE_API_KEY
    - 需要 config_data.py 配置文件

注意：本脚本复用 Agent 项目中的 RAG 工具（rag_search / rag_answer），
     评估的是 Agent 内部使用的 RAG 流程，确保评估与实际使用一致。
"""

import argparse
import json
import os
import sys

# 确保能导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import RAGEvaluator, EvalSample
from rag_infra import get_chat_model
from tools.rag_tools import _do_rag_search, _do_rag_answer


def load_dataset(path: str):
    """加载评估数据集"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        EvalSample(
            question=item["question"],
            ground_truth_answer=item.get("ground_truth_answer", ""),
            relevant_doc_ids=item.get("relevant_doc_ids", []),
            relevant_doc_contents=item.get("relevant_doc_contents", []),
        )
        for item in data
    ]


def retrieve_only(question: str):
    """评估用检索函数：复用 Agent 的 rag_search 内部实现"""
    return _do_rag_search(question)


def generate_only(question: str):
    """评估用生成函数：复用 Agent 的 rag_answer 内部实现"""
    answer, context = _do_rag_answer(question)
    return answer, context


def main():
    parser = argparse.ArgumentParser(description="RAG 系统评估脚本（Agent 版）")
    parser.add_argument("--dataset", required=True, help="评估数据集 JSON 路径")
    parser.add_argument("--report", default="./eval_reports/report.json", help="评估报告保存路径")
    parser.add_argument(
        "--mode",
        choices=["full", "retrieval", "generation"],
        default="full",
        help="评估模式：full=完整, retrieval=仅检索, generation=仅生成",
    )

    args = parser.parse_args()

    # 加载数据集
    samples = load_dataset(args.dataset)
    print(f"已加载 {len(samples)} 条评估样本")

    evaluator = RAGEvaluator(chat_model=get_chat_model())

    if args.mode == "full":
        report = evaluator.evaluate(
            samples=samples,
            retrieve_fn=retrieve_only,
            generate_fn=generate_only,
            save_path=args.report,
        )
        print("\n" + "=" * 60)
        print(report.to_markdown())
    elif args.mode == "retrieval":
        results = evaluator.retrieval_evaluator.evaluate_batch(samples, retrieve_only)
        from evaluation import RetrievalEvaluator
        metrics = RetrievalEvaluator.aggregate(results)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.mode == "generation":
        results = evaluator.generation_evaluator.evaluate_batch(samples, generate_only)
        from evaluation import GenerationEvaluator
        metrics = GenerationEvaluator.aggregate(results)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
