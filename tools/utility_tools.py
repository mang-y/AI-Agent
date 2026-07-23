"""
通用工具
=========

提供与 RAG 无关的通用能力，增强 Agent 的实用性。
- calculator: 数学计算（支持基本四则运算与常用数学函数）
"""

import math
import logging
import ast
import operator
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# 安全的运算符白名单（防止 eval 执行任意代码）
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
}

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(expr: str) -> float:
    """安全地计算数学表达式（基于 AST 解析，仅允许白名单运算）"""
    node = ast.parse(expr, mode="eval").body
    return _eval_node(node)


def _eval_node(node):
    """递归求值 AST 节点"""
    if isinstance(node, ast.Constant):  # 数字
        return node.value
    elif isinstance(node, ast.BinOp):  # 二元运算
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    elif isinstance(node, ast.UnaryOp):  # 一元运算
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    elif isinstance(node, ast.Call):  # 函数调用
        if not isinstance(node.func, ast.Name):
            raise ValueError("仅支持简单函数调用")
        func_name = node.func.id
        func = _SAFE_FUNCTIONS.get(func_name)
        if func is None or not callable(func):
            raise ValueError(f"不支持的函数: {func_name}")
        args = [_eval_node(arg) for arg in node.args]
        return func(*args)
    elif isinstance(node, ast.Name):  # 变量（如 pi, e）
        val = _SAFE_FUNCTIONS.get(node.id)
        if val is None:
            raise ValueError(f"未知变量: {node.id}")
        return val
    else:
        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """
    数学计算器，支持基本四则运算和常用数学函数。

    支持的运算：
    - 四则运算: + - * / // %
    - 幂运算: ** 或 ^
    - 括号: ()
    - 函数: abs, round, min, max, sum, sqrt, log, log10, exp, sin, cos, tan
    - 常量: pi, e

    适用场景：
    - 用户需要进行数学计算
    - 需要精确数值结果而非估算
    - 涉及公式计算的问题

    Args:
        expression: 数学表达式字符串，例如 "2 + 3 * 4"、"sqrt(16) + pi"、"log(100)"

    Returns:
        计算结果
    """
    try:
        # 将 ^ 替换为 ** （用户习惯）
        expr = expression.replace("^", "**").strip()
        result = _safe_eval(expr)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        logger.error(f"calculator 执行失败: {e}")
        return (f"计算失败: {str(e)}\n"
                f"支持的表达式示例: '2 + 3 * 4', 'sqrt(16)', 'log(100)', 'pi * 2'")
