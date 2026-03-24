from __future__ import annotations

import ast
import operator as op
from typing import Any

from langchain_core.tools import Tool

from backend.agent.agent import RAGAgent


# 懒加载：避免模块导入时就连接 Milvus / 初始化 LLM
_agent: RAGAgent | None = None


def _get_agent() -> RAGAgent:
    global _agent
    if _agent is None:
        _agent = RAGAgent()
    return _agent


def rag_tool(query: str) -> str:
    """
    RAG 工具：基于 Milvus + LLM 回答问题。
    """
    try:
        agent = _get_agent()
        answer, chunks = agent.answer(query, top_k=3)

        refs = []
        for i, c in enumerate(chunks, start=1):
            refs.append(f"[{i}] source={c.source} score={c.score:.4f}")

        if refs:
            return f"{answer}\n\n参考片段:\n" + "\n".join(refs)
        return answer
    except Exception as e:
        return f"RAG 工具执行失败: {e}"


# ---- 安全计算器（替代 eval） ----
_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_BINOPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        operand = _safe_eval(node.operand)
        return _ALLOWED_UNARYOPS[type(node.op)](operand)

    raise ValueError("仅支持数字与 + - * / ** % () 运算")


def calculator_tool(expression: str) -> str:
    """
    计算器工具：安全执行基础数学表达式。
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        # 整数结果去掉 .0
        if result.is_integer():
            return f"计算结果: {int(result)}"
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算失败: {e}"


tools = [
    Tool(
        name="RAGTool",
        func=rag_tool,
        description="使用知识库回答问题，适用于需要从知识库中检索事实信息的问题。",
    ),
    Tool(
        name="CalculatorTool",
        func=calculator_tool,
        description="执行数学计算，支持 + - * / ** % 和括号。",
    ),
]