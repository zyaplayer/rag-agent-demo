"""
reranker.py

文档重排序模块（Reranker）

基于 CrossEncoder 对候选文档进行精排，
提升 Top-K 结果的相关性
"""

from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class Reranker:
    """
    基于 CrossEncoder 的重排序器
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        对文档进行重排序

        参数：
            query: 用户问题
            docs: 初步召回文档

        返回：
            排序后的文档列表
        """

        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked]