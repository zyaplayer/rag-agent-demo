from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from backend.config import Config


class ContextCompressor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def compress(self, query: str, docs: List[Document]) -> List[Document]:
        compressed_docs = []

        for doc in docs:
            prompt = f"""
请从以下文档中提取与问题最相关的内容，
如果没有相关内容，请返回“无关”。

【问题】
{query}

【文档】
{doc.page_content}

【提取结果】
"""

            resp = self.llm.invoke(prompt)
            content = str(resp.content).strip()

            # 过滤无关内容
            if content and "无关" not in content:
                doc.page_content = content
                compressed_docs.append(doc)

        return compressed_docs


class Compressor:
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.impl = llm or self._build_default()

    def _build_default(self) -> Optional[ContextCompressor]:
        if not Config.OPENAI_API_KEY:
            return None

        llm = ChatOpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.BASE_URL,
            model=Config.MODEL_NAME,
            temperature=0,
        )
        return ContextCompressor(llm)

    def compress(self, query: str, docs: List[Document]) -> List[Document]:
        if self.impl is None:
            return docs
        return self.impl.compress(query, docs)
