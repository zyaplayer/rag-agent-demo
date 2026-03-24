from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pymilvus import utility

from backend.config import Config
from backend.rag.embedding import embed_texts
from backend.rag.milvus_stores import MilvusVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    score: float


def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] source={c.source} score={c.score:.4f}\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )


class RAGAgent:
    """最小可用 RAG Agent：Milvus 检索 + LLM 生成。"""

    def __init__(
        self,
        *,
        collection_name: str = Config.COLLECTION_NAME,
        top_k: int = 3,
        model: str = Config.MODEL_NAME,
        base_url: str = Config.BASE_URL,
        api_key: Optional[str] = Config.OPENAI_API_KEY,
    ) -> None:
        self.collection_name = collection_name
        self.top_k = top_k

        # 1) 连接 Milvus
        self.store = MilvusVectorStore(collection_name=collection_name)
        self.store.connect()

        # 2) 只打开并 load 现有集合（更安全）
        # 如果集合不存在，明确报错并提示先跑 init_db
        if not utility.has_collection(collection_name):
            raise RuntimeError(
                f"Milvus collection '{collection_name}' 不存在。"
                f"请先运行 rag/init_db.py 初始化入库。"
            )

        # create_collection 在“集合已存在”时会直接 self.collection=Collection(...) 并 return
        # dim 在此分支不会被用到，但函数签名需要传
        dim = len(embed_texts(["dim_probe"])[0])
        self.store.create_collection(dim=dim, drop_if_exists=False)
        self.store.load_collection()

        # 3) 初始化 LLM（DeepSeek OpenAI 兼容）
        self.llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.2,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = top_k or self.top_k
        docs: List[Document] = self.store.search(query, top_k=k)

        chunks: List[RetrievedChunk] = []
        for d in docs:
            chunks.append(
                RetrievedChunk(
                    text=d.page_content,
                    source=str(d.metadata.get("source", "")),
                    score=float(d.metadata.get("score", 0.0)),
                )
            )
        return chunks

    def answer(self, question: str, top_k: Optional[int] = None) -> Tuple[str, List[RetrievedChunk]]:
        chunks = self.retrieve(question, top_k=top_k)
        context = _format_context(chunks)

        prompt = (
            "你是一个严谨的中文RAG助手。请仅依据【上下文】回答；"
            "若上下文不足，请明确说明并指出需要补充哪些资料。\n\n"
            f"【上下文】\n{context}\n\n"
            f"【问题】\n{question}\n\n"
            "【回答】"
        )

        resp = self.llm.invoke(prompt)
        return str(resp.content), chunks


if __name__ == "__main__":
    agent = RAGAgent()
    ans, _chunks = agent.answer("PyTorch 最新技术有哪些？", top_k=3)
    print(ans)