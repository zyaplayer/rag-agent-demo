from sentence_transformers import SentenceTransformer
from typing import List
from langchain.schema import Document

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

class Embedder:
    def embed(self, docs: List[Document]) -> List[dict]:
        """
        docs → 向量
        return: List[{"text": ..., "vector": ..., "metadata": ...}]
        """
        model = get_model()

        texts = [doc.page_content for doc in docs]
        embeddings = model.encode(texts, normalize_embeddings=True)

        results = []
        for doc, emb in zip(docs, embeddings):
            results.append({
                "text": doc.page_content,
                "vector": emb.tolist(),
                "metadata": doc.metadata
            })

        return results

    def embed_query(self, query: str) -> List[float]:
        """
        query → 向量（用于检索）
        """
        model = get_model()
        embedding = model.encode([query], normalize_embeddings=True)[0]
        return embedding.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    model = get_model()
    embedding = model.encode([query], normalize_embeddings=True)[0]
    return embedding.tolist()