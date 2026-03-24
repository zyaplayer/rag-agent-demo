from sentence_transformers import SentenceTransformer
from typing import List

# 全局加载（避免重复加载模型）
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    批量向量化（用于入库）
    """
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    单条向量化（用于检索）
    """
    model = get_model()
    embedding = model.encode([query], normalize_embeddings=True)[0]
    return embedding.tolist()
