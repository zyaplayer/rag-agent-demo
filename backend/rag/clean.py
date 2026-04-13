import re
from typing import Dict, List, Union

from bs4 import BeautifulSoup
from langchain_core.documents import Document


class Cleaner:
    def __init__(self, chunk_size=200, chunk_overlap=20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self, docs):
        """
        docs: List[Document]（来自 loader）
        return: List[Document]（清洗后的）
        """
        cleaned_docs = []

        for doc in docs:
            text = doc.page_content

            # 1. 清洗文本
            text = clean_text(text)

            # 2. 分句
            sentences = split_sentences(text)

            # 3. 分块
            chunks = chunk_text(
                sentences,
                self.chunk_size,
                self.chunk_overlap
            )

            # 4. 重新封装为 Document（保持 LangChain 兼容）
            for chunk in chunks:
                cleaned_docs.append(
                    Document(page_content=chunk, metadata=dict(doc.metadata))
                )

        return cleaned_docs



def clean_text(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\b[\w.-]+?@\w+?\.\w+?\b", "", text)
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9.,?!，。？！]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"[。！？.!?]", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(sentences: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            overlap_part = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = overlap_part + sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def process_data(
    data: Union[str, List[str]],
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 20,
    source: str = "unknown",
) -> List[Dict[str, str]]:
    texts = [data] if isinstance(data, str) else data
    results: List[Dict[str, str]] = []

    for text in texts:
        cleaned = clean_text(text)
        sentences = split_sentences(cleaned)
        chunks = chunk_text(sentences, chunk_size, chunk_overlap)
        for chunk in chunks:
            results.append({"text": chunk, "source": source})

    return results


def get_sample_data() -> List[str]:
    return [
        "Milvus 是一个面向向量检索场景的数据库，常用于 RAG 系统。",
        "FastAPI 适合构建 Python API，常与 Streamlit 前端联调。",
        "RAG 的核心流程包括加载文档、清洗切分、向量化、检索和生成。",
    ]
