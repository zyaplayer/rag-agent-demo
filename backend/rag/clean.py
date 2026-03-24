import re
from typing import List, Union, Dict
from bs4 import BeautifulSoup

SAMPLE_DATA = """
<h1>项目概述</h1>
<p>这是关于 PyTorch 2.5+ 和大模型微调的实战课程。</p>
<p><b>PyTorch</b> 是一个强大的深度学习框架。</p>
<p>课程内容包括理论、实战项目，并紧跟最新技术，如 <a href="#">torch.compile</a>。</p>
<p>请注意：本课程适合具有一定理论基础但缺乏编程经验的学员。</p>
<p>联系方式：<a href="mailto:info@example.com">info@example.com</a></p>
"""


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
    chunk_size: int = 200,
    chunk_overlap: int = 20,
    source: str = "unknown",
) -> List[Dict[str, str]]:
    if isinstance(data, list):
        data = " ".join(data)

    cleaned_text = clean_text(data)
    sentences = split_sentences(cleaned_text)
    chunks = chunk_text(sentences, chunk_size, chunk_overlap)
    return [{"text": chunk, "source": source} for chunk in chunks]


def get_sample_data() -> str:
    return SAMPLE_DATA
