from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentLoader:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, file_path: str):
        # 1. 加载文件
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        # 2. 切分文本
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        split_docs = splitter.split_documents(docs)

        return split_docs