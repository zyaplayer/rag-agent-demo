from backend.rag.loader import DocumentLoader
from backend.rag.clean import Cleaner
from backend.rag.embedding import Embedder
from backend.rag.retriever import Retriever
from backend.rag.reranker import Reranker
from backend.rag.compressor import Compressor


class RAGPipeline:
    def __init__(self):
        self.loader = DocumentLoader()
        self.cleaner = Cleaner()
        self.embedder = Embedder()
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.compressor = Compressor()

    def ingest(self, file_path):
        docs = self.loader.load(file_path)
        docs = self.cleaner.process(docs)
        embedded_docs = self.embedder.embed(docs)
        self.retriever.add(embedded_docs)

    def query(self, query, top_k=3):
        docs = self.retriever.search(query, top_k=top_k)

        if self.reranker:
            docs = self.reranker.rerank(query, docs)

        if self.compressor:
            docs = self.compressor.compress(query, docs)

        return docs[:top_k]
