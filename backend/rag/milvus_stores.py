from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_community.vectorstores import Milvus as LangChainMilvus
from langchain.embeddings.base import Embeddings
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from backend.config import Config
from backend.rag.embedding import embed_query, embed_texts


class SBERTEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return embed_query(text)


class MilvusVectorStore:
    def __init__(
        self,
        collection_name: str = Config.COLLECTION_NAME,
        host: str = Config.MILVUS_HOST,
        port: str = Config.MILVUS_PORT,
    ) -> None:
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.collection: Optional[Collection] = None
        self.embeddings = SBERTEmbeddings()

    def connect(self, alias: str = "default") -> None:
        connections.connect(alias=alias, host=self.host, port=self.port)

    def create_collection(self, dim: int, drop_if_exists: bool = False) -> None:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        ]
        schema = CollectionSchema(fields, description="RAG knowledge base collection")

        if utility.has_collection(self.collection_name):
            if drop_if_exists:
                Collection(self.collection_name).drop()
            else:
                self.collection = Collection(self.collection_name)
                return

        self.collection = Collection(name=self.collection_name, schema=schema)

    def create_index(self) -> None:
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection first.")
        if self.collection.indexes:
            return

        index_params = {
            "metric_type": "IP",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 200},
        }
        self.collection.create_index(field_name="vector", index_params=index_params)

    def load_collection(self) -> None:
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection first.")
        self.collection.load()

    def release_collection(self) -> None:
        if self.collection is not None:
            self.collection.release()

    def insert_data(self, chunks: Sequence[Dict[str, str]]) -> int:
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection first.")

        texts = [c["text"] for c in chunks]
        sources = [c.get("source", "") for c in chunks]
        vectors = embed_texts(texts)

        entities = [vectors, texts, sources]
        mr = self.collection.insert(entities)
        self.collection.flush()
        return mr.insert_count

    def search(self, query_text: str, top_k: int = 3) -> List[Document]:
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection first.")

        q = embed_query(query_text)
        results = self.collection.search(
            data=[q],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["text", "source"],
        )

        docs: List[Document] = []
        for hit in results[0]:
            docs.append(
                Document(
                    page_content=hit.entity.get("text") or "",
                    metadata={
                        "score": float(hit.distance),
                        "source": hit.entity.get("source") or "",
                    },
                )
            )
        return docs

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        if search_kwargs is None:
            search_kwargs = {"k": 3}

        vs = LangChainMilvus(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            connection_args={"host": self.host, "port": self.port},
        )
        return vs.as_retriever(search_kwargs=search_kwargs)


def init_milvus(chunks: Sequence[Dict[str, str]], drop_if_exists: bool = False) -> MilvusVectorStore:
    if not chunks:
        raise ValueError("chunks is empty")

    sample_vec = embed_texts([chunks[0]["text"]])[0]
    dim = len(sample_vec)

    store = MilvusVectorStore()
    store.connect()
    store.create_collection(dim=dim, drop_if_exists=drop_if_exists)
    store.create_index()
    store.insert_data(chunks)
    store.load_collection()
    return store
