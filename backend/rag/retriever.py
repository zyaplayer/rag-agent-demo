from typing import Iterable, List

from langchain_core.documents import Document
from pymilvus import Collection, utility

from backend.config import Config
from backend.rag.milvus_stores import MilvusVectorStore


class Retriever:
    def __init__(self, collection_name: str = Config.COLLECTION_NAME):
        self.store = MilvusVectorStore(collection_name=collection_name)
        self._connected = False
        self._loaded = False

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        self.store.connect()
        self._connected = True

    def _ensure_loaded(self) -> None:
        self._ensure_connected()
        if self._loaded:
            return
        if not utility.has_collection(self.store.collection_name):
            raise ValueError(
                f"Milvus collection '{self.store.collection_name}' does not exist. "
                "Run `python -m backend.rag.init_db` first."
            )
        self.store.collection = Collection(self.store.collection_name)
        self.store.load_collection()
        self._loaded = True

    def add(self, vectors: Iterable[dict], drop_if_exists: bool = False) -> int:
        items = list(vectors)
        if not items:
            return 0

        chunks = []
        for item in items:
            metadata = item.get("metadata") or {}
            chunks.append(
                {
                    "text": item["text"],
                    "source": metadata.get("source", ""),
                }
            )

        self._ensure_connected()
        dim = len(items[0]["vector"])
        self.store.create_collection(dim=dim, drop_if_exists=drop_if_exists)
        self.store.create_index()
        inserted = self.store.insert_data(chunks)
        self.store.load_collection()
        self._loaded = True
        return inserted

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        self._ensure_loaded()
        return self.store.search(query, top_k=top_k)
