from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from backend.rag.clean import process_data, get_sample_data
from backend.rag.milvus_stores import MilvusVectorStore, init_milvus


@dataclass(frozen=True)
class InitDBOptions:
    chunk_size: int = 200
    chunk_overlap: int = 20
    source: str = "unknown"
    drop_if_exists: bool = False


def build_chunks(
    data: Union[str, List[str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    source: str,
) -> List[Dict[str, str]]:
    return process_data(
        data,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source=source,
    )


def init_db_from_data(
    data: Union[str, List[str]],
    *,
    options: InitDBOptions,
) -> MilvusVectorStore:
    chunks = build_chunks(
        data,
        chunk_size=options.chunk_size,
        chunk_overlap=options.chunk_overlap,
        source=options.source,
    )
    return init_milvus(chunks, drop_if_exists=options.drop_if_exists)


def init_db_from_file(
    path: Union[str, Path],
    *,
    options: InitDBOptions,
    encoding: str = "utf-8",
) -> MilvusVectorStore:
    p = Path(path)
    data = p.read_text(encoding=encoding, errors="ignore")
    source = options.source if options.source != "unknown" else str(p)
    opts = InitDBOptions(
        chunk_size=options.chunk_size,
        chunk_overlap=options.chunk_overlap,
        source=source,
        drop_if_exists=options.drop_if_exists,
    )
    return init_db_from_data(data, options=opts)


def init_db_demo(*, drop_if_exists: bool = True) -> MilvusVectorStore:
    data = get_sample_data()
    opts = InitDBOptions(source="demo", drop_if_exists=drop_if_exists)
    return init_db_from_data(data, options=opts)


if __name__ == "__main__":
    store = init_db_demo(drop_if_exists=True)
    print("Milvus init done:", store.collection_name)
