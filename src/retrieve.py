from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from src.config import EMBEDDING_MODEL
from src.vector_store import QdrantVectorStore, COLLECTION_NAMES

# Module-level singletons — prevent Streamlit from recreating them on every rerun
_emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
_stores: dict[str, QdrantVectorStore] = {}


def _get_store(chunker: str) -> QdrantVectorStore:
    if chunker not in _stores:
        collection_name = COLLECTION_NAMES.get(chunker)
        if collection_name is None:
            raise ValueError(
                f"Unknown chunker '{chunker}'. Choose from: {list(COLLECTION_NAMES)}"
            )
        url = os.environ["QDRANT_URL"]
        api_key = os.environ["QDRANT_API_KEY"]
        _stores[chunker] = QdrantVectorStore(
            url=url, api_key=api_key, collection_name=collection_name
        )
    return _stores[chunker]


def search(query: str, k: int = 8, chunker: str = "token"):
    """Return top-k Qdrant hits for a query using the specified collection."""
    qv = _emb.embed_query(query)
    return _get_store(chunker).search(qv, k=k)
