from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import zvec
from langchain_openai import OpenAIEmbeddings

VECTOR_FIELD = "text_embedding"
EMBEDDING_MODEL = "text-embedding-3-small"

ZVEC_PATHS = {
    "token":    "index/zvec_wiki_ml",
    "semantic": "index/zvec_wiki_ml_semantic",
}

# Keep these as module-level singletons so Streamlit doesn't recreate them on every rerun
_emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Lazy-load collections — only open an index when it is first requested
_collections: dict[str, zvec.Collection] = {}


def _get_collection(chunker: str) -> zvec.Collection:
    if chunker not in _collections:
        path = ZVEC_PATHS.get(chunker)
        if path is None:
            raise ValueError(f"Unknown chunker '{chunker}'. Choose from: {list(ZVEC_PATHS)}")
        _collections[chunker] = zvec.open(
            path=path,
            option=zvec.CollectionOption(read_only=True, enable_mmap=True),
        )
    return _collections[chunker]


def search(query: str, k: int = 8, chunker: str = "token"):
    """Return top-k Zvec hits for a query using the specified index."""
    qv = _emb.embed_query(query)
    col = _get_collection(chunker)
    hits = col.query(
        vectors=zvec.VectorQuery(field_name=VECTOR_FIELD, vector=qv),
        topk=k,
    )
    return hits
