from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import zvec
from langchain_openai import OpenAIEmbeddings
from src.config import EMBEDDING_MODEL

VECTOR_FIELD = "text_embedding"

ZVEC_PATHS = {
    "token":        "index/zvec_wiki_ml",
    "semantic":     "index/zvec_wiki_ml_semantic",
    "parent_child": "index/zvec_wiki_ml_parent_child",
}

# Keep these as module-level singletons so Streamlit doesn't recreate them on every rerun
_emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Lazy-load collections — only open an index when it is first requested
_collections: dict[str, zvec.Collection] = {}


def _get_collection(chunker: str) -> zvec.Collection:
    if chunker not in _collections:
        path = ZVEC_PATHS.get(chunker)
        if path is None:
            raise ValueError(
                f"Unknown chunker '{chunker}'. Choose from: {list(ZVEC_PATHS)}"
            )
        try:
            _collections[chunker] = zvec.open(
                path=path,
                option=zvec.CollectionOption(read_only=True, enable_mmap=True),
            )
        except Exception as exc:
            rebuild_scripts = {
                "token":        "bash scripts/rebuild_index.sh",
                "semantic":     "bash scripts/rebuild_index_semantic.sh",
                "parent_child": "bash scripts/rebuild_index_parent_child.sh",
            }
            hint = rebuild_scripts.get(chunker, "the corresponding rebuild script")
            raise RuntimeError(
                f"Could not open the '{chunker}' index at '{path}'. "
                f"Run `{hint}` first, then restart the app.\n"
                f"  Cause: {exc}"
            ) from exc
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
