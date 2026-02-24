from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import zvec
from langchain_openai import OpenAIEmbeddings

ZVEC_PATH = "index/zvec_wiki_ml"
VECTOR_FIELD = "text_embedding"
EMBEDDING_MODEL = "text-embedding-3-small"

# Keep these as module-level singletons so Streamlit doesn't recreate them on every rerun
_emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
_col = zvec.open(path=ZVEC_PATH, option=zvec.CollectionOption(read_only=True, enable_mmap=True))


def search(query: str, k: int = 8):
    """Return top-k Zvec hits for a query."""
    qv = _emb.embed_query(query)
    hits = _col.query(
        vectors=zvec.VectorQuery(field_name=VECTOR_FIELD, vector=qv),
        topk=k,
    )
    return hits