"""Index chunks into Qdrant Cloud.

Replaces index_zvec.py.  Reads chunks.jsonl, embeds with OpenAI, and upserts
into the appropriate Qdrant collection for the chosen chunking strategy.

Usage:
    uv run python -m src.index_qdrant --chunker token
    uv run python -m src.index_qdrant --chunker semantic
    uv run python -m src.index_qdrant --chunker parent_child

Requires QDRANT_URL and QDRANT_API_KEY in the environment (or .env file).
"""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import PointStruct

from src.vector_store import COLLECTION_NAMES, EMBEDDING_DIM, QdrantVectorStore

load_dotenv()

DEFAULT_CHUNKS_PATH = Path("data_processed/chunks.jsonl")
EMBEDDING_MODEL = "text-embedding-3-small"


def load_chunks(chunks_path: Path, limit: int | None = None):
    rows = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def embed_with_retry(emb, texts, retries=5):
    """Embed with retry logic to make indexing resilient to transient API errors."""
    for attempt in range(retries):
        try:
            return emb.embed_documents(texts)
        except Exception as e:
            wait = 2 ** attempt
            print(f"Embedding batch failed ({type(e).__name__}). Retrying in {wait}s...")
            time.sleep(wait)
    return emb.embed_documents(texts)


def main(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    chunker: str = "token",
    limit: int | None = None,
    batch_size: int = 64,
):
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing {chunks_path}. Run ingest_wiki_api first.")

    url = os.environ["QDRANT_URL"]
    api_key = os.environ["QDRANT_API_KEY"]
    collection_name = COLLECTION_NAMES[chunker]

    store = QdrantVectorStore(url=url, api_key=api_key, collection_name=collection_name)
    store.create_collection_if_not_exists(vector_size=EMBEDDING_DIM)

    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chunks = load_chunks(chunks_path, limit=limit)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    inserted = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [r["text"] for r in batch]
        vectors = embed_with_retry(emb, texts)

        points = [
            PointStruct(
                id=r["id"],
                vector=v,
                payload={
                    "text": r["text"],
                    "title": r.get("title"),
                    "section": r.get("section"),
                    "source_url": r.get("source_url"),
                    "chunk_id": str(r.get("chunk_index")),
                    "parent_text": r.get("parent_text"),
                },
            )
            for r, v in zip(batch, vectors)
        ]
        store.upsert(points)
        inserted += len(points)
        print(f"Inserted {inserted}/{len(chunks)}")

    print("Done indexing.")

    # Quick retrieval smoke test
    query = "Explain the difference between supervised and unsupervised learning."
    qv = emb.embed_query(query)
    hits = store.search(qv, k=5)
    print("\nQuery:", query)
    for h in hits:
        preview = (h.fields.get("text") or "")[:100].replace("\n", " ")
        print(f"  [{h.score:.3f}] {h.fields.get('title')} — {preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index chunks into Qdrant Cloud.")
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks JSONL (default: data_processed/chunks.jsonl).",
    )
    parser.add_argument(
        "--chunker",
        default="token",
        choices=list(COLLECTION_NAMES),
        help="Chunker strategy — determines the target Qdrant collection.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(chunks_path=args.chunks_file, chunker=args.chunker, limit=args.limit)
