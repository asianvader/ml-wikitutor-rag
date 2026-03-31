"""Qdrant Cloud vector store abstraction.

All Qdrant interactions are contained here. retrieve.py and index_qdrant.py
talk to this module — swapping the vector DB in the future means touching only
this file and the indexing script.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Maps chunking strategy names to Qdrant collection names.
COLLECTION_NAMES = {
    "token":        "wiki_ml_token",
    "semantic":     "wiki_ml_semantic",
    "parent_child": "wiki_ml_parent_child",
}

EMBEDDING_DIM = 1536  # text-embedding-3-small output dimension


@dataclass
class Hit:
    """
    Uniform result object returned by QdrantVectorStore.search().

    ``score`` is cosine *distance* (1 - cosine_similarity), so lower = more
    similar.  Converting Qdrant's similarity scores to distances preserves
    full compatibility with the confidence heuristics in rag.py, which were
    originally tuned for this "lower is better" convention.
    """

    id: str
    score: float          # cosine distance; lower = more similar
    fields: dict = field(default_factory=dict)


class QdrantVectorStore:
    """Thin wrapper around QdrantClient for upsert and similarity search."""

    def __init__(self, url: str, api_key: str, collection_name: str) -> None:
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection_name = collection_name

    def create_collection_if_not_exists(self, vector_size: int = EMBEDDING_DIM) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection_name not in existing:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection: {self._collection_name}")
        else:
            print(f"Qdrant collection already exists: {self._collection_name}")

    def upsert(self, points: list[PointStruct]) -> None:
        """Upsert a batch of PointStructs into the collection."""
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(self, query_vector: list[float], k: int) -> list[Hit]:
        """
        Return the top-k most similar points as Hit objects.

        Qdrant returns cosine *similarity* scores in [0, 1] (higher = more
        similar).  We convert to cosine *distance* (1 - score) so the rest of
        the pipeline treats lower scores as better, matching the convention
        used throughout rag.py and config.py.
        """
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        return [
            Hit(
                id=str(r.id),
                score=1.0 - r.score,  # similarity → distance (lower = more similar)
                fields=r.payload or {},
            )
            for r in response.points
        ]
