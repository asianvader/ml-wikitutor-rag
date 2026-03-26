"""
Semantic chunking: splits text at points where the meaning shifts.

Algorithm (inspired by LangChain's SemanticChunker):
1. Split text into sentences.
2. Build "combined" sentences: each sentence merged with its neighbours
   in a sliding window so embeddings capture local context.
3. Embed all combined sentences in one batched call.
4. Compute cosine distance between adjacent embeddings.
5. Declare a breakpoint wherever the distance exceeds a threshold
   (computed as a percentile of all observed distances).
6. Group sentences between breakpoints into chunks, enforcing
   min/max size guards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import numpy as np
from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class Chunk:
    text: str
    token_count: int
    chunk_index: int


# ── Sentence splitting ────────────────────────────────────────────────────────

_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, also breaking on paragraph boundaries."""
    text = text.replace("\r\n", "\n").strip()
    raw: List[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        raw.extend(_SENT_RE.split(para))
    return [s.strip() for s in raw if s.strip()]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine_similarity(a: list, b: list) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


def _approx_tokens(text: str) -> int:
    """Rough estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


# ── Main chunker ──────────────────────────────────────────────────────────────

def chunk_text_semantic(
    text: str,
    *,
    embeddings: OpenAIEmbeddings | None = None,
    buffer_size: int = 1,
    breakpoint_percentile: float = 85.0,
    min_chunk_tokens: int = 60,
    max_chunk_tokens: int = 600,
) -> List[Chunk]:
    """
    Split *text* into semantically coherent chunks.

    Parameters
    ----------
    text                  : Input text.
    embeddings            : Reuse a shared OpenAIEmbeddings instance (avoids
                            re-creating it per article during bulk ingest).
    buffer_size           : Number of adjacent sentences to merge on each side
                            when building the embedding window (1 = triplet).
    breakpoint_percentile : Only distances above this percentile become
                            breakpoints. Higher → fewer, larger chunks.
    min_chunk_tokens      : Skip a breakpoint if the current chunk is still
                            smaller than this threshold.
    max_chunk_tokens      : Force a break even without a semantic signal if
                            the chunk grows beyond this limit.
    """
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    text = (text or "").strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    if len(sentences) == 1:
        return [Chunk(
            text=sentences[0],
            token_count=_approx_tokens(sentences[0]),
            chunk_index=0,
        )]

    # Build combined (windowed) sentences for contextual embeddings.
    # Each position i is represented by sentences[i-buffer:i+buffer+1] joined,
    # so the embedding captures surrounding context rather than one sentence alone.
    combined = []
    for i in range(len(sentences)):
        lo = max(0, i - buffer_size)
        hi = min(len(sentences), i + buffer_size + 1)
        combined.append(" ".join(sentences[lo:hi]))

    # Embed in one batch
    vecs = embeddings.embed_documents(combined)

    # Cosine distance between every adjacent pair of combined sentences
    distances = [
        1.0 - _cosine_similarity(vecs[i], vecs[i + 1])
        for i in range(len(vecs) - 1)
    ]

    # Percentile threshold → breakpoint indices
    threshold = float(np.percentile(distances, breakpoint_percentile))
    breakpoint_idxs = {i for i, d in enumerate(distances) if d >= threshold}

    # Group sentences into chunks
    chunks: List[Chunk] = []
    current: List[str] = []
    chunk_index = 0

    for i, sent in enumerate(sentences):
        current.append(sent)
        approx = _approx_tokens(" ".join(current))

        at_break = i in breakpoint_idxs
        too_long = approx >= max_chunk_tokens
        is_last = i == len(sentences) - 1

        # Flush if: meaningful semantic break, forced by size, or end of text
        should_flush = (at_break and approx >= min_chunk_tokens) or too_long or is_last

        if should_flush:
            chunk_text = " ".join(current).strip()
            chunks.append(Chunk(
                text=chunk_text,
                token_count=_approx_tokens(chunk_text),
                chunk_index=chunk_index,
            ))
            chunk_index += 1
            current = []

    return chunks
