"""
Parent-child chunking.

Strategy:
- Split text into parent chunks (~500 tokens) using the existing paragraph-aware chunker.
- For each parent, subdivide into smaller child chunks (~150 tokens, 20-token overlap).
- Each child stores its own text (indexed for retrieval) and the full parent text
  (returned to the LLM for richer context).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import tiktoken


@dataclass
class ParentChildChunk:
    text: str           # child text — embedded and indexed
    parent_text: str    # parent text — sent to the LLM
    token_count: int    # child token count
    chunk_index: int    # global child index across the article
    parent_index: int   # which parent chunk this child belongs to


def chunk_text_parent_child(
    text: str,
    *,
    parent_size: int = 500,
    child_size: int = 150,
    child_overlap: int = 20,
    encoding_name: str = "cl100k_base",
) -> List[ParentChildChunk]:
    """
    Split *text* into parent chunks, then subdivide each into child chunks.

    Parameters
    ----------
    parent_size   : Target token size for parent chunks (no overlap between parents).
    child_size    : Target token size for child chunks (indexed for retrieval).
    child_overlap : Token overlap between adjacent child chunks within a parent.
    """
    from src.chunk import chunk_text  # reuse existing paragraph-aware chunker

    text = (text or "").strip()
    if not text:
        return []

    enc = tiktoken.get_encoding(encoding_name)

    # Step 1: produce parent chunks (no overlap so parent boundaries are clean)
    parent_chunks = chunk_text(text, chunk_size=parent_size, overlap=0)

    result: List[ParentChildChunk] = []
    global_child_index = 0

    for parent_idx, parent_chunk in enumerate(parent_chunks):
        parent_text = parent_chunk.text
        parent_tokens = enc.encode(parent_text)

        # Step 2: slice parent tokens into child windows
        start = 0
        while start < len(parent_tokens):
            end = min(start + child_size, len(parent_tokens))
            child_tokens = parent_tokens[start:end]
            child_text = enc.decode(child_tokens)

            result.append(
                ParentChildChunk(
                    text=child_text,
                    parent_text=parent_text,
                    token_count=len(child_tokens),
                    chunk_index=global_child_index,
                    parent_index=parent_idx,
                )
            )
            global_child_index += 1

            if end == len(parent_tokens):
                break
            start = end - child_overlap

    return result
