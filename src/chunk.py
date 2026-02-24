"""
Chunking utilities.

Goal:
- Produce chunks around `chunk_size` tokens with `overlap` tokens.
- Keep chunk boundaries reasonably natural by first splitting into paragraphs,
  then packing paragraphs into token budgets.

This is a prototype implementation:
- It will work well enough for Wikipedia-style text.
- You can later upgrade to heading-aware splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import tiktoken


@dataclass
class Chunk:
    """A single chunk of text plus minimal metadata."""
    text: str
    token_count: int
    chunk_index: int


def chunk_text(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 80,
    encoding_name: str = "cl100k_base",
) -> List[Chunk]:
    """
    Split `text` into token-budgeted chunks.

    Strategy:
    1) Split text into paragraphs (blank lines).
    2) Pack paragraphs into chunks until adding the next paragraph would exceed `chunk_size`.
    3) Add overlap by carrying over the last `overlap` tokens from the previous chunk.

    Notes:
    - This keeps paragraphs intact most of the time.
    - For very long paragraphs, we fall back to token-splitting that paragraph.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    enc = tiktoken.get_encoding(encoding_name)

    # Normalize line endings and trim
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[Chunk] = []
    current_tokens: List[int] = []
    current_text_parts: List[str] = []

    def flush_chunk(chunk_index: int) -> None:
        """Finalize current chunk into output list."""
        if not current_tokens:
            return
        chunk_text_str = "\n\n".join(current_text_parts).strip()
        chunks.append(
            Chunk(
                text=chunk_text_str,
                token_count=len(current_tokens),
                chunk_index=chunk_index,
            )
        )

    chunk_index = 0

    for p in paragraphs:
        p_tokens = enc.encode(p)

        # If a single paragraph is longer than chunk_size, split it by tokens.
        if len(p_tokens) > chunk_size:
            # Flush whatever we have so far before handling the long paragraph.
            flush_chunk(chunk_index)
            if current_tokens:
                chunk_index += 1

            start = 0
            while start < len(p_tokens):
                end = min(start + chunk_size, len(p_tokens))
                window_tokens = p_tokens[start:end]
                window_text = enc.decode(window_tokens)
                chunks.append(
                    Chunk(
                        text=window_text,
                        token_count=len(window_tokens),
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

                # Overlap for next window inside the long paragraph
                start = end - overlap if end < len(p_tokens) else end

            # Reset current accumulators after splitting long paragraph
            current_tokens = []
            current_text_parts = []
            continue

        # If adding this paragraph would exceed chunk_size, flush current chunk.
        if len(current_tokens) + len(p_tokens) > chunk_size:
            flush_chunk(chunk_index)
            chunk_index += 1

            # Start next chunk with overlap tokens from previous chunk
            if overlap > 0 and chunks:
                overlap_tokens = enc.encode(chunks[-1].text)[-overlap:]
                current_tokens = overlap_tokens[:]
                current_text_parts = [enc.decode(overlap_tokens)]
            else:
                current_tokens = []
                current_text_parts = []

        # Add paragraph to current chunk
        current_tokens.extend(p_tokens)
        current_text_parts.append(p)

    # Flush last chunk
    flush_chunk(chunk_index)

    return chunks