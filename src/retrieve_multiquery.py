"""
Multi-Query retrieval: generates several rephrasings of the user's question,
runs each through the vector index, then merges and deduplicates the results.

Why this helps
--------------
A single query embedding captures only one phrasing of the question. If the
relevant chunks use different vocabulary (e.g. the question says "memorise"
but the article says "overfit"), the correct chunk may rank low. By generating
alternative phrasings — more formal, more informal, more specific, more general
— we cast a wider net and significantly improve recall.

The original question is always included as query #1, so the method can only
add relevant results; it never removes what the plain search would find.

Usage (standalone)
------------------
    uv run python -m src.retrieve_multiquery
"""

from __future__ import annotations

import time
from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.retrieve import search

# ── Query generation ──────────────────────────────────────────────────────────

_QUERY_GEN_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at improving document retrieval for an ML/AI tutor system.

Given the question below, write {n} alternative rephrasings that:
- Preserve the original intent exactly
- Use different vocabulary, synonyms, or levels of technical detail
- May be more formal, more informal, or approach the concept from a different angle

Return ONLY the rephrasings, one per line, with no numbering, bullets, or extra text.

Question: {question}"""
)

_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)
_parser = StrOutputParser()
_query_chain = _QUERY_GEN_PROMPT | _llm | _parser


def generate_queries(question: str, n: int = 3, retries: int = 3) -> list[str]:
    """
    Use the LLM to produce *n* alternative phrasings of *question*.
    Always returns a list that starts with the original question.
    """
    for attempt in range(retries):
        try:
            raw = _query_chain.invoke({"question": question, "n": n})
            variants = [q.strip() for q in raw.strip().splitlines() if q.strip()]
            # Deduplicate while preserving order; always put original first
            seen = {question.lower()}
            unique = [question]
            for v in variants:
                if v.lower() not in seen:
                    seen.add(v.lower())
                    unique.append(v)
            return unique[:n + 1]  # original + up to n variants
        except Exception as e:
            wait = 2 ** attempt
            print(f"Query generation failed ({type(e).__name__}). Retrying in {wait}s…")
            time.sleep(wait)

    # Fallback: just use the original question
    return [question]


# ── Multi-query search ────────────────────────────────────────────────────────

def search_multiquery(
    question: str,
    k: int = 8,
    chunker: str = "token",
    n_variants: int = 3,
) -> list:
    """
    Run retrieval for the original question plus *n_variants* rephrasings,
    then merge and deduplicate results ranked by best (lowest) cosine distance.

    Returns up to k * 2 hits so the caller has a richer pool to re-rank.
    """
    queries = generate_queries(question, n=n_variants)

    print(f"  [multiquery] Generated {len(queries)} queries:")
    for i, q in enumerate(queries):
        label = "(original)" if i == 0 else f"(variant {i})"
        print(f"    {label} {q}")

    # Collect hits across all queries; deduplicate by chunk id
    seen_ids: set[str] = set()
    merged: list = []

    for query in queries:
        hits = search(query, k=k, chunker=chunker)
        for hit in hits:
            hit_id = getattr(hit, "id", None) or hit.fields.get("chunk_id", "")
            if hit_id not in seen_ids:
                seen_ids.add(hit_id)
                merged.append(hit)

    # Re-rank by cosine distance score (lower = more similar)
    merged.sort(key=lambda h: getattr(h, "score", 1e9))

    return merged[: k * 2]


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    question = "What is overfitting and how can it be prevented?"
    print(f"Question: {question}\n")
    hits = search_multiquery(question, k=5, chunker="token", n_variants=3)
    print(f"\nTop {len(hits)} merged hits:")
    for h in hits[:8]:
        title = h.fields.get("title", "?")
        score = getattr(h, "score", None)
        preview = (h.fields.get("text") or "")[:100].replace("\n", " ")
        print(f"  [{score:.3f}] {title} — {preview}")
