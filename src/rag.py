from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retrieve import search

LLM_MODEL = "gpt-4.1-mini"
_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
_parser = StrOutputParser()

# Token counter for context budgeting
_enc = tiktoken.get_encoding("cl100k_base")


_prompt = ChatPromptTemplate.from_template(
"""
You are an ML/AI tutor.

Use ONLY the provided context to answer the question.
If the context does not contain enough information to answer, say:
"I don't have that information in my sources."

Rules:
- Keep the answer concise and correct.
- Include citations inline like [1], [2] matching the context blocks.
- Do not cite sources you did not use.

CONTEXT:
{context}

QUESTION:
{question}
"""
)


def _token_len(text: str) -> int:
    return len(_enc.encode(text or ""))


def _build_context(
    hits,
    *,
    max_context_tokens: int = 3000,
    max_chunks_per_title: int = 2,
):
    """
    Build a bounded, deduplicated context string with numbered citations.

    Returns:
      context_str: formatted text blocks with [n] labels
      sources: list of dicts with citation metadata for UI display
    """
    # Zvec cosine distance: lower score = more similar
    hits_sorted = sorted(hits, key=lambda h: getattr(h, "score", 1e9))

    sources = []
    context_parts = []
    used_tokens = 0

    per_title_count = {}
    seen = set()  # (title, chunk_id) guard

    citation_num = 1

    for h in hits_sorted:
        title = h.fields.get("title") or "Unknown"
        chunk_id = h.fields.get("chunk_id")
        text = h.fields.get("text") or ""
        url = h.fields.get("source_url")
        score = getattr(h, "score", None)

        key = (title, chunk_id)
        if key in seen:
            continue
        seen.add(key)

        # Limit redundancy: only include up to N chunks per page/title
        per_title_count.setdefault(title, 0)
        if per_title_count[title] >= max_chunks_per_title:
            continue

        block = f"[{citation_num}] Title: {title}\nURL: {url}\n\n{text}".strip()
        block_tokens = _token_len(block)

        # Stop when budget exceeded (leave room for prompt + question)
        if used_tokens + block_tokens > max_context_tokens:
            continue

        context_parts.append(block)
        used_tokens += block_tokens

        sources.append({
            "n": citation_num,
            "title": title,
            "url": url,
            "score": score,
            "chunk_id": chunk_id,
            "preview": text[:180].replace("\n", " "),
        })

        per_title_count[title] += 1
        citation_num += 1

        if used_tokens >= max_context_tokens:
            break

    return "\n\n---\n\n".join(context_parts), sources


def generate_answer(question: str, k: int = 10):
    """
    RAG:
      1) retrieve hits from Zvec
      2) build bounded context + citation map
      3) LLM answer constrained to context
    """
    hits = search(question, k=k)

    context, sources = _build_context(
        hits,
        max_context_tokens=3000,
        max_chunks_per_title=2,
    )

    chain = _prompt | _llm | _parser
    answer = chain.invoke({"question": question, "context": context})

    return answer, sources, hits