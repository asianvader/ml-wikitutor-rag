from __future__ import annotations
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.retrieve import search
from dotenv import load_dotenv
load_dotenv()

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

def _select_diverse_hits(hits, *, max_per_title: int = 2, max_total: int = 12):
    """
    Select a diverse subset of hits:
    - sort by score (lower is better)
    - keep at most `max_per_title` per title
    - return at most `max_total` hits
    """
    hits_sorted = sorted(hits, key=lambda h: getattr(h, "score", 1e9))

    out = []
    per_title = {}

    for h in hits_sorted:
        title = h.fields.get("title") or "Unknown"
        per_title.setdefault(title, 0)
        if per_title[title] >= max_per_title:
            continue
        out.append(h)
        per_title[title] += 1
        if len(out) >= max_total:
            break

    return out

def _confidence_from_sources(sources):
    """
    Simple heuristic confidence from Zvec cosine distance scores.
    Lower score = more similar.

    Returns:
      dict with label + numeric score in [0, 1] + explanation fields
    """
    scores = [s["score"] for s in sources if s.get("score") is not None]
    if not scores:
        return {"label": "Low", "value": 0.0, "best": None, "worst": None, "reason": "No retrieval scores"}

    best = min(scores)
    worst = max(scores)

    # Map best score (distance) to a 0.1 confidence value.
    # These thresholds are empirical; tweak later.
    # Typical “good” distances often land ~0.25–0.45 in your current setup.
    if best <= 0.38:
        label = "High"
        value = 0.85
    elif best <= 0.45:
        label = "Medium"
        value = 0.65
    else:
        label = "Low"
        value = 0.45
    # Slightly adjust by spread: if worst is much worse than best, penalize
    spread = worst - best
    if spread > 0.20:
        value -= 0.10
    if spread > 0.30:
        value -= 0.10

    # Clamp
    value = max(0.0, min(1.0, value))

    return {
        "label": label,
        "value": value,
        "best": best,
        "worst": worst,
        "spread": spread,
    }

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


def generate_answer(question: str, k: int = 15):
    hits = search(question, k=k)

    # choose a diverse subset before building context
    diverse_hits = _select_diverse_hits(hits, max_per_title=1, max_total=8)

    context, sources = _build_context(
        diverse_hits,
        max_context_tokens=3000,
        max_chunks_per_title=1,
    )

    chain = _prompt | _llm | _parser
    answer = chain.invoke({"question": question, "context": context})

    confidence = _confidence_from_sources(sources)
    return answer, sources, hits, confidence