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
You are an ML/AI tutor. You ONLY answer questions about machine learning, data science, and AI.

If the question is not about machine learning, data science, or AI — regardless of what the context contains — respond with exactly:
"I can only answer questions about machine learning, data science, and AI."

If the question is on-topic but the context does not contain enough information to answer, say:
"I don't have that information in my sources."

Otherwise, write a clear explanation in **8-12 sentences** (or ~150-250 words).
Include:
- a 1-2 sentence direct answer
- a short example or intuition (when relevant)
- common pitfalls or trade-offs (when relevant)

STRICT GROUNDING RULES — you MUST follow these:
- Every factual claim you make MUST be directly supported by the context below.
- Do NOT add facts, numbers, formulas, or examples from your training knowledge that are absent from the context.
- If a detail is not in the context, omit it or say "my sources don't cover this detail."
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
    Heuristic confidence from Zvec cosine distance scores.
    Lower score = more similar.

    Anchors on mean-of-top-3 (more robust than single best hit).
    Boosts when multiple distinct titles score well (multi-source agreement).
    Penalises wide spread between best and worst.

    Returns:
      dict with label + numeric value in [0, 1] + debug fields
    """
    valid = [s for s in sources if s.get("score") is not None]
    if not valid:
        return {"label": "Low", "value": 0.0, "best": None, "worst": None, "reason": "No retrieval scores"}

    scores_sorted = sorted(s["score"] for s in valid)
    best = scores_sorted[0]
    worst = scores_sorted[-1]
    spread = worst - best

    # Anchor on mean of top-3 — less sensitive to a single lucky hit
    top3 = scores_sorted[:3]
    anchor = sum(top3) / len(top3)

    if anchor <= 0.38:
        label = "High"
        value = 0.85
    elif anchor <= 0.45:
        label = "Medium"
        value = 0.65
    else:
        label = "Low"
        value = 0.45

    # Spread penalty: noisy hit set lowers confidence
    if spread > 0.20:
        value -= 0.10
    if spread > 0.30:
        value -= 0.10

    # Multi-source agreement boost: ≥3 good hits from ≥2 distinct titles
    good_hits = [s for s in valid if s["score"] < 0.40]
    unique_good_titles = len({s.get("title", "") for s in good_hits})
    if len(good_hits) >= 3 and unique_good_titles >= 2:
        value += 0.05

    value = max(0.0, min(1.0, value))

    return {
        "label": label,
        "value": value,
        "best": best,
        "worst": worst,
        "anchor": round(anchor, 4),
        "spread": round(spread, 4),
        "good_hits": len(good_hits),
        "unique_good_titles": unique_good_titles,
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
        text = h.fields.get("parent_text") or h.fields.get("text") or ""
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


def generate_answer(
    question: str,
    k: int = 15,
    chunker: str = "token",
    use_multiquery: bool = False,
):
    if use_multiquery:
        from src.retrieve_multiquery import search_multiquery
        hits = search_multiquery(question, k=k, chunker=chunker)
    else:
        hits = search(question, k=k, chunker=chunker)

    diverse_hits = _select_diverse_hits(hits, max_per_title=1, max_total=8)

    context, sources = _build_context(
        diverse_hits,
        max_context_tokens=3000,
        max_chunks_per_title=1,
    )

    # Compute confidence from ALL diverse hits (not just those that fit the token budget)
    all_hit_scores = [
        {"score": getattr(h, "score", None), "title": h.fields.get("title")}
        for h in diverse_hits
    ]
    confidence = _confidence_from_sources(all_hit_scores)

    chain = _prompt | _llm | _parser
    answer = chain.invoke({"question": question, "context": context})

    # hide sources when refusing
    refused = (
        "don't have that information in my sources" in answer.lower()
        or "i can only answer questions about machine learning" in answer.lower()
    )
    if refused:
        return answer, [], hits, confidence, context

    return answer, sources, hits, confidence, context


def stream_answer(
    question: str,
    k: int = 15,
    chunker: str = "token",
    use_multiquery: bool = False,
):
    """
    Like generate_answer() but returns a streaming token iterator for the LLM response.

    Returns:
      token_stream: generator yielding str chunks (pass to st.write_stream)
      sources: list of source dicts (available immediately, before streaming)
      hits: raw retrieval hits
      confidence: confidence dict
      context: context string
    """
    if use_multiquery:
        from src.retrieve_multiquery import search_multiquery
        hits = search_multiquery(question, k=k, chunker=chunker)
    else:
        hits = search(question, k=k, chunker=chunker)

    diverse_hits = _select_diverse_hits(hits, max_per_title=1, max_total=8)

    context, sources = _build_context(
        diverse_hits,
        max_context_tokens=3000,
        max_chunks_per_title=1,
    )

    all_hit_scores = [
        {"score": getattr(h, "score", None), "title": h.fields.get("title")}
        for h in diverse_hits
    ]
    confidence = _confidence_from_sources(all_hit_scores)

    chain = _prompt | _llm | _parser
    token_stream = chain.stream({"question": question, "context": context})

    return token_stream, sources, hits, confidence, context