from __future__ import annotations
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.retrieve import search
from src.config import (
    LLM_MODEL,
    DEFAULT_K,
    MAX_PER_TITLE,
    MAX_TOTAL_HITS,
    MAX_CONTEXT_TOKENS,
    CONF_HIGH_THRESHOLD,
    CONF_MEDIUM_THRESHOLD,
    CONF_HIGH_VALUE,
    CONF_MEDIUM_VALUE,
    CONF_LOW_VALUE,
    CONF_SPREAD_THRESHOLD_1,
    CONF_SPREAD_THRESHOLD_2,
    CONF_SPREAD_PENALTY,
    CONF_GOOD_HIT_THRESHOLD,
    CONF_MULTI_SOURCE_BOOST,
    REFUSAL_PHRASES,
    PREVIEW_CHARS,
)
from dotenv import load_dotenv
load_dotenv()

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


def _select_diverse_hits(hits, *, max_per_title: int = MAX_PER_TITLE, max_total: int = MAX_TOTAL_HITS):
    """
    Select a diverse subset of hits:
    - sort by score (lower is better for Zvec cosine distance)
    - keep at most `max_per_title` per title
    - return at most `max_total` hits
    """
    hits_sorted = sorted(hits, key=lambda h: getattr(h, "score", 1e9))

    out = []
    per_title: dict[str, int] = {}

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


def _confidence_from_sources(sources: list[dict]) -> dict:
    """
    Heuristic confidence derived from Zvec cosine distance scores in `sources`.

    Expects each item to have 'score' (float | None) and optionally 'title'.
    Lower score = more similar (good).

    Anchors on mean-of-top-3 (more robust than a single best hit).
    Penalises wide spread between best and worst hit.
    Boosts when multiple distinct titles score well (multi-source agreement).

    Returns a dict with label, numeric value in [0, 1], and debug fields.
    """
    valid = [s for s in sources if s.get("score") is not None]
    if not valid:
        return {"label": "Low", "value": 0.0, "best": None, "worst": None,
                "reason": "No retrieval scores"}

    scores_sorted = sorted(s["score"] for s in valid)
    best = scores_sorted[0]
    worst = scores_sorted[-1]
    spread = worst - best

    # Anchor on mean of top-3 — less sensitive to a single lucky hit
    top3 = scores_sorted[:3]
    anchor = sum(top3) / len(top3)

    if anchor <= CONF_HIGH_THRESHOLD:
        label, value = "High", CONF_HIGH_VALUE
    elif anchor <= CONF_MEDIUM_THRESHOLD:
        label, value = "Medium", CONF_MEDIUM_VALUE
    else:
        label, value = "Low", CONF_LOW_VALUE

    # Spread penalty: noisy hit set lowers confidence
    if spread > CONF_SPREAD_THRESHOLD_1:
        value -= CONF_SPREAD_PENALTY
    if spread > CONF_SPREAD_THRESHOLD_2:
        value -= CONF_SPREAD_PENALTY

    # Multi-source agreement boost: ≥3 good hits from ≥2 distinct titles
    good_hits = [s for s in valid if s["score"] < CONF_GOOD_HIT_THRESHOLD]
    unique_good_titles = len({s.get("title", "") for s in good_hits})
    if len(good_hits) >= 3 and unique_good_titles >= 2:
        value += CONF_MULTI_SOURCE_BOOST

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
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    max_chunks_per_title: int = MAX_PER_TITLE,
) -> tuple[str, list[dict]]:
    """
    Build a bounded, deduplicated context string with numbered citations.

    Returns:
      context_str: formatted text blocks with [n] labels
      sources: list of dicts with citation metadata for UI display
    """
    # Zvec cosine distance: lower score = more similar
    hits_sorted = sorted(hits, key=lambda h: getattr(h, "score", 1e9))

    sources: list[dict] = []
    context_parts: list[str] = []
    used_tokens = 0

    per_title_count: dict[str, int] = {}
    seen: set[tuple] = set()  # (title, chunk_id) dedup guard

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

        # Skip chunks that would exceed the token budget
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
            "preview": (text or "")[:PREVIEW_CHARS].replace("\n", " "),
        })

        per_title_count[title] += 1
        citation_num += 1

        if used_tokens >= max_context_tokens:
            break

    return "\n\n---\n\n".join(context_parts), sources


def _is_refused(answer: str) -> bool:
    """Return True if the answer is a refusal (off-topic or missing sources)."""
    lower = answer.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def _retrieve_and_build(
    question: str,
    k: int,
    chunker: str,
    use_multiquery: bool,
) -> tuple:
    """
    Shared retrieval + context-building logic used by both generate_answer()
    and stream_answer().

    Returns:
      hits: raw retrieval hits (for debug display)
      context: context string passed to the LLM
      sources: citation metadata for the chunks that made it into context
      confidence: confidence dict computed from sources (what the LLM actually sees)
    """
    if use_multiquery:
        from src.retrieve_multiquery import search_multiquery
        hits = search_multiquery(question, k=k, chunker=chunker)
    else:
        hits = search(question, k=k, chunker=chunker)

    diverse_hits = _select_diverse_hits(hits)
    context, sources = _build_context(diverse_hits)

    # Confidence is computed from `sources` — the exact chunks sent to the LLM,
    # not the broader diverse_hits pool, so the signal reflects what the model saw.
    confidence = _confidence_from_sources(sources)

    return hits, context, sources, confidence


def generate_answer(
    question: str,
    k: int = DEFAULT_K,
    chunker: str = "token",
    use_multiquery: bool = False,
):
    """Retrieve context and return a complete LLM answer (blocking)."""
    hits, context, sources, confidence = _retrieve_and_build(
        question, k, chunker, use_multiquery
    )

    chain = _prompt | _llm | _parser
    answer = chain.invoke({"question": question, "context": context})

    if _is_refused(answer):
        return answer, [], hits, confidence, context

    return answer, sources, hits, confidence, context


def stream_answer(
    question: str,
    k: int = DEFAULT_K,
    chunker: str = "token",
    use_multiquery: bool = False,
):
    """
    Retrieve context and return a streaming token iterator for the LLM response.

    Retrieval and context-building happen upfront (blocking); only the LLM
    generation streams token-by-token.

    Returns:
      token_stream: generator yielding str chunks (pass to st.write_stream)
      sources: citation metadata (available before streaming starts)
      hits: raw retrieval hits (for debug display)
      confidence: confidence dict computed from sources
      context: context string sent to the LLM
    """
    hits, context, sources, confidence = _retrieve_and_build(
        question, k, chunker, use_multiquery
    )

    chain = _prompt | _llm | _parser
    token_stream = chain.stream({"question": question, "context": context})

    return token_stream, sources, hits, confidence, context


def retrieve_context(
    question: str,
    k: int = DEFAULT_K,
    chunker: str = "token",
    use_multiquery: bool = False,
) -> tuple:
    """
    Run retrieval and context-building only — no LLM call.

    Use this together with answer_stream() when you want to show step-by-step
    status updates in the UI between the retrieval and generation phases.

    Returns: (hits, context, sources, confidence)
    """
    return _retrieve_and_build(question, k, chunker, use_multiquery)


def answer_stream(context: str, question: str):
    """
    Return a LangChain token stream for a pre-built context string.

    Pair with retrieve_context() to stream the LLM answer after showing
    retrieval progress in the UI:

        hits, context, sources, confidence = retrieve_context(question, ...)
        token_stream = answer_stream(context, question)
        answer = st.write_stream(token_stream)
    """
    chain = _prompt | _llm | _parser
    return chain.stream({"question": question, "context": context})
