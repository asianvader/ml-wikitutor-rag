"""Central configuration for ML WikiTutor RAG pipeline.

All tunable constants live here. Import from this module rather than
hardcoding values in individual source files.
"""

# ── Models ────────────────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_K = 15          # top-k chunks to retrieve from the vector index
MAX_PER_TITLE = 1       # diversity cap: at most N chunks per Wikipedia article
MAX_TOTAL_HITS = 8      # total diverse hits passed to the context builder
MAX_CONTEXT_TOKENS = 3000  # token budget for context sent to the LLM

# ── Confidence (Zvec cosine distance; lower = more similar) ───────────────────
# mean-of-top-3 distance thresholds that determine label
CONF_HIGH_THRESHOLD = 0.38
CONF_MEDIUM_THRESHOLD = 0.45

# base numeric values per label
CONF_HIGH_VALUE = 0.85
CONF_MEDIUM_VALUE = 0.65
CONF_LOW_VALUE = 0.45

# spread penalties: applied when (worst - best) distance exceeds threshold
CONF_SPREAD_THRESHOLD_1 = 0.20   # first penalty trigger
CONF_SPREAD_THRESHOLD_2 = 0.30   # second penalty trigger
CONF_SPREAD_PENALTY = 0.10       # value subtracted per triggered threshold

# multi-source agreement boost: ≥3 good hits from ≥2 distinct titles
CONF_GOOD_HIT_THRESHOLD = 0.40   # score below this counts as a "good hit"
CONF_MULTI_SOURCE_BOOST = 0.05

# ── Refusal detection (keep in sync with system prompt in rag.py) ─────────────
REFUSAL_PHRASES = [
    "i don't have that information in my sources",
    "i can only answer questions about machine learning",
]

# ── UI ────────────────────────────────────────────────────────────────────────
UI_DEFAULT_K = 15      # default value shown in the Top-K number input
PREVIEW_CHARS = 400    # characters of chunk text shown in source preview
