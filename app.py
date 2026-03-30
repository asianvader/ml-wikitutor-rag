import os
import streamlit as st
from src.rag import retrieve_context, answer_stream, _is_refused
from src.config import UI_DEFAULT_K

st.set_page_config(page_title="ML WikiTutor", page_icon="📚", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 900px; }
h1 { margin-bottom: 0.25rem; }
</style>
""", unsafe_allow_html=True)

st.title("📚 ML WikiTutor")
st.write("Ask questions about Machine Learning, Data Science, and AI using a curated Wikipedia knowledge base.")

with st.expander("Settings", expanded=False):
    k = st.number_input("Top-K", min_value=1, max_value=30, value=UI_DEFAULT_K, step=1)

    chunker = st.radio(
        "Chunking strategy",
        options=["token", "semantic", "parent_child"],
        format_func=lambda x: {
            "token": "Token-based (default)",
            "semantic": "Semantic",
            "parent_child": "Parent-Child",
        }[x],
        horizontal=True,
        help=(
            "Token-based: fixed-size chunks with overlap. "
            "Semantic: splits at points where meaning shifts. "
            "Parent-Child: small child chunks indexed for precision; "
            "parent chunk (~500 tok) returned to the LLM for richer context. "
            "Semantic and Parent-Child require their indexes to be built first."
        ),
    )

    use_multiquery = st.checkbox(
        "Multi-Query retrieval",
        value=False,
        help=(
            "Generates 3 alternative rephrasings of your question and merges "
            "the results. Improves recall for vague or informally phrased questions "
            "at the cost of a small extra LLM call."
        ),
    )

    # Warn if the selected index doesn't exist yet
    if chunker == "semantic" and not os.path.exists("index/zvec_wiki_ml_semantic"):
        st.warning(
            "⚠️ Semantic index not found. Run `bash scripts/rebuild_index_semantic.sh` first, "
            "then restart the app."
        )
    if chunker == "parent_child" and not os.path.exists("index/zvec_wiki_ml_parent_child"):
        st.warning(
            "⚠️ Parent-child index not found. Run `bash scripts/rebuild_index_parent_child.sh` first, "
            "then restart the app."
        )

with st.form("ask_form", clear_on_submit=False):
    question = st.text_input(
        "Ask a question:",
        placeholder="What is the difference between supervised and unsupervised learning?",
    )
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            # ── Step 1: retrieval with live status updates ─────────────────
            with st.status("Retrieving from knowledge base...", expanded=True) as status:
                if use_multiquery:
                    st.write("🔀 Generating query variants with LLM...")
                st.write("🔍 Searching vector index...")
                hits, context, sources, confidence = retrieve_context(
                    question, k=k, chunker=chunker, use_multiquery=use_multiquery
                )
                n_chunks = len(sources)
                n_articles = len({s["title"] for s in sources})
                st.write(
                    f"✅ Found {n_chunks} chunk{'s' if n_chunks != 1 else ''} "
                    f"across {n_articles} article{'s' if n_articles != 1 else ''}"
                )
                status.update(
                    label=f"✅ Retrieved {n_chunks} source{'s' if n_chunks != 1 else ''}",
                    state="complete",
                    expanded=False,
                )

            # ── Step 2: stream the LLM answer ──────────────────────────────
            generating_msg = st.empty()
            generating_msg.caption("✍️ Generating answer...")
            st.subheader("Answer")
            answer = st.write_stream(answer_stream(context, question))
            generating_msg.empty()

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.stop()

        refused = _is_refused(answer)

        if not refused:
            st.markdown(
                f"**Confidence:** {confidence['label']} "
                f"({confidence['value']*100:.0f}%)"
            )

            if sources:
                st.subheader("Sources")
                for s in sources:
                    # Zvec cosine distance: lower = more similar → similarity = 1 - distance
                    similarity = f"{(1 - s['score']) * 100:.0f}%" if s.get("score") is not None else "n/a"
                    st.markdown(
                        f"**[{s['n']}] {s['title']}**  \n"
                        f"Similarity: `{similarity}`  \n"
                        f"{s['url']}"
                    )
                    with st.expander(f"Preview [{s['n']}]"):
                        st.write(s["preview"])

        with st.expander("🔎 Debug"):
            st.json(confidence)
            for i, h in enumerate(hits, start=1):
                st.write(i, h.fields.get("title"), getattr(h, "score", None))
