import os
import streamlit as st
from src.rag import generate_answer

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
    k = st.number_input("Top-K", min_value=1, max_value=30, value=12, step=1)

    chunker = st.radio(
        "Chunking strategy",
        options=["token", "semantic"],
        format_func=lambda x: "Token-based (default)" if x == "token" else "Semantic",
        horizontal=True,
        help=(
            "Token-based: fixed-size chunks with overlap. "
            "Semantic: splits at points where meaning shifts. "
            "Requires the semantic index to be built first."
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

    # Warn if the semantic index doesn't exist yet
    if chunker == "semantic" and not os.path.exists("index/zvec_wiki_ml_semantic"):
        st.warning(
            "⚠️ Semantic index not found. Run `bash scripts/rebuild_index_semantic.sh` first, "
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
        strategy_label = f"{'Multi-Query + ' if use_multiquery else ''}{chunker.capitalize()}"
        with st.spinner(f"Retrieving ({strategy_label}) and generating answer…"):
            answer, sources, hits, confidence = generate_answer(
                question, k=k, chunker=chunker, use_multiquery=use_multiquery
            )

        st.subheader("Answer")
        st.write(answer)

        refused = (
            "don't have that information in my sources" in answer.lower()
            or "i can only answer questions about machine learning" in answer.lower()
        )

        if not refused:
            st.markdown(
                f"**Confidence:** {confidence['label']} "
                f"({confidence['value']*100:.0f}%)"
            )

            if sources:
                st.subheader("Sources")
                for s in sources:
                    st.markdown(
                        f"**[{s['n']}] {s['title']}**  \n"
                        f"Score: `{s['score']}`  \n"
                        f"{s['url']}"
                    )
                    with st.expander(f"Preview [{s['n']}]"):
                        st.write(s["preview"])

            with st.expander("🔎 Debug"):
                st.json(confidence)
                for i, h in enumerate(hits, start=1):
                    st.write(i, h.fields.get("title"), getattr(h, "score", None))
