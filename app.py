import streamlit as st
from src.retrieve import search

st.set_page_config(page_title="ML WikiTutor", page_icon="📚", layout="wide")

st.title("📚 ML WikiTutor")
st.write("Retrieval demo (Zvec + OpenAI embeddings). No LLM answering yet.")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-K", 1, 15, 8)

question = st.text_input("Ask a question:", placeholder="What is the difference between supervised and unsupervised learning?")

if st.button("Search"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Searching..."):
            hits = search(question, k=k)

        st.subheader("Top results")
        for i, h in enumerate(hits, start=1):
            title = h.fields.get("title")
            url = h.fields.get("source_url")
            score = getattr(h, "score", None)
            text = h.fields.get("text") or ""

            st.markdown(f"**{i}. {title}**  \nScore: `{score}`  \nSource: {url}")

            with st.expander("Show chunk text"):
                st.write(text)