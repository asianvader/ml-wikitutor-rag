import streamlit as st
from src.rag import generate_answer

st.set_page_config(page_title="ML WikiTutor", page_icon="📚", layout="wide")

st.title("📚 ML WikiTutor")
st.write("Retrieval demo (Zvec + OpenAI embeddings).")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-K", 1, 15, 8)

# Put input + submit in a form so Enter works
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
        with st.spinner("Retrieving and generating answer..."):
            answer, hits = generate_answer(question, k=k)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, h in enumerate(hits, start=1):
            title = h.fields.get("title")
            url = h.fields.get("source_url")
            score = getattr(h, "score", None)

            # Safer display if url is missing
            if url:
                st.markdown(f"**{i}. {title}** — Score: `{score}`  \n{url}")
            else:
                st.markdown(f"**{i}. {title}** — Score: `{score}`")