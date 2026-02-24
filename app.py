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
            answer, sources, hits, confidence = generate_answer(question, k=k)

        st.subheader("Answer")
        st.write(answer)

        st.markdown(
            f"**Confidence:** {confidence['label']} "
            f"({confidence['value']*100:.0f}%)"
        )

        st.subheader("Sources")

        for s in sources:
            st.markdown(
                f"**[{s['n']}] {s['title']}**  \n"
                f"Score: `{s['score']}`  \n"
                f"{s['url']}"
            )

            with st.expander(f"Preview [{s['n']}]"):
                st.write(s["preview"])

        with st.expander("🔎 Raw Retrieval Debug"):
            for i, h in enumerate(hits, start=1):
                st.write(
                    i,
                    h.fields.get("title"),
                    getattr(h, "score", None)
                )

        with st.expander("🧪 Debug"):
            st.json(confidence)