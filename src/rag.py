from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retrieve import search


LLM_MODEL = "gpt-4.1-mini"  # good balance for RAG

_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

_prompt = ChatPromptTemplate.from_template(
"""
You are an ML/AI tutor.

Answer the question using ONLY the context provided.
If the answer is not contained in the context, say:
"I don't have that information in my sources."

Cite sources inline using:
[Title]

CONTEXT:
{context}

QUESTION:
{question}
"""
)

_parser = StrOutputParser()


def generate_answer(question: str, k: int = 8):
    hits = search(question, k=k)

    # Build context string
    context_blocks = []
    print("\nTop retrieved titles:")
    for h in hits:
        print("-", h.fields.get("title"))
        title = h.fields.get("title")
        text = h.fields.get("text")
        context_blocks.append(f"[{title}]\n{text}")

    context = "\n\n".join(context_blocks)

    chain = _prompt | _llm | _parser

    answer = chain.invoke({
        "question": question,
        "context": context
    })

    return answer, hits