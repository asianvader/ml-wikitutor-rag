"""
Generation-level evaluation using LLM-as-judge.

Metrics:
  faithfulness    (1–5): are all claims in the answer grounded in the context?
  answer_relevance (1–5): does the answer actually address the question?

Usage:
  uv run python -m src.eval_generation
  uv run python -m src.eval_generation --chunker parent_child
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.rag import generate_answer

QUESTIONS_PATH = Path("eval/questions.jsonl")
OUT_PATH = Path("eval/generation_results.jsonl")

_judge_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
_parser = StrOutputParser()

_JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """You are an impartial evaluator for a Retrieval-Augmented Generation (RAG) system.

Score the following answer on TWO criteria. Return ONLY a JSON object — no prose.

CRITERIA:
1. faithfulness (1–5): Every factual claim in the answer is directly supported by the provided context.
   1 = many unsupported claims, 5 = fully grounded in context.

2. answer_relevance (1–5): The answer directly and completely addresses the question.
   1 = off-topic or incomplete, 5 = precise and thorough.

QUESTION:
{question}

CONTEXT (what the RAG system retrieved):
{context}

ANSWER (what the RAG system generated):
{answer}

Return exactly this JSON (no markdown fences, no extra keys):
{{"faithfulness": <int 1-5>, "answer_relevance": <int 1-5>, "faithfulness_reason": "<one sentence>", "answer_relevance_reason": "<one sentence>"}}"""
)

_judge_chain = _judge_llm | _parser


def _parse_scores(raw: str) -> dict:
    """Extract JSON from judge output, tolerating minor formatting issues."""
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        data = json.loads(cleaned)
        return {
            "faithfulness": int(data["faithfulness"]),
            "answer_relevance": int(data["answer_relevance"]),
            "faithfulness_reason": data.get("faithfulness_reason", ""),
            "answer_relevance_reason": data.get("answer_relevance_reason", ""),
        }
    except Exception as e:
        return {
            "faithfulness": None,
            "answer_relevance": None,
            "faithfulness_reason": f"parse error: {e}",
            "answer_relevance_reason": raw[:200],
        }


def _build_context_text(sources: list[dict]) -> str:
    """Reconstruct a readable context string from the sources metadata."""
    parts = []
    for s in sources:
        preview = s.get("preview", "")
        title = s.get("title", "Unknown")
        parts.append(f"[{s['n']}] {title}: {preview}")
    return "\n\n".join(parts) if parts else "(no context retrieved)"


def evaluate(chunker: str = "token", k: int = 15) -> None:
    questions = []
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    # Skip refusal questions — no answer quality to judge
    ml_questions = [q for q in questions if not q.get("expect_refusal")]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"\nEvaluating {len(ml_questions)} questions  [chunker={chunker}, k={k}]\n")
    print(f"{'ID':<6} {'Faithful':>8} {'Relevant':>8}  Question")
    print("-" * 70)

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for q in ml_questions:
            qid = q["id"]
            question = q["question"]

            # 1. Generate answer (context is the full string passed to the LLM)
            answer, sources, _hits, confidence, context_text = generate_answer(
                question, k=k, chunker=chunker
            )

            # 3. Judge
            raw = _judge_chain.invoke(
                _JUDGE_PROMPT.format_messages(
                    question=question,
                    context=context_text,
                    answer=answer,
                )
            )
            scores = _parse_scores(raw)

            record = {
                "id": qid,
                "question": question,
                "chunker": chunker,
                "answer": answer,
                "context_preview": context_text[:400],
                "sources": [s.get("title") for s in sources],
                "confidence": confidence,
                "faithfulness": scores["faithfulness"],
                "answer_relevance": scores["answer_relevance"],
                "faithfulness_reason": scores["faithfulness_reason"],
                "answer_relevance_reason": scores["answer_relevance_reason"],
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            results.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            f_str = str(scores["faithfulness"]) if scores["faithfulness"] else "ERR"
            r_str = str(scores["answer_relevance"]) if scores["answer_relevance"] else "ERR"
            print(f"{qid:<6} {f_str:>8} {r_str:>8}  {question[:55]}")

    # Aggregate summary
    valid = [r for r in results if r["faithfulness"] is not None]
    if valid:
        avg_f = sum(r["faithfulness"] for r in valid) / len(valid)
        avg_r = sum(r["answer_relevance"] for r in valid) / len(valid)

        print("\n" + "=" * 70)
        print("  AGGREGATE SUMMARY")
        print("=" * 70)
        bar_f = "█" * round(avg_f * 4) + "░" * (20 - round(avg_f * 4))
        bar_r = "█" * round(avg_r * 4) + "░" * (20 - round(avg_r * 4))
        print(f"  Faithfulness     {avg_f:.2f}/5.00  {bar_f}")
        print(f"  Answer Relevance {avg_r:.2f}/5.00  {bar_r}")
        print(f"  Questions scored: {len(valid)}/{len(ml_questions)}")
        print("=" * 70)

    print(f"\nResults saved to: {OUT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Generation-level evaluation")
    parser.add_argument(
        "--chunker",
        default="token",
        choices=["token", "semantic", "parent_child"],
        help="Retrieval strategy to evaluate (default: token)",
    )
    parser.add_argument(
        "--k", type=int, default=15, help="Number of chunks to retrieve (default: 15)"
    )
    args = parser.parse_args()
    evaluate(chunker=args.chunker, k=args.k)


if __name__ == "__main__":
    main()
