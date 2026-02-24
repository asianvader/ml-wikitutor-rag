import json
from pathlib import Path
from datetime import datetime

from src.rag import generate_answer

QUESTIONS_PATH = Path("eval/questions.jsonl")
OUT_PATH = Path("eval/results.jsonl")


def load_questions():
    qs = []
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def sources_titles(sources):
    return [s.get("title") for s in sources]


def check_expectations(q, answer, sources):
    """
    Very lightweight checks:
    - If expect_source_contains: ensure at least one expected title appears
    - If expect_refusal: ensure answer contains refusal phrase
    """
    titles = sources_titles(sources)
    ok = True
    notes = []

    expected_titles = q.get("expect_source_contains")
    if expected_titles:
        hit = any(any(exp.lower() in (t or "").lower() for t in titles) for exp in expected_titles)
        if not hit:
            ok = False
            notes.append(f"Missing expected source(s): {expected_titles}")

    if q.get("expect_refusal"):
        refusal_phrase = "I don't have that information in my sources"
        if refusal_phrase not in answer:
            ok = False
            notes.append("Expected refusal, but answer did not refuse.")

    return ok, notes


def main():
    questions = load_questions()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for q in questions:
            qid = q.get("id")
            question = q["question"]

            answer, sources, hits, confidence = generate_answer(question, k=15)

            ok, notes = check_expectations(q, answer, sources)

            record = {
                "id": qid,
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "ok": ok,
                "notes": notes,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            status = "OK" if ok else "FAIL"
            print(f"[{status}] {qid}: {question}")

    print(f"\nWrote results to: {OUT_PATH}")


if __name__ == "__main__":
    main()