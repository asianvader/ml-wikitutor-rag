"""
Retrieval-level evaluation: precision, recall, hit rate, and MRR.

For each question in eval/questions.jsonl (that has a `relevant_titles` list),
this script runs retrieval using one or more chunker strategies and computes:

  Precision@k  — fraction of uniquely retrieved article titles that are relevant
  Recall@k     — fraction of relevant article titles that were retrieved
  Hit rate     — 1 if at least one relevant article was retrieved, else 0
  MRR          — reciprocal rank of the first relevant chunk hit

Title matching is done with substring containment in both directions to handle
Wikipedia redirects (e.g. "Transformer (machine learning)" → "Transformer
(deep learning)").

Usage
-----
    uv run python -m src.eval_retrieval                          # token vs semantic
    uv run python -m src.eval_retrieval --chunkers token multiquery_token
    uv run python -m src.eval_retrieval --k 15                   # change top-k
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.retrieve import search
from src.retrieve_multiquery import search_multiquery

QUESTIONS_PATH = Path("eval/questions.jsonl")
OUT_PATH = Path("eval/retrieval_results.jsonl")


# ── Title matching ────────────────────────────────────────────────────────────

def _titles_match(retrieved: str, relevant: str) -> bool:
    """
    True if the titles refer to the same article (case-insensitive).

    Matching rules (in order):
    1. Exact match.
    2. Prefix boundary match: one title is a prefix of the other, ending at
       a space or '(' — handles parenthetical qualifiers like
       "Regularization" → "Regularization (mathematics)" but prevents
       "Supervised learning" from matching "Self-supervised learning".
    """
    r = retrieved.lower().strip()
    g = relevant.lower().strip()
    if r == g:
        return True
    # Prefix boundary: check both directions
    for shorter, longer in [(g, r), (r, g)]:
        if longer.startswith(shorter):
            rest = longer[len(shorter):]
            # Only accept if the next character is a word/qualifier boundary
            if not rest or rest[0] in (" ", "("):
                return True
    return False


def _is_relevant(retrieved_title: str, relevant_titles: list[str]) -> bool:
    return any(_titles_match(retrieved_title, rel) for rel in relevant_titles)


# ── Per-question metrics ──────────────────────────────────────────────────────

def _run_search(question: str, k: int, mode: str) -> list:
    """Dispatch to the correct search function based on *mode*."""
    if mode.startswith("multiquery_"):
        chunker = mode[len("multiquery_"):]
        return search_multiquery(question, k=k, chunker=chunker)
    return search(question, k=k, chunker=mode)


def compute_metrics(
    question: str,
    relevant_titles: list[str],
    chunker: str,
    k: int,
) -> dict:
    """Run retrieval and return precision, recall, hit rate, and MRR."""
    hits = _run_search(question, k=k, mode=chunker)

    # Deduplicate retrieved titles (a single article may have many chunks)
    seen: set[str] = set()
    unique_titles: list[str] = []
    for h in hits:
        t = (h.fields.get("title") or "").strip()
        if t and t not in seen:
            seen.add(t)
            unique_titles.append(t)

    relevant_retrieved = [t for t in unique_titles if _is_relevant(t, relevant_titles)]

    n_retrieved = len(unique_titles)
    n_relevant_retrieved = len(relevant_retrieved)
    n_relevant = len(relevant_titles)

    precision = n_relevant_retrieved / n_retrieved if n_retrieved > 0 else 0.0
    recall    = min(n_relevant_retrieved / n_relevant, 1.0) if n_relevant > 0 else 0.0
    hit       = n_relevant_retrieved > 0

    # MRR: rank is position in the full (non-deduplicated) hit list
    mrr = 0.0
    for rank, h in enumerate(hits, start=1):
        t = (h.fields.get("title") or "").strip()
        if _is_relevant(t, relevant_titles):
            mrr = 1.0 / rank
            break

    return {
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "hit":       hit,
        "mrr":       round(mrr, 3),
        "n_relevant":           n_relevant,
        "n_relevant_retrieved": n_relevant_retrieved,
        "n_retrieved_unique":   n_retrieved,
        "relevant_retrieved":   relevant_retrieved,
        "retrieved_titles":     unique_titles,
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    """Mean metrics across all evaluated questions."""
    if not results:
        return {}
    keys = ["precision", "recall", "hit", "mrr"]
    return {k: round(sum(r[k] for r in results) / len(results), 3) for k in keys}


# ── Pretty printing ───────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def print_results_table(all_results: dict[str, list[dict]], questions: list[dict]) -> None:
    chunkers = list(all_results.keys())
    q_map = {q["id"]: q for q in questions}

    # Per-question table
    col_w = 14
    header_q = f"{'ID':<6} {'Question':<42}"
    header_m = "  ".join(
        f"{'P':>5} {'R':>5} {'Hit':>4} {'MRR':>5}".center(col_w * 2)
        for _ in chunkers
    )
    chunker_header = "  ".join(f"{c.upper():^{col_w * 2}}" for c in chunkers)

    sep = "-" * (6 + 42 + len(chunkers) * (col_w * 2 + 4))
    print()
    print("=" * len(sep))
    print("  RETRIEVAL EVALUATION — PER QUESTION")
    print("=" * len(sep))
    print(f"{'':<48}" + "  ".join(f"{c.upper():^26}" for c in chunkers))
    print(f"{'ID':<6} {'Question':<42}" + "  ".join(f"{'Prec':>5} {'Rec':>5} {'Hit':>4} {'MRR':>5}  " for _ in chunkers))
    print(sep)

    for q in questions:
        qid = q["id"]
        if not q.get("relevant_titles"):
            continue  # skip refusal questions
        snippet = q["question"][:40] + ("…" if len(q["question"]) > 40 else "")
        row = f"{qid:<6} {snippet:<42}"
        for c in chunkers:
            res = next((r for r in all_results[c] if r["id"] == qid), None)
            if res:
                hit_sym = "✓" if res["hit"] else "✗"
                row += f"  {res['precision']:>5.3f} {res['recall']:>5.3f} {hit_sym:>4} {res['mrr']:>5.3f}  "
        print(row)

    # Summary table
    print(sep)
    print(f"{'MEAN':<6} {'':<42}" + "  ".join(
        f"  {agg['precision']:>5.3f} {agg['recall']:>5.3f} {'':>4} {agg['mrr']:>5.3f}  "
        for agg in [aggregate(all_results[c]) for c in chunkers]
    ))
    print()

    # Aggregate summary with bars
    print("=" * len(sep))
    print("  AGGREGATE SUMMARY")
    print("=" * len(sep))
    metrics = ["precision", "recall", "hit", "mrr"]
    labels  = ["Precision@k", "Recall@k   ", "Hit Rate   ", "MRR        "]
    for label, metric in zip(labels, metrics):
        print(f"  {label}  " + "   ".join(
            f"{c.upper()}: {agg[metric]:.3f} {_bar(agg[metric])}"
            for c, agg in [(c, aggregate(all_results[c])) for c in chunkers]
        ))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def load_questions() -> list[dict]:
    qs = []
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def main(chunkers: list[str], k: int) -> None:
    questions = load_questions()
    eval_qs   = [q for q in questions if q.get("relevant_titles")]  # skip refusal-only

    print(f"\nEvaluating {len(eval_qs)} questions with k={k}, chunkers={chunkers} …\n")

    all_results: dict[str, list[dict]] = {c: [] for c in chunkers}
    records = []

    for q in eval_qs:
        qid      = q["id"]
        question = q["question"]
        relevant = q["relevant_titles"]

        row: dict = {"id": qid, "question": question, "relevant_titles": relevant, "k": k}

        for chunker in chunkers:
            metrics = compute_metrics(question, relevant, chunker, k)
            all_results[chunker].append({"id": qid, **metrics})
            row[chunker] = metrics
            p, r, h, m = metrics["precision"], metrics["recall"], metrics["hit"], metrics["mrr"]
            hit_sym = "✓" if h else "✗"
            print(f"  [{chunker:>8}] {qid}: P={p:.3f}  R={r:.3f}  Hit={hit_sym}  MRR={m:.3f}")

        print()
        records.append(row)

    # Print comparison table
    print_results_table(all_results, questions)

    # Save results
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            rec["timestamp"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Write aggregate summary as final line
        summary = {
            "id": "_summary",
            "k": k,
            "chunkers": chunkers,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for c in chunkers:
            summary[c] = aggregate(all_results[c])
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Results saved to: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval-level precision/recall evaluation.")
    parser.add_argument(
        "--chunkers",
        nargs="+",
        choices=["token", "semantic", "parent_child", "multiquery_token", "multiquery_semantic"],
        default=["token", "semantic"],
        help="Retrieval modes to evaluate (default: token semantic).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10).",
    )
    args = parser.parse_args()
    main(chunkers=args.chunkers, k=args.k)
