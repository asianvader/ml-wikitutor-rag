"""
Generate a comprehensive list of Wikipedia article titles for the ML/AI corpus
using the Wikipedia Category API.

Rather than link-crawling from seed pages (which misses important figures like
Marvin Minsky unless they happen to be linked from a seed), this script queries
Wikipedia categories directly — giving structured, comprehensive coverage of:
  - ML/AI concepts and algorithms
  - AI/ML researchers and pioneers
  - AI history and institutions
  - Data science and statistics

The off-topic noise articles are kept separately in titles.txt and are NOT
touched by this script. Output goes to data_raw/titles_generated.txt.

Usage
-----
    uv run python -m src.generate_titles
    uv run python -m src.generate_titles --max-per-category 300
    uv run python -m src.generate_titles --dry-run   # print counts only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "ml-wikitutor-rag/0.1 (education project; contact: phoebe.voong@gmail.com)",
    "Accept": "application/json",
}

OUT_PATH = Path("data_raw/titles_generated.txt")

# ── Categories to harvest ─────────────────────────────────────────────────────
# Each entry is a Wikipedia category name (without the "Category:" prefix).
# Organised by theme for readability — order doesn't matter for the output.

CATEGORIES = [
    # Core ML concepts
    "Machine learning",
    "Deep learning",
    "Supervised learning",
    "Unsupervised learning",
    "Reinforcement learning",
    "Semi-supervised learning",
    "Self-supervised learning",

    # Neural networks
    "Artificial neural networks",
    "Convolutional neural networks",
    "Recurrent neural networks",

    # NLP
    "Natural language processing",
    "Computational linguistics",
    "Information retrieval",

    # Statistics & maths foundations
    "Statistical classification",
    "Regression analysis",
    "Bayesian statistics",
    "Statistical learning theory",
    "Probabilistic models",
    "Dimensionality reduction",
    "Cluster analysis",

    # AI broadly
    "Artificial intelligence",
    "History of artificial intelligence",
    "Applications of artificial intelligence",

    # People — this is what catches Marvin Minsky, Turing, Hinton, etc.
    "Artificial intelligence researchers",
    "Machine learning researchers",
    "Computer scientists",
    "Mathematicians",

    # Evaluation & methodology
    "Model selection",
    "Cross-validation (statistics)",
]

# ── Exclude patterns ──────────────────────────────────────────────────────────
EXCLUDE_SUBSTRINGS = [
    "(disambiguation)",
    "List of",
    "Outline of",
    "Index of",
    "Template:",
    "Category:",
    "Wikipedia:",
    "Portal:",
    "Talk:",
]


def _should_exclude(title: str) -> bool:
    return any(excl.lower() in title.lower() for excl in EXCLUDE_SUBSTRINGS)


# ── Wikipedia Category API ────────────────────────────────────────────────────

def fetch_category_members(category: str, max_titles: int = 200) -> list[str]:
    """
    Fetch article titles that are direct members of *category*.
    Uses cmtype=page to skip subcategories and files.
    Handles API continuation automatically.
    """
    titles: list[str] = []
    cmcontinue: str | None = None

    while len(titles) < max_titles:
        params: dict = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "page",
            "cmlimit": min(500, max_titles - len(titles)),
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        try:
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  WARNING: API error for '{category}': {e}")
            break

        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            t = m.get("title", "").strip()
            if t and not _should_exclude(t):
                titles.append(t)

        cont = data.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue:
            break

        time.sleep(0.2)

    return titles


# ── Main ──────────────────────────────────────────────────────────────────────

def main(max_per_category: int = 200, dry_run: bool = False) -> None:
    all_titles: set[str] = set()

    print(f"Harvesting {len(CATEGORIES)} categories (max {max_per_category} per category)…\n")

    for category in CATEGORIES:
        members = fetch_category_members(category, max_titles=max_per_category)
        new = [t for t in members if t not in all_titles]
        all_titles.update(members)
        print(f"  [{len(all_titles):>4} total]  {category}: +{len(new)} new ({len(members)} fetched)")
        time.sleep(0.3)

    sorted_titles = sorted(all_titles, key=str.lower)

    print(f"\nTotal unique titles: {len(sorted_titles)}")

    if dry_run:
        print("Dry run — not writing file.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(sorted_titles) + "\n", encoding="utf-8")
    print(f"Written to: {OUT_PATH}")
    print("\nSample titles:")
    for t in sorted_titles[:10]:
        print(f"  {t}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ML/AI Wikipedia title list via Category API.")
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=200,
        help="Max articles to fetch per category (default: 200).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing the output file.",
    )
    args = parser.parse_args()
    main(max_per_category=args.max_per_category, dry_run=args.dry_run)
