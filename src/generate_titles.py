import time
from pathlib import Path
import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "ml-wikitutor-rag/0.1 (https://github.com/<you>/ml-wikitutor-rag)",
    "Accept": "application/json",
}

SEEDS_PATH = Path("data_raw/titles.txt")
OUT_PATH = Path("data_raw/titles_generated.txt")

# Keywords to keep titles on-topic (tweak as you like)
KEEP_KEYWORDS = [
    "learning", "regression", "classification", "clustering", "neural", "network",
    "bayes", "probability", "statistics", "feature", "embedding", "transformer",
    "optimization", "gradient", "loss", "regularization", "dimensionality",
    "principal component", "inference", "decision tree", "random forest", "boost",
    "support vector", "svm", "precision", "recall", "confusion", "roc", "auc",
    "natural language", "information retrieval", "data mining", "deep learning",
]

# Titles to exclude (disambiguations and obvious off-topic)
EXCLUDE_SUBSTRINGS = ["(disambiguation)", "List of", "Outline of"]

def fetch_links(title: str, limit: int = 200) -> list[str]:
    """Fetch outgoing links from a Wikipedia page title."""
    links = []
    plcontinue = None

    while True:
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "links",
            "plnamespace": 0,      # only main/article namespace
            "pllimit": min(limit, 500),
            "redirects": 1,
        }
        if plcontinue:
            params["plcontinue"] = plcontinue

        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()

        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        page_links = page.get("links", []) or []
        links.extend([x["title"] for x in page_links if "title" in x])

        cont = data.get("continue", {})
        plcontinue = cont.get("plcontinue")
        if not plcontinue or len(links) >= limit:
            break

        time.sleep(0.2)

    return links[:limit]

def is_good(title: str) -> bool:
    t = title.lower()
    if any(x.lower() in title for x in EXCLUDE_SUBSTRINGS):
        return False
    return any(k in t for k in KEEP_KEYWORDS)

def main(
    per_seed_links: int = 200,
    target_total: int = 120,
):
    seeds = [s.strip() for s in SEEDS_PATH.read_text(encoding="utf-8").splitlines() if s.strip()]
    all_titles = set(seeds)

    for s in seeds:
        try:
            links = fetch_links(s, limit=per_seed_links)
        except Exception as e:
            print(f"Failed on seed '{s}': {type(e).__name__} {e}")
            continue

        for t in links:
            if is_good(t):
                all_titles.add(t)

        print(f"{s}: collected {len(all_titles)} titles so far")
        if len(all_titles) >= target_total:
            break

        time.sleep(0.2)

    # Write a deterministic, readable file
    out = sorted(all_titles, key=str.lower)
    OUT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nWrote {len(out)} titles to {OUT_PATH}")

if __name__ == "__main__":
    main()