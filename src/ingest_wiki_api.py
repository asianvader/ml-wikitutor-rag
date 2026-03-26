import argparse
import json
from pathlib import Path
import requests
import time

from src.chunk import chunk_text

WIKI_API = "https://en.wikipedia.org/w/api.php"

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

HEADERS = {
    # Use a descriptive UA. Wikipedia recommends identifying your app + contact.
    "User-Agent": "ml-wikitutor-rag/0.1 (education project; contact: phoebe.voong@gmail.com)",
    "Accept": "application/json",
}


def fetch_wikipedia_extract(title: str) -> dict:
    """
    Fetch plain text extract for a Wikipedia page title.
    Uses MediaWiki API 'extracts' (plaintext).
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
    }
    # Small delay helps avoid rate/anti-bot triggers
    time.sleep(0.2)

    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    page_id = page.get("pageid")
    extract = page.get("extract", "") or ""
    url = f"https://en.wikipedia.org/?curid={page_id}" if page_id else None

    return {
        "title": page.get("title", title),
        "page_id": page_id,
        "url": url,
        "text": extract,
    }


def main(limit: int | None = None, chunker: str = "token"):
    titles_path = DATA_RAW / "titles.txt"
    titles = [t.strip() for t in titles_path.read_text(encoding="utf-8").splitlines() if t.strip()]
    if limit is not None:
        titles = titles[:limit]

    # Choose output path and chunking function based on strategy
    if chunker == "semantic":
        from src.chunk_semantic import chunk_text_semantic
        from langchain_openai import OpenAIEmbeddings
        from dotenv import load_dotenv
        load_dotenv()
        emb = OpenAIEmbeddings(model="text-embedding-3-small")
        out_path = DATA_PROCESSED / "chunks_semantic.jsonl"
        print(f"Using semantic chunker → {out_path}")
    else:
        out_path = DATA_PROCESSED / "chunks.jsonl"
        print(f"Using token chunker → {out_path}")

    total_chunks = 0
    token_counts = []

    with out_path.open("w", encoding="utf-8") as f:
        for title in titles:
            page = fetch_wikipedia_extract(title)
            text = page["text"].strip()

            # Skip empty pages (rare, but happens)
            if not text:
                continue

            if chunker == "semantic":
                chunks = chunk_text_semantic(text, embeddings=emb)
            else:
                chunks = chunk_text(text, chunk_size=500, overlap=80)

            for c in chunks:
                record = {
                    "id": f"wiki_{page['page_id']}_{c.chunk_index}",
                    "title": page["title"],
                    "section": None,
                    "source_url": page["url"],
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "text": c.text,
                    "chunker": chunker,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_chunks += len(chunks)
            token_counts.extend([c.token_count for c in chunks])

            print(f"{page['title']}: {len(chunks)} chunks")

    if token_counts:
        print("\n--- Summary ---")
        print("Chunks written:", total_chunks)
        print("Avg tokens/chunk:", round(sum(token_counts) / len(token_counts), 1))
        print("Min tokens/chunk:", min(token_counts))
        print("Max tokens/chunk:", max(token_counts))
        print("Output:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Wikipedia articles into chunks.")
    parser.add_argument(
        "--chunker",
        choices=["token", "semantic"],
        default="token",
        help="Chunking strategy: 'token' (default) or 'semantic'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N titles (useful for testing).",
    )
    args = parser.parse_args()
    main(limit=args.limit, chunker=args.chunker)
