#!/usr/bin/env bash
set -euo pipefail

# 0) Regenerate ML/AI title list from Wikipedia Category API
#    Output: data_raw/titles_generated.txt
#    (data_raw/titles.txt with off-topic noise articles is left untouched)
echo "=== Generating ML/AI titles from Wikipedia Category API ==="
uv run python -m src.generate_titles

# 1) Ingest all titles -> chunks.jsonl
#    Automatically combines titles_generated.txt (ML) + titles.txt (noise)
echo "=== Ingesting articles (token chunker) ==="
uv run python -m src.ingest_wiki_api

# 2) Upsert into the Qdrant Cloud token collection (wiki_ml_token)
echo "=== Indexing into Qdrant (token chunker) ==="
uv run python -m src.index_qdrant --chunker token
