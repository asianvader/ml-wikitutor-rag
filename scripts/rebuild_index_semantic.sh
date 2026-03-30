#!/usr/bin/env bash
set -euo pipefail

# 0) Regenerate ML/AI title list from Wikipedia Category API
#    Output: data_raw/titles_generated.txt
#    (data_raw/titles.txt with off-topic noise articles is left untouched)
echo "=== Generating ML/AI titles from Wikipedia Category API ==="
uv run python -m src.generate_titles

# 1) Ingest all titles using semantic chunker → chunks_semantic.jsonl
#    Automatically combines titles_generated.txt (ML) + titles.txt (noise)
echo "=== Ingesting articles (semantic chunker) ==="
uv run python -m src.ingest_wiki_api --chunker semantic

# 2) Upsert into the Qdrant Cloud semantic collection (wiki_ml_semantic)
echo "=== Indexing into Qdrant (semantic chunker) ==="
uv run python -m src.index_qdrant \
  --chunks-file data_processed/chunks_semantic.jsonl \
  --chunker semantic
