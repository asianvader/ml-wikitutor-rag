#!/usr/bin/env bash
set -euo pipefail

# 0) Regenerate ML/AI title list from Wikipedia Category API
#    Output: data_raw/titles_generated.txt
#    (data_raw/titles.txt with off-topic noise articles is left untouched)
echo "=== Generating ML/AI titles from Wikipedia Category API ==="
uv run python -m src.generate_titles

# 1) Ingest all titles -> chunks_parent_child.jsonl
#    Automatically combines titles_generated.txt (ML) + titles.txt (noise)
echo "=== Ingesting articles (parent-child chunker) ==="
uv run python -m src.ingest_wiki_api --chunker parent_child

# 2) Rebuild Zvec from scratch
echo "=== Rebuilding parent-child index ==="
rm -rf index/zvec_wiki_ml_parent_child
uv run python -m src.index_zvec \
    --chunks-file data_processed/chunks_parent_child.jsonl \
    --index-path index/zvec_wiki_ml_parent_child
