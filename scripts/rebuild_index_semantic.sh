#!/usr/bin/env bash
set -euo pipefail

# 1) Ingest all titles using semantic chunker → chunks_semantic.jsonl
uv run python -m src.ingest_wiki_api --chunker semantic

# 2) Rebuild Zvec index for semantic chunks
rm -rf index/zvec_wiki_ml_semantic
uv run python -m src.index_zvec \
  --chunks-file data_processed/chunks_semantic.jsonl \
  --index-path index/zvec_wiki_ml_semantic
