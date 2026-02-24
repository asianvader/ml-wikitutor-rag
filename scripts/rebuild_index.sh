#!/usr/bin/env bash
set -euo pipefail

# 1) Ingest all titles -> chunks.jsonl
uv run python -m src.ingest_wiki_api

# 2) Rebuild Zvec from scratch (schema is fixed, easiest is clean rebuild)
rm -rf index/zvec_wiki_ml
uv run python -m src.index_zvec