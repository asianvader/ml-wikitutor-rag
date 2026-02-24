import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import zvec
from langchain_openai import OpenAIEmbeddings

CHUNKS_PATH = Path("data_processed/chunks.jsonl")
ZVEC_PATH = "index/zvec_wiki_ml"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
VECTOR_FIELD = "text_embedding"


def create_or_open_collection(path: str):
    schema = zvec.CollectionSchema(
        name="wiki_ml",
        fields=[
            zvec.FieldSchema(name="text", data_type=zvec.DataType.STRING),
            zvec.FieldSchema(name="title", data_type=zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema(name="section", data_type=zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema(name="source_url", data_type=zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema(name="chunk_id", data_type=zvec.DataType.STRING, nullable=True),
        ],
        vectors=[
            zvec.VectorSchema(
                name=VECTOR_FIELD,
                data_type=zvec.DataType.VECTOR_FP32,
                dimension=EMBEDDING_DIM,
                index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
            )
        ],
    )

    option = zvec.CollectionOption(read_only=False, enable_mmap=True)

    try:
        col = zvec.create_and_open(path=path, schema=schema, option=option)
        print("Created new Zvec collection:", path)
    except Exception:
        col = zvec.open(path=path, option=option)
        print("Opened existing Zvec collection:", path)

    return col


def load_chunks(limit: int | None = None):
    rows = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def main(limit: int | None = None, batch_size: int = 64):
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing {CHUNKS_PATH}. Run ingest_wiki_api first.")

    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    col = create_or_open_collection(ZVEC_PATH)

    chunks = load_chunks(limit=limit)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    # Insert in batches to reduce API calls overhead and memory spikes
    inserted = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [r["text"] for r in batch]
        vectors = emb.embed_documents(texts)

        zdocs = []
        for r, v in zip(batch, vectors):
            zdocs.append(
                zvec.Doc(
                    id=r["id"],
                    vectors={VECTOR_FIELD: v},
                    fields={
                        "text": r["text"],
                        "title": r.get("title"),
                        "section": r.get("section"),
                        "source_url": r.get("source_url"),
                        "chunk_id": str(r.get("chunk_index")),
                    },
                )
            )

        col.insert(zdocs)
        inserted += len(zdocs)
        print(f"Inserted {inserted}/{len(chunks)}")

    # Optimize once after bulk insert
    col.optimize()
    print("Optimized collection.")

    # Quick retrieval test
    query = "Explain the difference between supervised and unsupervised learning."
    qv = emb.embed_query(query)
    hits = col.query(
    vectors=zvec.VectorQuery(field_name=VECTOR_FIELD, vector=qv),
    topk=5
)

    print("\nQuery:", query)
    for h in hits:
        title = h.fields.get("title")
        score = getattr(h, "score", None)
        url = h.fields.get("source_url")
        preview = (h.fields.get("text") or "")[:140].replace("\n", " ")
        print(f"- {title} | score={score} | {preview} | {url}")


if __name__ == "__main__":
    main()