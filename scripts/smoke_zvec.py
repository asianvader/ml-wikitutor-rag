"""
Smoke test for:
1. OpenAI embeddings
2. Zvec collection creation
3. Vector insertion
4. Vector similarity search

This script does NOT use Wikipedia yet.
It inserts 3 dummy ML-related documents and queries them.

If this works, your environment, embeddings, and Zvec are wired correctly.
"""

from dotenv import load_dotenv
load_dotenv()  

import zvec
from langchain_openai import OpenAIEmbeddings

# -----------------------------
# Configuration
# -----------------------------

# Where Zvec will store its local collection files
COLLECTION_PATH = "index/zvec_demo"

# Embedding model (OpenAI)
EMBEDDING_MODEL = "text-embedding-3-small"

# Default dimension for text-embedding-3-small
EMBEDDING_DIM = 1536


def main():
    # --------------------------------------------------
    # 1️⃣ Initialize embedding model
    # --------------------------------------------------
    # This is responsible for converting text -> vector
    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # --------------------------------------------------
    # 2️⃣ Define Zvec schema
    # --------------------------------------------------
    # We define:
    # - Scalar fields (text, title)
    # - One vector field (text_embedding)
    # - Cosine similarity metric
    schema = zvec.CollectionSchema(
        name="demo",
        fields=[
            zvec.FieldSchema(name="text", data_type=zvec.DataType.STRING),
            zvec.FieldSchema(name="title", data_type=zvec.DataType.STRING, nullable=True),
        ],
        vectors=[
            zvec.VectorSchema(
                name="text_embedding",
                data_type=zvec.DataType.VECTOR_FP32,
                dimension=EMBEDDING_DIM,
                index_param=zvec.HnswIndexParam(
                    metric_type=zvec.MetricType.COSINE
                ),
            )
        ],
    )

    # Collection options
    option = zvec.CollectionOption(
        read_only=False,
        enable_mmap=True
    )

    # --------------------------------------------------
    # 3️⃣ Create or open the collection
    # --------------------------------------------------
    # First run → creates collection
    # Subsequent runs → opens existing collection
    try:
        col = zvec.create_and_open(
            path=COLLECTION_PATH,
            schema=schema,
            option=option,
        )
        print("Created new collection.")
    except Exception:
        col = zvec.open(path=COLLECTION_PATH, option=option)
        print("Opened existing collection.")

    # --------------------------------------------------
    # 4️⃣ Prepare dummy documents
    # --------------------------------------------------
    # Tuple format: (title, text content)
    documents = [
        ("Regularization", "L2 regularization adds a squared penalty on weights."),
        ("Decision Trees", "A decision tree splits data to reduce impurity."),
        ("PCA", "PCA projects data onto directions of maximum variance."),
    ]

    # Extract text for embedding
    texts = [doc[1] for doc in documents]

    # --------------------------------------------------
    # 5️⃣ Convert text -> vectors
    # --------------------------------------------------
    vectors = emb.embed_documents(texts)

    # --------------------------------------------------
    # 6️⃣ Convert into Zvec Doc objects
    # --------------------------------------------------
    zdocs = []
    for i, ((title, text), vector) in enumerate(zip(documents, vectors)):
        zdocs.append(
            zvec.Doc(
                id=f"demo_{i}",  # Unique document ID
                vectors={"text_embedding": vector},
                fields={
                    "text": text,
                    "title": title,
                },
            )
        )

    # --------------------------------------------------
    # 7️⃣ Insert into collection
    # --------------------------------------------------
    col.insert(zdocs)

    # Optimize index after bulk insert (improves query performance)
    col.optimize()

    print("Inserted documents into collection.")

    # --------------------------------------------------
    # 8️⃣ Perform a similarity search
    # --------------------------------------------------
    query_text = "What is L2 regularization?"
    query_vector = emb.embed_query(query_text)

    # Vector query against "text_embedding" field
    results = col.query(
        vectors=zvec.VectorQuery(
            field_name="text_embedding",
            vector=query_vector
        ),
        topk=3,
    )

    # --------------------------------------------------
    # 9️⃣ Print results
    # --------------------------------------------------
    print("\nQuery:", query_text)
    print("Top results:")

    for r in results:
        print(
            f"- {r.fields.get('title')} | "
            f"Score: {getattr(r, 'score', None)}"
        )


if __name__ == "__main__":
    main()