from src.chunk import chunk_text

SAMPLE = """
Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function.

L2 regularization adds the squared magnitude of coefficients as a penalty term.

L1 regularization adds the absolute magnitude of coefficients as a penalty term and can encourage sparsity.

In practice, the choice depends on your modeling goals and data properties.
"""

def main():
    chunks = chunk_text(SAMPLE, chunk_size=80, overlap=10)  # small numbers to see behavior quickly
    print(f"Chunks: {len(chunks)}")
    for c in chunks:
        print(f"\n--- chunk {c.chunk_index} ({c.token_count} tokens) ---\n{c.text}")

if __name__ == "__main__":
    main()