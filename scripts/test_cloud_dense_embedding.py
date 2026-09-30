from __future__ import annotations

from app.config.settings import CloudConfig
from app.embeddings.cloud_dense_embedding_service import (
    CloudDenseEmbeddingService,
)


def main() -> None:

    config = CloudConfig()

    service = CloudDenseEmbeddingService(
        cloud_config=config,
    )

    print("\nTesting hosted dense embedding...")

    query = "What does ef_construct control in HNSW?"

    vector = service.embed_query(query)

    print(f"Provider model : {config.embedding_model}")
    print(f"Expected dims  : {config.embedding_dimension}")
    print(f"Actual dims    : {len(vector)}")
    print(f"First 5 values : {vector[:5]}")

    if len(vector) != config.embedding_dimension:
        raise RuntimeError("Hosted embedding dimension mismatch.")

    documents = [
        "ef_construct controls graph construction effort.",
        "The M parameter controls HNSW connectivity.",
    ]

    document_vectors = service.embed_documents(documents)

    print(f"Document vectors returned: " f"{len(document_vectors)}")

    for index, embedding in enumerate(document_vectors):
        if len(embedding) != config.embedding_dimension:
            raise RuntimeError(f"Document {index} dimension mismatch.")

    print("\nCloud dense embedding smoke test PASSED.")


if __name__ == "__main__":
    main()
