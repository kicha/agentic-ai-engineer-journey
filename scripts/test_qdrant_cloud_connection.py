from __future__ import annotations
import logging
import uuid

from qdrant_client import QdrantClient, models
from app.utils.logging_config import configure_logging
from app.config.settings import (
    QdrantConfig,
    QdrantDeployment,
)

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    config = QdrantConfig(
        deployment=QdrantDeployment.CLOUD,
    )

    if not config.cloud_url:
        raise RuntimeError("QDRANT_URL is missing from environment.")

    if not config.api_key:
        raise RuntimeError("QDRANT_API_KEY is missing from environment.")

    collection_name = f"cloud_smoke_test_{uuid.uuid4().hex[:8]}"

    client = QdrantClient(
        url=config.cloud_url,
        api_key=config.api_key,
        cloud_inference=False,
        timeout=config.timeout_seconds,
    )

    logger.info("\nConnecting to Qdrant Cloud...")

    collections = client.get_collections()

    logger.info(
        f"Connection successful. "
        f"Existing collections: "
        f"{len(collections.collections)}"
    )

    try:
        logger.info(f"\nCreating temporary collection: " f"{collection_name}")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=4,
                distance=models.Distance.COSINE,
            ),
        )

        exists = client.collection_exists(collection_name=collection_name)

        if not exists:
            raise RuntimeError("Temporary collection was not created.")

        logger.info("Temporary collection created successfully.")

    finally:

        if client.collection_exists(collection_name=collection_name):
            logger.info(f"\nDeleting temporary collection: " f"{collection_name}")

            client.delete_collection(collection_name=collection_name)

            logger.info("Temporary collection deleted.")

    logger.info("\nQdrant Cloud smoke test PASSED.")


if __name__ == "__main__":
    main()
