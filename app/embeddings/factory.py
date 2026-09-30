from __future__ import annotations

from app.config.settings import (
    ChunkingConfig,
    CloudConfig,
    DenseEmbeddingProvider,
    SparseEmbeddingProvider,
)

from app.embeddings.base import (
    DenseEmbeddingService,
    SparseEmbeddingService,
)

from app.embeddings.local_dense_embedding_service import (
    LocalDenseEmbeddingService,
)

from app.embeddings.local_sparse_embedding_service import (
    LocalSparseEmbeddingService,
)

from app.embeddings.cloud_dense_embedding_service import (
    CloudDenseEmbeddingService,
)


def create_dense_embedding_service(
    config: ChunkingConfig,
    cloud_config: CloudConfig | None = None,
) -> DenseEmbeddingService:
    """
    Create the configured dense embedding provider.
    """

    if config.dense_embedding_provider == DenseEmbeddingProvider.LOCAL:
        return LocalDenseEmbeddingService(
            model_name=config.embedding_model_name,
        )

    if config.dense_embedding_provider == DenseEmbeddingProvider.CLOUD:
        if cloud_config is None:
            raise ValueError("CloudConfig is required " "for cloud dense embeddings.")

        return CloudDenseEmbeddingService(cloud_config=cloud_config)

    raise ValueError(
        "Unsupported dense embedding provider: " f"{config.dense_embedding_provider}"
    )


def create_sparse_embedding_service(
    config: ChunkingConfig,
) -> SparseEmbeddingService:
    """
    Create the configured sparse embedding provider.
    """

    if config.sparse_embedding_provider == SparseEmbeddingProvider.LOCAL:
        return LocalSparseEmbeddingService(
            model_name=config.sparse_embedding_model,
        )

    if config.sparse_embedding_provider == SparseEmbeddingProvider.QDRANT_CLOUD:
        raise NotImplementedError(
            "Cloud sparse embedding provider " "has not been implemented yet."
        )

    raise ValueError(
        "Unsupported sparse embedding provider: " f"{config.sparse_embedding_provider}"
    )
