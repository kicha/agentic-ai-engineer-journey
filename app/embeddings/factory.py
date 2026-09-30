from __future__ import annotations

from app.config.settings import (
    CloudConfig,
    DenseEmbeddingProvider,
    EmbeddingConfig,
    SparseEmbeddingProvider,
)
from app.embeddings.base import DenseEmbeddingService, SparseEmbeddingService
from app.embeddings.cloud_dense_embedding_service import CloudDenseEmbeddingService
from app.embeddings.cloud_sparse_embedding_service import CloudSparseEmbeddingService
from app.embeddings.local_dense_embedding_service import LocalDenseEmbeddingService
from app.embeddings.local_sparse_embedding_service import LocalSparseEmbeddingService


def create_dense_embedding_service(
    config: EmbeddingConfig,
    cloud_config: CloudConfig | None = None,
) -> DenseEmbeddingService:
    """Create the configured dense embedding provider."""

    if config.dense_provider == DenseEmbeddingProvider.LOCAL:
        return LocalDenseEmbeddingService(model_name=config.local_dense_model)

    if config.dense_provider == DenseEmbeddingProvider.CLOUD:
        if cloud_config is None:
            raise ValueError("CloudConfig is required for cloud dense embeddings.")
        return CloudDenseEmbeddingService(cloud_config=cloud_config)

    raise ValueError(f"Unsupported dense embedding provider: {config.dense_provider}")


def create_sparse_embedding_service(
    config: EmbeddingConfig,
) -> SparseEmbeddingService:
    """Create the configured sparse embedding provider."""

    if config.sparse_provider == SparseEmbeddingProvider.LOCAL:
        return LocalSparseEmbeddingService(model_name=config.sparse_model)

    if config.sparse_provider == SparseEmbeddingProvider.QDRANT_CLOUD:
        return CloudSparseEmbeddingService(model_name=config.sparse_model)

    raise ValueError(f"Unsupported sparse embedding provider: {config.sparse_provider}")
