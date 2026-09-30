from unittest.mock import patch

import pytest

from app.config.settings import (
    CloudConfig,
    DenseEmbeddingProvider,
    EmbeddingConfig,
    SparseEmbeddingProvider,
)
from app.embeddings.factory import (
    create_dense_embedding_service,
    create_sparse_embedding_service,
)


@patch("app.embeddings.factory.LocalDenseEmbeddingService")
def test_dense_factory_creates_local_provider(mock_local_dense) -> None:
    config = EmbeddingConfig(dense_provider=DenseEmbeddingProvider.LOCAL)
    service = create_dense_embedding_service(config)
    mock_local_dense.assert_called_once_with(model_name=config.local_dense_model)
    assert service is mock_local_dense.return_value


@patch("app.embeddings.factory.CloudDenseEmbeddingService")
def test_dense_factory_creates_cloud_provider(mock_cloud_dense) -> None:
    config = EmbeddingConfig(dense_provider=DenseEmbeddingProvider.CLOUD)
    cloud_config = CloudConfig(api_key="test-key")
    service = create_dense_embedding_service(config, cloud_config=cloud_config)
    mock_cloud_dense.assert_called_once_with(cloud_config=cloud_config)
    assert service is mock_cloud_dense.return_value


def test_dense_factory_requires_cloud_config() -> None:
    config = EmbeddingConfig(dense_provider=DenseEmbeddingProvider.CLOUD)
    with pytest.raises(ValueError, match="CloudConfig is required"):
        create_dense_embedding_service(config)


@patch("app.embeddings.factory.LocalSparseEmbeddingService")
def test_sparse_factory_creates_local_provider(mock_local_sparse) -> None:
    config = EmbeddingConfig(sparse_provider=SparseEmbeddingProvider.LOCAL)
    service = create_sparse_embedding_service(config)
    mock_local_sparse.assert_called_once_with(model_name=config.sparse_model)
    assert service is mock_local_sparse.return_value


@patch("app.embeddings.factory.CloudSparseEmbeddingService")
def test_sparse_factory_creates_qdrant_cloud_provider(mock_cloud_sparse) -> None:
    config = EmbeddingConfig(sparse_provider=SparseEmbeddingProvider.QDRANT_CLOUD)
    service = create_sparse_embedding_service(config)
    mock_cloud_sparse.assert_called_once_with(model_name=config.sparse_model)
    assert service is mock_cloud_sparse.return_value
