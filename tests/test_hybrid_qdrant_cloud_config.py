from unittest.mock import patch

import pytest

from app.config.settings import (
    HybridSearchConfig,
    QdrantConfig,
    QdrantDeployment,
)

from app.vectorstore.hybrid_qdrant_store import (
    HybridQdrantStore,
)


@patch("app.vectorstore.hybrid_qdrant_store.QdrantClient")
def test_local_qdrant_client_configuration(
    mock_qdrant_client,
) -> None:

    qdrant_config = QdrantConfig(
        deployment=QdrantDeployment.LOCAL,
        local_url="http://localhost:6333",
        timeout_seconds=60,
    )

    HybridQdrantStore(
        config=HybridSearchConfig(),
        vector_size=384,
        qdrant_config=qdrant_config,
    )

    mock_qdrant_client.assert_called_once_with(
        url="http://localhost:6333",
        timeout=60,
    )


@patch("app.vectorstore.hybrid_qdrant_store.QdrantClient")
def test_cloud_qdrant_client_configuration(
    mock_qdrant_client,
) -> None:

    qdrant_config = QdrantConfig(
        deployment=QdrantDeployment.CLOUD,
        cloud_url="https://example.cloud.qdrant.io",
        api_key="test-key",
        cloud_inference=True,
        timeout_seconds=60,
    )

    HybridQdrantStore(
        config=HybridSearchConfig(),
        vector_size=384,
        qdrant_config=qdrant_config,
    )

    mock_qdrant_client.assert_called_once_with(
        url="https://example.cloud.qdrant.io",
        api_key="test-key",
        cloud_inference=True,
        timeout=60,
    )


def test_cloud_qdrant_requires_url() -> None:

    qdrant_config = QdrantConfig(
        deployment=QdrantDeployment.CLOUD,
        cloud_url=None,
        api_key="test-key",
    )

    with pytest.raises(
        ValueError,
        match="QDRANT_URL is required",
    ):
        HybridQdrantStore(
            config=HybridSearchConfig(),
            vector_size=384,
            qdrant_config=qdrant_config,
        )


def test_cloud_qdrant_requires_api_key() -> None:

    qdrant_config = QdrantConfig(
        deployment=QdrantDeployment.CLOUD,
        cloud_url="https://example.cloud.qdrant.io",
        api_key=None,
    )

    with pytest.raises(
        ValueError,
        match="QDRANT_API_KEY is required",
    ):
        HybridQdrantStore(
            config=HybridSearchConfig(),
            vector_size=384,
            qdrant_config=qdrant_config,
        )
