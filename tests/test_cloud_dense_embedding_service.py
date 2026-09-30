from unittest.mock import MagicMock, patch

import pytest

from app.config.settings import CloudConfig
from app.embeddings.cloud_dense_embedding_service import (
    CloudDenseEmbeddingService,
)


def make_config() -> CloudConfig:
    return CloudConfig(
        embedding_model="jina-embeddings-v4",
        embedding_dimension=3,
        cloud_url="https://example.test/embeddings",
        api_key="test-key",
        timeout_seconds=10.0,
    )


@patch("app.embeddings.cloud_dense_embedding_service.httpx.Client")
def test_embed_query_uses_retrieval_query(
    mock_client_cls,
) -> None:

    response = MagicMock()

    response.json.return_value = {
        "data": [
            {
                "index": 0,
                "embedding": [0.1, 0.2, 0.3],
            }
        ]
    }

    mock_client = mock_client_cls.return_value.__enter__.return_value

    mock_client.post.return_value = response

    service = CloudDenseEmbeddingService(make_config())

    result = service.embed_query("What is HNSW?")

    assert result == [0.1, 0.2, 0.3]

    _, kwargs = mock_client.post.call_args

    assert kwargs["json"]["task"] == "retrieval.query"

    assert kwargs["json"]["input"] == ["What is HNSW?"]

    response.raise_for_status.assert_called_once()


@patch("app.embeddings.cloud_dense_embedding_service.httpx.Client")
def test_embed_documents_uses_retrieval_passage(
    mock_client_cls,
) -> None:

    response = MagicMock()

    response.json.return_value = {
        "data": [
            {
                "index": 0,
                "embedding": [0.1, 0.2, 0.3],
            },
            {
                "index": 1,
                "embedding": [0.4, 0.5, 0.6],
            },
        ]
    }

    mock_client = mock_client_cls.return_value.__enter__.return_value

    mock_client.post.return_value = response

    service = CloudDenseEmbeddingService(make_config())

    result = service.embed_documents(
        [
            "Document one",
            "Document two",
        ]
    )

    assert len(result) == 2

    _, kwargs = mock_client.post.call_args

    assert kwargs["json"]["task"] == "retrieval.passage"


def test_cloud_dense_requires_api_key() -> None:

    config = CloudConfig(api_key=None)

    with pytest.raises(
        ValueError,
        match="CLOUD_EMBEDDING_API_KEY",
    ):
        CloudDenseEmbeddingService(config)


def test_embed_query_rejects_empty_query() -> None:

    service = CloudDenseEmbeddingService(make_config())

    with pytest.raises(
        ValueError,
        match="Query must not be empty",
    ):
        service.embed_query("   ")
