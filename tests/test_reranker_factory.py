from unittest.mock import patch

import pytest

from app.config.settings import (
    CloudConfig,
    RerankerProvider,
    RerankingConfig,
)

from app.reranking.factory import (
    create_reranker,
)


@patch("app.reranking.factory.LocalCrossEncoderReranker")
def test_reranker_factory_creates_local_provider(
    mock_local_reranker,
) -> None:

    config = RerankingConfig(
        provider=RerankerProvider.LOCAL,
    )

    reranker = create_reranker(config)

    mock_local_reranker.assert_called_once_with(
        model_name=config.model_name,
    )

    assert reranker is mock_local_reranker.return_value


def test_reranker_factory_requires_cloud_config_for_cloud_provider() -> None:
    """
    Cloud reranking requires cloud connection configuration.

    The factory must fail clearly when CLOUD is selected without
    the corresponding CloudConfig.
    """

    config = RerankingConfig(
        provider=RerankerProvider.CLOUD,
        model_name="jina-reranker-v2-base-multilingual",
    )

    with pytest.raises(
        ValueError,
        match=r"CloudConfig is required for cloud reranking\.",
    ):
        create_reranker(
            config=config,
            cloud_config=None,
        )


@patch("app.reranking.factory.CloudReranker")
def test_reranker_factory_creates_cloud_provider(
    mock_cloud_reranker,
) -> None:

    reranking_config = RerankingConfig(
        provider=RerankerProvider.CLOUD,
        model_name=("jina-reranker-v2-base-multilingual"),
    )

    cloud_config = CloudConfig(
        api_key="test-api-key",
    )

    reranker = create_reranker(
        config=reranking_config,
        cloud_config=cloud_config,
    )

    mock_cloud_reranker.assert_called_once_with(
        reranking_config=reranking_config,
        cloud_config=cloud_config,
    )

    assert reranker is mock_cloud_reranker.return_value
