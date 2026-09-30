from __future__ import annotations

from app.config.settings import (
    CloudConfig,
    RerankerProvider,
    RerankingConfig,
)

from app.reranking.base import Reranker
from app.reranking.cross_encoder_reranker import (
    LocalCrossEncoderReranker,
)
from app.reranking.cloud_reranker import CloudReranker


def create_reranker(
    config: RerankingConfig,
    cloud_config: CloudConfig | None = None,
) -> Reranker:
    """
    Create the configured reranking provider.
    """

    if config.provider == RerankerProvider.LOCAL:
        return LocalCrossEncoderReranker(
            model_name=config.model_name,
        )

    if config.provider == RerankerProvider.CLOUD:
        if cloud_config is None:
            raise ValueError("CloudConfig is required " "for cloud reranking.")

        return CloudReranker(
            reranking_config=config,
            cloud_config=cloud_config,
        )

    raise ValueError(f"Unsupported reranker provider: " f"{config.provider}")
