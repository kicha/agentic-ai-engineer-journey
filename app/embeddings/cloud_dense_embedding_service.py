from __future__ import annotations
import logging
from typing import Sequence
import time
import httpx

from app.embeddings.base import DenseEmbeddingService, DenseVector
from app.config.settings import CloudConfig

logger = logging.getLogger(__name__)


class CloudDenseEmbeddingService(DenseEmbeddingService):
    """
    Hosted dense embedding provider.

    The concrete cloud vendor is selected through configuration.
    """

    def __init__(
        self,
        cloud_config: CloudConfig,
    ) -> None:

        self.model_name = cloud_config.embedding_model
        self._embedding_dimension = cloud_config.embedding_dimension
        self.embedding_url = cloud_config.embedding_url
        self.api_key = cloud_config.api_key
        self.timeout_seconds = cloud_config.timeout_seconds
        self.max_retries = cloud_config.max_retries
        self.retry_backoff_seconds = cloud_config.retry_backoff_seconds

        if not self.api_key:
            raise ValueError("CLOUD_EMBEDDING_API_KEY is required.")

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    def embed_query(
        self,
        query: str,
    ) -> DenseVector:

        query = query.strip()

        if not query:
            raise ValueError("Query must not be empty.")

        embeddings = self._embed(
            texts=[query],
            task="retrieval.query",
        )

        return embeddings[0]

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[DenseVector]:

        cleaned_texts = [text.strip() for text in texts if text and text.strip()]

        if not cleaned_texts:
            return []

        return self._embed(
            texts=cleaned_texts,
            task="retrieval.passage",
        )

    def _embed(
        self,
        *,
        texts: Sequence[str],
        task: str,
    ) -> list[DenseVector]:

        payload = {
            "model": self.model_name,
            "task": task,
            "dimensions": (self._embedding_dimension),
            "normalized": True,
            "embedding_type": "float",
            "input": list(texts),
        }

        headers = {
            "Authorization": (f"Bearer {self.api_key}"),
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):

            try:
                with httpx.Client(
                    timeout=self.timeout_seconds,
                ) as client:

                    response = client.post(
                        self.embedding_url,
                        headers=headers,
                        json=payload,
                    )

                    response.raise_for_status()

                body = response.json()

                break

            except httpx.HTTPStatusError as exc:

                last_error = exc

                status_code = (
                    exc.response.status_code
                )

                retryable_status_codes = {
                    429,
                    500,
                    502,
                    503,
                    504,
                }

                if (
                    status_code
                    not in retryable_status_codes
                ):
                    raise

                if attempt == self.max_retries:
                    raise RuntimeError(
                        "Cloud embedding request failed "
                        f"after {self.max_retries} attempts "
                        f"with HTTP {status_code}."
                    ) from exc

                delay = (
                    self.retry_backoff_seconds
                    * (2 ** (attempt - 1))
                )

                logger.warning(
                    f"Cloud embedding request returned "
                    f"retryable HTTP status "
                    f"attempt={attempt} "
                    f"status_code={status_code} "
                    f"retry_in_seconds={delay}"
                )

                time.sleep(delay)

            except (
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.ConnectError,
            ) as exc:

                last_error = exc

                if attempt == self.max_retries:
                    raise RuntimeError(
                        "Cloud embedding request failed "
                        f"after {self.max_retries} attempts."
                    ) from exc

                delay = (
                    self.retry_backoff_seconds
                    * (2 ** (attempt - 1))
                )

                logger.warning(
                    f"Cloud embedding request failed "
                    f"attempt={attempt} "
                    f"retry_in_seconds={delay} "
                    f"error={type(exc).__name__}"
                )

                time.sleep(delay)

        else:
            raise RuntimeError("Cloud embedding request failed.") from last_error

        body = response.json()
        data = body.get("data")
        if not data:
            raise RuntimeError("Cloud embedding API returned no data.")

        ordered = sorted(
            data,
            key=lambda item: item["index"],
        )

        embeddings = [item["embedding"] for item in ordered]

        for embedding in embeddings:
            if len(embedding) != self._embedding_dimension:
                raise RuntimeError("Unexpected cloud embedding dimension.")

        return embeddings
