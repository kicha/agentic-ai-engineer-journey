from __future__ import annotations

import logging
from typing import Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from app.config.settings import HybridSearchConfig, QdrantConfig, QdrantDeployment
from app.embeddings.base import DenseVector, SparseEmbedding, SparseText, SparseVector
from app.models.child_chunk import ChildChunk

logger = logging.getLogger(__name__)


class HybridQdrantStore:
    """Qdrant persistence and search boundary for hybrid retrieval.

    The public API is intentionally provider-neutral. Dense and sparse
    embeddings may be produced locally or by hosted providers; callers still
    use the same insertion/search methods.
    """

    def __init__(
        self,
        config: HybridSearchConfig,
        vector_size: int,
        qdrant_config: QdrantConfig,
        collection_name: str,
    ) -> None:
        self.config = config
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.qdrant_config = qdrant_config
        self.client = self._create_client(qdrant_config)

    @staticmethod
    def _create_client(qdrant_config: QdrantConfig) -> QdrantClient:
        if qdrant_config.deployment == QdrantDeployment.LOCAL:
            return QdrantClient(
                url=qdrant_config.local_url,
                timeout=qdrant_config.timeout_seconds,
            )

        if qdrant_config.deployment == QdrantDeployment.CLOUD:
            if not qdrant_config.cloud_url:
                raise ValueError("QDRANT_URL is required for Qdrant Cloud.")
            if not qdrant_config.api_key:
                raise ValueError("QDRANT_API_KEY is required for Qdrant Cloud.")

            return QdrantClient(
                url=qdrant_config.cloud_url,
                api_key=qdrant_config.api_key,
                cloud_inference=qdrant_config.cloud_inference,
                timeout=qdrant_config.timeout_seconds,
            )

        raise ValueError(
            f"Unsupported Qdrant deployment: {qdrant_config.deployment}"
        )

    # ========================================================
    # COLLECTION MANAGEMENT
    # ========================================================

    def recreate_collection(self) -> None:
        collection_name = self.collection_name

        if self.client.collection_exists(collection_name):
            logger.info(f"Deleting existing Qdrant collection={collection_name}")
            self.client.delete_collection(collection_name=collection_name)

        logger.info(f"Creating Qdrant hybrid collection={collection_name}")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                self.config.dense_vector_name: VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.config.sparse_vector_name: SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        keyword_fields = [
            "document_id",
            "parent_id",
            "parent_key",
            "child_id",
            "child_key",
            "source",
            "content_type",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "section_path",
        ]

        for field_name in keyword_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        logger.info(
            f"Created Qdrant payload indexes collection="
            f"{self.collection_name} fields={len(keyword_fields)}"
        )

    # ========================================================
    # PAYLOAD PROJECTION
    # ========================================================

    @staticmethod
    def _build_payload(child: ChildChunk) -> dict:
        heading = child.heading_context
        payload = child.model_dump(mode="json", exclude={"heading_context"})
        payload.update(
            {
                "h1": heading.h1,
                "h2": heading.h2,
                "h3": heading.h3,
                "h4": heading.h4,
                "h5": heading.h5,
                "h6": heading.h6,
                "section_path": heading.get_section_path(),
            }
        )
        return payload

    # ========================================================
    # INSERTION
    # ========================================================

    def insert_children(
        self,
        children: Sequence[ChildChunk],
        dense_embeddings: Sequence[DenseVector],
        sparse_embeddings: Sequence[SparseEmbedding],
    ) -> None:
        if not (
            len(children) == len(dense_embeddings) == len(sparse_embeddings)
        ):
            raise ValueError(
                "children, dense_embeddings and sparse_embeddings must have equal length."
            )

        if not children:
            logger.info("No children supplied for Qdrant insertion.")
            return

        self._validate_unique_child_keys(children)

        points: list[PointStruct] = []
        for child, dense_vector, sparse_embedding in zip(
            children,
            dense_embeddings,
            sparse_embeddings,
        ):
            points.append(
                PointStruct(
                    id=str(child.child_id),
                    vector={
                        self.config.dense_vector_name: dense_vector,
                        self.config.sparse_vector_name: self._to_qdrant_sparse(
                            sparse_embedding
                        ),
                    },
                    payload=self._build_payload(child),
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info(
            f"Inserted {len(points)} hybrid child points into "
            f"collection={self.collection_name}"
        )

    def delete_by_document_id(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        logger.info(f"Deleted Qdrant children for document_id={document_id}")

    # ========================================================
    # SEARCH
    # ========================================================

    def dense_search(
        self,
        query_vector: DenseVector,
        limit: int = 10,
    ):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=self.config.dense_vector_name,
            limit=limit,
            with_payload=True,
        )
        return response.points

    def sparse_search(
        self,
        query_embedding: SparseEmbedding,
        limit: int = 10,
    ):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=self._to_qdrant_sparse(query_embedding),
            using=self.config.sparse_vector_name,
            limit=limit,
            with_payload=True,
        )
        return response.points

    def hybrid_search(
        self,
        dense_query_vector: DenseVector,
        sparse_query_embedding: SparseEmbedding,
        limit: int | None = None,
    ):
        final_limit = limit if limit is not None else self.config.final_k
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query_vector,
                    using=self.config.dense_vector_name,
                    limit=self.config.dense_prefetch_k,
                ),
                models.Prefetch(
                    query=self._to_qdrant_sparse(sparse_query_embedding),
                    using=self.config.sparse_vector_name,
                    limit=self.config.sparse_prefetch_k,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=self.config.rrf_k)),
            limit=final_limit,
            with_payload=True,
        )
        return response.points

    # ========================================================
    # VALIDATION / TRANSPORT CONVERSION
    # ========================================================

    @staticmethod
    def _validate_unique_child_keys(children: Sequence[ChildChunk]) -> None:
        child_keys = [child.child_key for child in children]
        if len(child_keys) != len(set(child_keys)):
            raise ValueError("Duplicate child_key values detected before Qdrant insertion.")

    @staticmethod
    def _to_qdrant_sparse(
        embedding: SparseEmbedding,
    ) -> models.SparseVector | models.Document:
        if isinstance(embedding, SparseVector):
            return models.SparseVector(
                indices=embedding.indices,
                values=embedding.values,
            )

        if isinstance(embedding, SparseText):
            return models.Document(
                text=embedding.text,
                model=embedding.model,
            )

        raise TypeError(
            f"Unsupported sparse embedding type: {type(embedding).__name__}"
        )
