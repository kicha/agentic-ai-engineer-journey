import logging
from typing import List

from qdrant_client import QdrantClient, models
from app.embeddings.base import SparseVector as EmbeddingSparseVector, DenseVector
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)
from app.config.settings import (
    HybridSearchConfig,
    QdrantConfig,
    QdrantDeployment,
    SparseEmbeddingProvider,
)

from app.models.child_chunk import ChildChunk

logger = logging.getLogger(__name__)


class HybridQdrantStore:
    """
    Qdrant storage for hybrid retrieval.

    Every ChildChunk is represented by:

        one Qdrant point
            ├── dense vector
            ├── sparse BM25 vector
            └── search-oriented payload

    The domain model remains normalized.

    The Qdrant payload is intentionally denormalized where
    useful for filtering and diagnostics.
    """

    def __init__(
        self,
        config: HybridSearchConfig,
        vector_size: int,
        qdrant_config: QdrantConfig,
        sparse_embedding_provider: SparseEmbeddingProvider,
        sparse_model_name: str,
    ) -> None:

        self.config = config
        self.vector_size = vector_size
        self.qdrant_config = qdrant_config
        self.sparse_embedding_provider = sparse_embedding_provider
        self.sparse_model_name = sparse_model_name

        if qdrant_config.deployment == QdrantDeployment.LOCAL:
            self.client = QdrantClient(
                url=qdrant_config.local_url,
                timeout=qdrant_config.timeout_seconds,
            )

        elif qdrant_config.deployment == QdrantDeployment.CLOUD:

            if not qdrant_config.cloud_url:
                raise ValueError("QDRANT_URL is required " "for Qdrant Cloud.")

            if not qdrant_config.api_key:
                raise ValueError("QDRANT_API_KEY is required " "for Qdrant Cloud.")

            self.client = QdrantClient(
                url=qdrant_config.cloud_url,
                api_key=qdrant_config.api_key,
                cloud_inference=(qdrant_config.cloud_inference),
                timeout=qdrant_config.timeout_seconds,
            )

        else:
            raise ValueError(
                "Unsupported Qdrant deployment: " f"{qdrant_config.deployment}"
            )

    @property
    def uses_cloud_sparse_inference(self) -> bool:
        return self.sparse_embedding_provider == SparseEmbeddingProvider.QDRANT_CLOUD

    # ========================================================
    # COLLECTION MANAGEMENT
    # ========================================================

    def recreate_collection(
        self,
    ) -> None:

        collection_name = self.config.collection_name

        if self.client.collection_exists(collection_name):

            logger.info(
                "Deleting existing Qdrant collection=%s",
                collection_name,
            )

            self.client.delete_collection(collection_name=(collection_name))

        logger.info(
            "Creating Qdrant hybrid collection=%s",
            collection_name,
        )

        self.client.create_collection(
            collection_name=(collection_name),
            vectors_config={
                self.config.dense_vector_name: VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self.config.sparse_vector_name: SparseVectorParams(
                    modifier=(models.Modifier.IDF),
                ),
            },
        )

        self._create_payload_indexes()

    # ========================================================
    # PAYLOAD INDEXES
    # ========================================================

    def _create_payload_indexes(
        self,
    ) -> None:

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
                collection_name=(self.config.collection_name),
                field_name=field_name,
                field_schema=(models.PayloadSchemaType.KEYWORD),
            )

        logger.info(
            "Created Qdrant payload indexes collection=%s fields=%d",
            self.config.collection_name,
            len(keyword_fields),
        )

    # ========================================================
    # PAYLOAD PROJECTION
    # ========================================================

    @staticmethod
    def _build_payload(
        child: ChildChunk,
    ) -> dict:
        """
        Build the Qdrant search projection.

        Heading hierarchy is canonical in:

            child.heading_context

        We intentionally flatten it here because Qdrant
        filtering should not need to understand our domain
        object hierarchy.
        """

        heading = child.heading_context

        payload = child.model_dump(
            mode="json",
            exclude={
                "heading_context",
            },
        )

        payload.update(
            {
                "h1": heading.h1,
                "h2": heading.h2,
                "h3": heading.h3,
                "h4": heading.h4,
                "h5": heading.h5,
                "h6": heading.h6,
                "section_path": (heading.get_section_path()),
            }
        )

        return payload

    # ========================================================
    # INSERT CHILDREN LOCAL
    # ========================================================

    def insert_children(
        self,
        children: List[ChildChunk],
        dense_embeddings: List[DenseVector],
        sparse_embeddings: List[EmbeddingSparseVector],
    ) -> None:

        if not (len(children) == len(dense_embeddings) == len(sparse_embeddings)):

            raise ValueError(
                "children, dense_embeddings and "
                "sparse_embeddings must have "
                "equal length."
            )

        if not children:
            logger.info("No children supplied for Qdrant insertion.")
            return

        self._validate_unique_child_keys(children)

        points: List[PointStruct] = []

        for (
            child,
            dense_vector,
            sparse_vector,
        ) in zip(
            children,
            dense_embeddings,
            sparse_embeddings,
        ):

            payload = self._build_payload(child)

            qdrant_sparse_vector = self._to_qdrant_sparse_vector(sparse_vector)

            points.append(
                PointStruct(
                    id=str(child.child_id),
                    vector={
                        self.config.dense_vector_name: (dense_vector),
                        self.config.sparse_vector_name: (qdrant_sparse_vector),
                    },
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=(self.config.collection_name),
            points=points,
            wait=True,
        )

        logger.info(
            "Inserted %d hybrid child points into collection=%s",
            len(points),
            self.config.collection_name,
        )

    # ========================================================
    # INSERT CHILDREN CLOUD
    # ========================================================

    def insert_children_cloud_sparse(
        self,
        children: List[ChildChunk],
        dense_embeddings: List[DenseVector],
    ) -> None:
        """
        Insert ChildChunks using:

            dense vector:
                already generated by our configured
                DenseEmbeddingService

            sparse vector:
                generated server-side by Qdrant BM25
        """

        if not self.uses_cloud_sparse_inference:
            raise RuntimeError(
                "Cloud sparse insertion requires "
                "SparseEmbeddingProvider.QDRANT_CLOUD."
            )

        if len(children) != len(dense_embeddings):
            raise ValueError("children and dense_embeddings " "must have equal length.")

        if not children:
            logger.info("No children supplied for " "Qdrant insertion.")
            return

        self._validate_unique_child_keys(children)

        points: List[PointStruct] = []

        for child, dense_vector in zip(
            children,
            dense_embeddings,
        ):

            payload = self._build_payload(child)
            logger.info("Sparse txt in HybridQdrantStore...", child.content)
            points.append(
                PointStruct(
                    id=str(child.child_id),
                    vector={
                        self.config.dense_vector_name: (dense_vector),
                        self.config.sparse_vector_name: (
                            models.Document(
                                text=child.content,
                                model=(self.sparse_model_name),
                            )
                        ),
                    },
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=(self.config.collection_name),
            points=points,
            wait=True,
        )

        logger.info(
            f"Inserted {len(points)} "
            f"hybrid child points "
            f"with Qdrant Cloud BM25 "
            f"collection="
            f"{self.config.collection_name}"
        )

    # ========================================================
    # DELETE CHILDREN BY DOCUMENT
    # ========================================================

    def delete_by_document_id(
        self,
        document_id: str,
    ) -> None:

        self.client.delete(
            collection_name=(self.config.collection_name),
            points_selector=(
                models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=(models.MatchValue(value=(document_id))),
                            )
                        ]
                    )
                )
            ),
            wait=True,
        )

        logger.info(
            "Deleted Qdrant children for document_id=%s",
            document_id,
        )

    # ========================================================
    # DENSE SEARCH
    # ========================================================

    def dense_search(
        self,
        query_vector: List[float],
        limit: int = 10,
    ):

        response = self.client.query_points(
            collection_name=(self.config.collection_name),
            query=query_vector,
            using=(self.config.dense_vector_name),
            limit=limit,
            with_payload=True,
        )

        return response.points

    # ========================================================
    # SPARSE SEARCH
    # ========================================================

    def sparse_search(
        self,
        query_vector: EmbeddingSparseVector,
        limit: int = 10,
    ):
        qdrant_query_vector = self._to_qdrant_sparse_vector(query_vector)

        response = self.client.query_points(
            collection_name=(self.config.collection_name),
            query=qdrant_query_vector,
            using=(self.config.sparse_vector_name),
            limit=limit,
            with_payload=True,
        )

        return response.points

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def hybrid_search(
        self,
        dense_query_vector: List[float],
        sparse_query_vector: EmbeddingSparseVector,
        limit: int | None = None,
    ):

        final_limit = limit if limit is not None else self.config.final_k
        qdrant_sparse_query = self._to_qdrant_sparse_vector(sparse_query_vector)

        response = self.client.query_points(
            collection_name=(self.config.collection_name),
            prefetch=[
                models.Prefetch(
                    query=(dense_query_vector),
                    using=(self.config.dense_vector_name),
                    limit=(self.config.dense_prefetch_k),
                ),
                models.Prefetch(
                    query=(qdrant_sparse_query),
                    using=(self.config.sparse_vector_name),
                    limit=(self.config.sparse_prefetch_k),
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=self.config.rrf_k)),
            limit=final_limit,
            with_payload=True,
        )

        return response.points

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_unique_child_keys(children: List[ChildChunk]) -> None:

        child_keys = [child.child_key for child in children]
        if len(child_keys) != len(set(child_keys)):

            raise ValueError(
                "Duplicate child_key values " "detected before Qdrant insertion."
            )

    @staticmethod
    def _to_qdrant_sparse_vector(
        vector: EmbeddingSparseVector,
    ) -> models.SparseVector:
        """
        Convert the provider-neutral sparse embedding model
        into Qdrant's transport model.

        Embedding services must not depend on Qdrant types.
        Conversion belongs at the vector-store boundary.
        """

        return models.SparseVector(
            indices=vector.indices,
            values=vector.values,
        )

    def sparse_search_text(
        self,
        query_text: str,
        limit: int = 10,
    ):
        """
        Sparse BM25 search using Qdrant server-side inference.
        """

        if not self.uses_cloud_sparse_inference:
            raise RuntimeError(
                "Cloud sparse text search requires "
                "SparseEmbeddingProvider.QDRANT_CLOUD."
            )

        response = self.client.query_points(
            collection_name=(self.config.collection_name),
            query=models.Document(
                text=query_text,
                model=self.sparse_model_name,
            ),
            using=(self.config.sparse_vector_name),
            limit=limit,
            with_payload=True,
        )

        return response.points

    def hybrid_search_text(
        self,
        dense_query_vector: List[float],
        query_text: str,
        limit: int | None = None,
    ):
        """
        Hybrid search where:

            dense query vector
                is supplied by our dense provider

            sparse BM25 vector
                is generated inside Qdrant Cloud
        """

        if not self.uses_cloud_sparse_inference:
            raise RuntimeError(
                "Cloud hybrid text search requires "
                "SparseEmbeddingProvider.QDRANT_CLOUD."
            )

        final_limit = limit if limit is not None else self.config.final_k

        response = self.client.query_points(
            collection_name=(self.config.collection_name),
            prefetch=[
                models.Prefetch(
                    query=dense_query_vector,
                    using=(self.config.dense_vector_name),
                    limit=(self.config.dense_prefetch_k),
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query_text,
                        model=(self.sparse_model_name),
                    ),
                    using=(self.config.sparse_vector_name),
                    limit=(self.config.sparse_prefetch_k),
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=self.config.rrf_k)),
            limit=final_limit,
            with_payload=True,
        )

        return response.points
