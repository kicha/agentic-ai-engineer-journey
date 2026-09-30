import logging

from pydantic import BaseModel

from app.chunking.child_chunker import ChildChunker
from app.chunking.structure_aware_parent import (
    StructureAwareParentChunker,
)
from app.chunking.structure_detector import StructureDetector
from app.config.settings import (
    ChunkingConfig,
    CloudConfig,
    HybridSearchConfig,
    QdrantConfig,
    SparseEmbeddingProvider,
)

from app.embeddings.factory import (
    create_dense_embedding_service,
    create_sparse_embedding_service,
)

from app.models.child_chunk import ChildChunk
from app.models.document import Document
from app.models.parent_chunk import ParentChunk
from app.repositories.parent_repository import (
    ParentRepository,
)
from app.vectorstore.hybrid_qdrant_store import (
    HybridQdrantStore,
)

logger = logging.getLogger(__name__)


# ============================================================
# INGESTION RESULT
# ============================================================


class IngestionResult(BaseModel):
    """
    Result returned by the ingestion pipeline.
    """

    parents: list[ParentChunk]
    children: list[ChildChunk]


# ============================================================
# INGESTION PIPELINE
# ============================================================


class StructureAwareRAGDemo:
    """
    Structure-aware Parent/Child ingestion pipeline.

    Flow
    ----
    Document
        -> structure detection
        -> ParentChunk generation
        -> MongoDB parent synchronization
        -> persisted ParentChunk reload
        -> ChildChunk generation
        -> dense embeddings
        -> sparse embeddings
        -> hybrid Qdrant persistence
    """

    def __init__(
        self,
        config: ChunkingConfig,
        hybrid_config: HybridSearchConfig,
        qdrant_config: QdrantConfig,
        cloud_config: CloudConfig | None = None,
    ) -> None:

        self.config = config
        self.hybrid_config = hybrid_config
        self.detector = StructureDetector()
        self.parent_chunker = StructureAwareParentChunker(config)
        self.child_chunker = ChildChunker(config)
        self.parent_repository = ParentRepository(config)
        self.dense_embedder = create_dense_embedding_service(
            config=config,
            cloud_config=cloud_config,
        )

        self.sparse_embedder = None

        if config.sparse_embedding_provider == SparseEmbeddingProvider.LOCAL:
            self.sparse_embedder = create_sparse_embedding_service(
                config=config,
            )
        self.sparse_embedder = create_sparse_embedding_service(config=config)

        vector_size = self.dense_embedder.embedding_dimension

        self.hybrid_vector_store = HybridQdrantStore(
            config=hybrid_config,
            vector_size=vector_size,
            qdrant_config=qdrant_config,
            sparse_embedding_provider=(config.sparse_embedding_provider),
            sparse_model_name=(config.sparse_embedding_model),
        )

        logger.info(
            f"Ingestion pipeline initialized "
            f"mongodb_collection="
            f"{config.mongodb_parent_collection} "
            f"qdrant_collection="
            f"{hybrid_config.collection_name}"
        )

    # ========================================================
    # INGEST
    # ========================================================

    def ingest(
        self,
        document: Document,
        recreate_hybrid_collection: bool = False,
    ) -> IngestionResult:

        logger.info(
            f"Starting ingestion "
            f"document_id={document.document_id} "
            f"source={document.source}"
        )

        # ====================================================
        # 1. STRUCTURE DETECTION
        # ====================================================

        elements = self.detector.detect(document)

        logger.info(
            f"document_id={document.document_id} "
            f"structural_elements={len(elements)}"
        )

        # ====================================================
        # 2. GENERATE FRESH PARENTS
        # ====================================================

        generated_parents = self.parent_chunker.chunk(
            document=document,
            elements=elements,
        )

        logger.info(
            f"document_id={document.document_id} "
            f"generated_parents="
            f"{len(generated_parents)}"
        )

        # ====================================================
        # 3. SYNCHRONIZE PARENTS IN MONGODB
        # ====================================================

        self.parent_repository.ping()

        sync_result = self.parent_repository.sync_parents(
            document_id=document.document_id,
            parents=generated_parents,
        )

        logger.info(
            f"document_id={document.document_id} "
            f"incoming_parents="
            f"{sync_result.incoming_count} "
            f"existing_before_sync="
            f"{sync_result.existing_count_before} "
            f"inserted_parents="
            f"{sync_result.inserted_count} "
            f"unchanged_parents="
            f"{sync_result.unchanged_count} "
            f"deleted_stale_parents="
            f"{sync_result.deleted_stale_count} "
            f"final_parent_count="
            f"{sync_result.final_count}"
        )

        # ====================================================
        # 4. READ AUTHORITATIVE PARENTS BACK
        # ====================================================

        persisted_parents = self.parent_repository.get_by_document_id(
            document.document_id
        )

        logger.info(
            f"document_id={document.document_id} "
            f"persisted_parents="
            f"{len(persisted_parents)}"
        )

        self._validate_parent_sync(
            generated_parents=generated_parents,
            persisted_parents=persisted_parents,
        )

        # ====================================================
        # 5. CHILD CHUNKING
        # ====================================================

        children = self.child_chunker.chunk_parents(persisted_parents)

        logger.info(
            f"document_id={document.document_id} "
            f"generated_children="
            f"{len(children)}"
        )

        # ====================================================
        # 6. REFERENTIAL INTEGRITY
        # ====================================================

        self._validate_child_parent_references(
            parents=persisted_parents,
            children=children,
        )

        # ====================================================
        # 7. CHILD TEXT
        # ====================================================

        child_texts = [child.content for child in children]

        # ====================================================
        # 8. DENSE EMBEDDINGS
        # ====================================================

        dense_embeddings = self.dense_embedder.embed_documents(child_texts)

        logger.info(
            f"document_id={document.document_id} "
            f"dense_embeddings="
            f"{len(dense_embeddings)}"
        )

        if dense_embeddings:
            logger.debug(
                f"document_id={document.document_id} "
                f"dense_dimension="
                f"{len(dense_embeddings[0])}"
            )

        # ====================================================
        # 9. SPARSE EMBEDDINGS
        # ====================================================

        if self.config.sparse_embedding_provider == SparseEmbeddingProvider.LOCAL:

            if self.sparse_embedder is None:
                raise RuntimeError(
                    "Local sparse provider configured "
                    "without SparseEmbeddingService."
                )

            sparse_embeddings = self.sparse_embedder.embed_documents(child_texts)

            logger.info(
                f"document_id="
                f"{document.document_id} "
                f"sparse_embeddings="
                f"{len(sparse_embeddings)}"
            )

        else:

            sparse_embeddings = None

            logger.info(
                f"document_id="
                f"{document.document_id} "
                f"sparse_embedding_provider="
                f"qdrant_cloud "
                f"model="
                f"{self.config.sparse_embedding_model}"
            )

        logger.info(
            f"document_id={document.document_id} "
            f"sparse_embeddings="
            f"{len(sparse_embeddings)}"
        )

        # ====================================================
        # 10. QDRANT COLLECTION
        # ====================================================

        if recreate_hybrid_collection:

            logger.info(
                f"Recreating Qdrant collection " f"{self.hybrid_config.collection_name}"
            )

            self.hybrid_vector_store.recreate_collection()

        # ====================================================
        # 11. INSERT CHILDREN
        # ====================================================
        if (
            self.config.sparse_embedding_provider
            == SparseEmbeddingProvider.QDRANT_CLOUD
        ):

            self.hybrid_vector_store.insert_children_cloud_sparse(
                children=children,
                dense_embeddings=dense_embeddings,
            )

        else:

            if sparse_embeddings is None:
                raise RuntimeError(
                    "Sparse embeddings are required " "for local sparse insertion."
                )

            self.hybrid_vector_store.insert_children(
                children=children,
                dense_embeddings=dense_embeddings,
                sparse_embeddings=(sparse_embeddings),
            )

        logger.info(
            f"document_id={document.document_id} "
            f"qdrant_children_written="
            f"{len(children)}"
        )

        logger.info(f"Completed ingestion " f"document_id={document.document_id}")

        return IngestionResult(
            parents=persisted_parents,
            children=children,
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_parent_sync(
        generated_parents: list[ParentChunk],
        persisted_parents: list[ParentChunk],
    ) -> None:

        generated_keys = {parent.parent_key for parent in generated_parents}
        persisted_keys = {parent.parent_key for parent in persisted_parents}

        if generated_keys != persisted_keys:
            missing_keys = generated_keys - persisted_keys

            unexpected_keys = persisted_keys - generated_keys

            raise RuntimeError(
                f"Parent synchronization mismatch. "
                f"missing_parent_keys="
                f"{sorted(missing_keys)} "
                f"unexpected_parent_keys="
                f"{sorted(unexpected_keys)}"
            )

    @staticmethod
    def _validate_child_parent_references(
        parents: list[ParentChunk],
        children: list[ChildChunk],
    ) -> None:

        parent_ids = {parent.parent_id for parent in parents}

        parent_keys = {parent.parent_key for parent in parents}

        invalid_parent_ids = {
            child.parent_id for child in children if child.parent_id not in parent_ids
        }

        invalid_parent_keys = {
            child.parent_key
            for child in children
            if child.parent_key not in parent_keys
        }

        if invalid_parent_ids:
            raise RuntimeError(
                f"Invalid child parent_id values: " f"{sorted(invalid_parent_ids)}"
            )

        if invalid_parent_keys:
            raise RuntimeError(
                f"Invalid child parent_key values: " f"{sorted(invalid_parent_keys)}"
            )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:

        self.parent_repository.close()
        logger.info("Ingestion pipeline closed.")
