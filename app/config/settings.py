from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from enum import StrEnum

load_dotenv()


class DenseEmbeddingProvider(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class SparseEmbeddingProvider(StrEnum):
    LOCAL = "local"
    QDRANT_CLOUD = "qdrant_cloud"


class CloudEmbeddingProvider(StrEnum):
    JINA = "jina"
    VOYAGE = "voyage"
    COHERE = "cohere"


class RerankerProvider(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class QdrantDeployment(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


# ============================================================
# CONFIGURATION
# ============================================================


class ChunkingConfig(BaseModel):

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    max_chunk_chars: int = 300
    min_chunk_chars: int = 100

    # --------------------------------------------------------
    # CHILD CHUNKING
    # --------------------------------------------------------

    child_chunk_chars: int = 120
    child_overlap_chars: int = 30

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_embedding_model: str = "Qdrant/bm25"
    dense_embedding_provider: DenseEmbeddingProvider = DenseEmbeddingProvider.CLOUD
    sparse_embedding_provider: SparseEmbeddingProvider = (
        SparseEmbeddingProvider.QDRANT_CLOUD
    )

    # --------------------------------------------------------
    # QDRANT - LEGACY DENSE COLLECTION
    # --------------------------------------------------------

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    collection_name: str = "structure_aware_v4_benchmark_v2"

    top_k: int = 3

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    child_top_k: int = 5
    child_score_threshold: float = 0.40
    parent_top_k: int = 2

    # --------------------------------------------------------
    # MONGODB
    # --------------------------------------------------------

    MONGODB_URI: str = os.getenv(
        "MONGODB_URI",
        "mongodb://localhost:27017",
    )

    mongodb_uri: str = MONGODB_URI
    mongodb_database: str = "agentic_rag"

    # IMPORTANT:
    # V2 benchmark gets its own collection.
    mongodb_parent_collection: str = "parents_benchmark_v2"

    # --------------------------------------------------------
    # VERSIONING
    # --------------------------------------------------------

    ingestion_version: str = "benchmark_v2"
    parent_chunking_version: str = "parent_v2"
    child_chunking_version: str = "child_v2"


class HybridSearchConfig(BaseModel):

    # --------------------------------------------------------
    # BENCHMARK V2 HYBRID COLLECTION
    # --------------------------------------------------------

    collection_name: str = "structure_aware_v4_benchmark_v2"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    dense_prefetch_k: int = Field(
        default=15,
        ge=1,
    )
    sparse_prefetch_k: int = Field(
        default=15,
        ge=1,
    )
    final_k: int = Field(
        default=10,
        ge=1,
    )

    rrf_k: int = Field(
        default=2,
        ge=1,
    )


class RerankingConfig(BaseModel):

    provider: RerankerProvider = RerankerProvider.LOCAL
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    top_k: int = Field(
        default=5,
        ge=1,
    )


class QdrantConfig(BaseModel):

    deployment: QdrantDeployment = QdrantDeployment.LOCAL
    local_url: str = "http://localhost:6333"
    cloud_url: str | None = os.getenv("QDRANT_URL")
    api_key: str | None = os.getenv("QDRANT_API_KEY")
    cloud_inference: bool = True
    timeout_seconds: int = 60


class CloudConfig(BaseModel):

    embedding_provider: CloudEmbeddingProvider = CloudEmbeddingProvider.JINA
    embedding_model: str = "jina-embeddings-v4"
    embedding_dimension: int = 512
    cloud_url: str = "https://api.jina.ai/v1/embeddings"
    reranker_url: str = "https://api.jina.ai/v1/rerank"
    api_key: str | None = os.getenv("CLOUD_EMBEDDING_API_KEY")
    timeout_seconds: int = 90
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
