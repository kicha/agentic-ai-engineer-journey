from __future__ import annotations

import os
from enum import StrEnum

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class DenseEmbeddingProvider(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class SparseEmbeddingProvider(StrEnum):
    LOCAL = "local"
    QDRANT_CLOUD = "qdrant_cloud"


class RerankerProvider(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class QdrantDeployment(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class QueryTransformationMode(StrEnum):
    ORIGINAL = "original"
    REWRITE = "rewrite"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"


class ChunkingConfig(BaseModel):
    """Structure-aware parent/child chunking only."""

    max_chunk_chars: int = Field(default=300, ge=1)
    child_chunk_chars: int = Field(default=120, ge=1)
    child_overlap_chars: int = Field(default=30, ge=0)

    ingestion_version: str = "benchmark_v2"
    parent_chunking_version: str = "parent_v2"
    child_chunking_version: str = "child_v2"


class EmbeddingConfig(BaseModel):
    """Dense/sparse embedding provider selection and model names."""

    dense_provider: DenseEmbeddingProvider = DenseEmbeddingProvider(
        os.getenv("DENSE_EMBEDDING_PROVIDER", DenseEmbeddingProvider.CLOUD.value)
    )
    sparse_provider: SparseEmbeddingProvider = SparseEmbeddingProvider(
        os.getenv(
            "SPARSE_EMBEDDING_PROVIDER",
            SparseEmbeddingProvider.QDRANT_CLOUD.value,
        )
    )
    local_dense_model: str = os.getenv(
        "LOCAL_DENSE_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    sparse_model: str = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")


class MongoConfig(BaseModel):
    """Authoritative ParentChunk persistence configuration."""

    uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    database: str = os.getenv("MONGODB_DATABASE", "agentic_rag")
    parent_collection: str = os.getenv(
        "MONGODB_PARENT_COLLECTION",
        "parents_benchmark_v2_cloud_refactor",
    )


class HybridSearchConfig(BaseModel):
    """Hybrid retrieval tuning plus deployment-specific collection names."""

    local_collection_name: str = os.getenv(
        "QDRANT_LOCAL_COLLECTION",
        "structure_aware_v4_benchmark_v2",
    )
    cloud_collection_name: str = os.getenv(
        "QDRANT_CLOUD_COLLECTION",
        "structure_aware_v4_benchmark_v2_cloud_refactor",
    )
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    dense_prefetch_k: int = Field(default=15, ge=1)
    sparse_prefetch_k: int = Field(default=15, ge=1)
    final_k: int = Field(default=10, ge=1)
    rrf_k: int = Field(default=2, ge=1)

    def collection_for(
        self,
        deployment: QdrantDeployment,
        override: str | None = None,
    ) -> str:
        if override:
            return override
        if deployment == QdrantDeployment.LOCAL:
            return self.local_collection_name
        return self.cloud_collection_name


class RerankingConfig(BaseModel):
    provider: RerankerProvider = RerankerProvider(
        os.getenv("RERANKER_PROVIDER", RerankerProvider.CLOUD.value)
    )
    model_name: str = os.getenv(
        "RERANKER_MODEL",
        "jina-reranker-v2-base-multilingual",
    )
    local_model_name: str = os.getenv(
        "LOCAL_RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L6-v2",
    )
    top_k: int = Field(default=5, ge=1)


class QdrantConfig(BaseModel):
    deployment: QdrantDeployment = QdrantDeployment(
        os.getenv("QDRANT_DEPLOYMENT", QdrantDeployment.CLOUD.value)
    )
    local_url: str = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
    cloud_url: str | None = os.getenv("QDRANT_URL")
    api_key: str | None = os.getenv("QDRANT_API_KEY")
    cloud_inference: bool = (
        os.getenv("QDRANT_CLOUD_INFERENCE", "true").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    timeout_seconds: int = int(os.getenv("QDRANT_TIMEOUT_SECONDS", "60"))


class CloudConfig(BaseModel):
    embedding_model: str = os.getenv("CLOUD_EMBEDDING_MODEL", "jina-embeddings-v4")
    embedding_dimension: int = int(os.getenv("CLOUD_EMBEDDING_DIMENSION", "512"))
    embedding_url: str = os.getenv(
        "CLOUD_EMBEDDING_URL",
        "https://api.jina.ai/v1/embeddings",
    )
    reranker_url: str = os.getenv(
        "CLOUD_RERANKER_URL",
        "https://api.jina.ai/v1/rerank",
    )
    api_key: str | None = os.getenv("CLOUD_EMBEDDING_API_KEY")
    timeout_seconds: int = int(os.getenv("CLOUD_TIMEOUT_SECONDS", "90"))
    max_retries: int = int(os.getenv("CLOUD_MAX_RETRIES", "3"))
    retry_backoff_seconds: float = float(os.getenv("CLOUD_RETRY_BACKOFF_SECONDS", "2"))


class QueryingConfig(BaseModel):
    mode: QueryTransformationMode = QueryTransformationMode(
        os.getenv("QUERY_TRANSFORMATION_MODE", QueryTransformationMode.MULTI_QUERY.value)
    )
    llm_model: str = os.getenv("QUERY_LLM_MODEL", "openai/gpt-oss-120b")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    generated_query_count: int = Field(
        default=int(os.getenv("MULTI_QUERY_COUNT", "3")),
        ge=1,
    )
    include_original_query: bool = True
    fail_open: bool = True
    multi_query_rrf_k: int = Field(
        default=int(os.getenv("MULTI_QUERY_RRF_K", "60")),
        ge=0,
    )
    child_top_k_per_query: int = Field(default=15, ge=1)
    fused_child_top_k: int = Field(default=30, ge=1)
    reranker_candidate_k: int = Field(default=5, ge=1)
    final_parent_k: int = Field(default=3, ge=1)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
