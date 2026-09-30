from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.config.settings import (
    ChunkingConfig,
    CloudConfig,
    DenseEmbeddingProvider,
    EmbeddingConfig,
    HybridSearchConfig,
    MongoConfig,
    QdrantConfig,
    QdrantDeployment,
    QueryingConfig,
    RerankerProvider,
    RerankingConfig,
    SparseEmbeddingProvider,
)
from app.embeddings.base import DenseEmbeddingService, SparseEmbeddingService
from app.embeddings.factory import (
    create_dense_embedding_service,
    create_sparse_embedding_service,
)
from app.querying.factory import create_query_llm
from app.querying.multi_query_generator import MultiQueryGenerator
from app.querying.query_rewriter import QueryRewriter
from app.repositories.parent_repository import ParentRepository
from app.reranking.base import Reranker
from app.reranking.factory import create_reranker
from app.retrieval.hybrid_child_retriever import HybridChildRetriever
from app.retrieval.hybrid_parent_child_retriever import HybridParentChildRetriever
from app.retrieval.hybrid_rerank_mmr_retriever import HybridRerankMMRRetriever
from app.retrieval.hyde.hyde_generator import HyDEGenerator
from app.retrieval.hyde.hyde_hybrid_retriever import HyDEHybridRetriever
from app.retrieval.multi_query_hybrid_retriever import MultiQueryHybridRetriever
from app.retrieval.parent_mmr_service import ParentMMRService
from app.retrieval.query_orchestrator import QueryOrchestrator
from app.vectorstore.hybrid_qdrant_store import HybridQdrantStore


class RuntimeConfigs(BaseModel):
    """Resolved runtime configuration for one local/cloud execution profile."""

    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    mongo: MongoConfig
    hybrid: HybridSearchConfig
    qdrant: QdrantConfig
    reranking: RerankingConfig
    cloud: CloudConfig
    querying: QueryingConfig
    collection_name: str


class RetrievalStack(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: RuntimeConfigs
    dense_embedder: DenseEmbeddingService
    sparse_embedder: SparseEmbeddingService
    vector_store: HybridQdrantStore
    parent_repository: ParentRepository
    hybrid_child_retriever: HybridChildRetriever
    parent_child_retriever: HybridParentChildRetriever
    reranker: Reranker
    parent_mmr_service: ParentMMRService
    base_retriever: HybridRerankMMRRetriever
    query_rewriter: QueryRewriter
    multi_query_generator: MultiQueryGenerator
    multi_query_retriever: MultiQueryHybridRetriever
    hyde_retriever: HyDEHybridRetriever
    orchestrator: QueryOrchestrator

    def close(self) -> None:
        self.parent_repository.close()


def build_runtime_configs(
    *,
    deployment: QdrantDeployment,
    collection_name: str | None = None,
    querying_config: QueryingConfig | None = None,
) -> RuntimeConfigs:
    chunking = ChunkingConfig()
    hybrid = HybridSearchConfig()
    mongo = MongoConfig()
    cloud = CloudConfig()
    querying = querying_config or QueryingConfig()

    if deployment == QdrantDeployment.LOCAL:
        embedding = EmbeddingConfig(
            dense_provider=DenseEmbeddingProvider.LOCAL,
            sparse_provider=SparseEmbeddingProvider.LOCAL,
        )
        reranking = RerankingConfig(provider=RerankerProvider.LOCAL)
        reranking = reranking.model_copy(
            update={"model_name": reranking.local_model_name}
        )
    else:
        embedding = EmbeddingConfig(
            dense_provider=DenseEmbeddingProvider.CLOUD,
            sparse_provider=SparseEmbeddingProvider.QDRANT_CLOUD,
        )
        reranking = RerankingConfig(provider=RerankerProvider.CLOUD)

    qdrant = QdrantConfig(deployment=deployment)
    active_collection = hybrid.collection_for(
        deployment=deployment,
        override=collection_name,
    )

    return RuntimeConfigs(
        chunking=chunking,
        embedding=embedding,
        mongo=mongo,
        hybrid=hybrid,
        qdrant=qdrant,
        reranking=reranking,
        cloud=cloud,
        querying=querying,
        collection_name=active_collection,
    )


def build_retrieval_stack(
    *,
    deployment: QdrantDeployment,
    collection_name: str | None = None,
    querying_config: QueryingConfig | None = None,
) -> RetrievalStack:
    runtime = build_runtime_configs(
        deployment=deployment,
        collection_name=collection_name,
        querying_config=querying_config,
    )

    dense_embedder = create_dense_embedding_service(
        config=runtime.embedding,
        cloud_config=runtime.cloud,
    )
    sparse_embedder = create_sparse_embedding_service(config=runtime.embedding)
    vector_store = HybridQdrantStore(
        config=runtime.hybrid,
        vector_size=dense_embedder.embedding_dimension,
        qdrant_config=runtime.qdrant,
        collection_name=runtime.collection_name,
    )
    parent_repository = ParentRepository(config=runtime.mongo)
    child_retriever = HybridChildRetriever(
        dense_embedding_service=dense_embedder,
        sparse_embedding_service=sparse_embedder,
        vector_store=vector_store,
    )
    parent_child_retriever = HybridParentChildRetriever(
        hybrid_child_retriever=child_retriever,
        parent_repository=parent_repository,
    )
    reranker = create_reranker(
        config=runtime.reranking,
        cloud_config=runtime.cloud,
    )
    parent_mmr_service = ParentMMRService(
        embedder=dense_embedder,
        lambda_mult=runtime.querying.mmr_lambda,
    )
    base_retriever = HybridRerankMMRRetriever(
        hybrid_parent_child_retriever=parent_child_retriever,
        reranker=reranker,
        parent_mmr_service=parent_mmr_service,
    )

    query_llm = create_query_llm(runtime.querying)
    query_rewriter = QueryRewriter(
        llm=query_llm,
        fail_open=runtime.querying.fail_open,
    )
    multi_query_generator = MultiQueryGenerator(
        llm=query_llm,
        generated_query_count=runtime.querying.generated_query_count,
        include_original=runtime.querying.include_original_query,
        fail_open=runtime.querying.fail_open,
    )
    multi_query_retriever = MultiQueryHybridRetriever(
        query_generator=multi_query_generator,
        hybrid_child_retriever=child_retriever,
        parent_repository=parent_repository,
        reranker=reranker,
        embedder=dense_embedder,
        lambda_mult=runtime.querying.mmr_lambda,
        multi_query_rrf_k=runtime.querying.multi_query_rrf_k,
        parent_mmr_service=parent_mmr_service,
    )
    hyde_retriever = HyDEHybridRetriever(
        hyde_generator=HyDEGenerator(query_llm),
        hybrid_child_retriever=child_retriever,
        parent_child_retriever=parent_child_retriever,
        reranker=reranker,
        parent_mmr_service=parent_mmr_service,
    )
    orchestrator = QueryOrchestrator(
        config=runtime.querying,
        query_rewriter=query_rewriter,
        multi_query_generator=multi_query_generator,
        base_retriever=base_retriever,
        multi_query_retriever=multi_query_retriever,
        hyde_retriever=hyde_retriever,
    )

    return RetrievalStack(
        runtime=runtime,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        vector_store=vector_store,
        parent_repository=parent_repository,
        hybrid_child_retriever=child_retriever,
        parent_child_retriever=parent_child_retriever,
        reranker=reranker,
        parent_mmr_service=parent_mmr_service,
        base_retriever=base_retriever,
        query_rewriter=query_rewriter,
        multi_query_generator=multi_query_generator,
        multi_query_retriever=multi_query_retriever,
        hyde_retriever=hyde_retriever,
        orchestrator=orchestrator,
    )
