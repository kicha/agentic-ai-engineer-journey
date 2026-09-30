from app.config.settings import (
    ChunkingConfig,
    EmbeddingConfig,
    HybridSearchConfig,
    MongoConfig,
    QdrantDeployment,
)


def test_chunking_config_has_only_chunking_responsibilities() -> None:
    config = ChunkingConfig()
    forbidden = {
        "embedding_model_name",
        "sparse_embedding_model",
        "dense_embedding_provider",
        "sparse_embedding_provider",
        "qdrant_host",
        "qdrant_port",
        "collection_name",
        "top_k",
        "child_top_k",
        "child_score_threshold",
        "parent_top_k",
        "mongodb_uri",
        "mongodb_database",
        "mongodb_parent_collection",
        "min_chunk_chars",
    }
    assert forbidden.isdisjoint(ChunkingConfig.model_fields)


def test_embedding_config_owns_embedding_provider_and_models() -> None:
    config = EmbeddingConfig()
    assert "dense_provider" in EmbeddingConfig.model_fields
    assert "sparse_provider" in EmbeddingConfig.model_fields
    assert "local_dense_model" in EmbeddingConfig.model_fields
    assert "sparse_model" in EmbeddingConfig.model_fields


def test_mongo_config_owns_parent_store_configuration() -> None:
    config = MongoConfig()
    assert "uri" in MongoConfig.model_fields
    assert "database" in MongoConfig.model_fields
    assert "parent_collection" in MongoConfig.model_fields


def test_hybrid_config_resolves_collection_from_deployment() -> None:
    config = HybridSearchConfig(
        local_collection_name="local-test",
        cloud_collection_name="cloud-test",
    )
    assert config.collection_for(QdrantDeployment.LOCAL) == "local-test"
    assert config.collection_for(QdrantDeployment.CLOUD) == "cloud-test"
    assert (
        config.collection_for(QdrantDeployment.CLOUD, override="experiment")
        == "experiment"
    )


def test_hybrid_config_has_no_ambiguous_generic_collection_field() -> None:
    config = HybridSearchConfig()
    assert "collection_name" not in HybridSearchConfig.model_fields
