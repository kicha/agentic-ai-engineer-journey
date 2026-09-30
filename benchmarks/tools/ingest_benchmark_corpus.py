import logging
from time import perf_counter
from app.application_factory import build_runtime_configs
from app.config.settings import QdrantDeployment
from app.ingestion.markdown_corpus_loader import (
    MarkdownCorpusLoader,
)
from app.ingestion.pipeline import (
    StructureAwareRAGDemo,
)
from benchmarks.paths import (
    BENCHMARK_DOCUMENTS_DIR,
)
from app.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

DOCUMENTS_DIRECTORY = BENCHMARK_DOCUMENTS_DIR


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    configure_logging()
    logger.info("Starting Benchmark V2 corpus ingestion.")
    runtime = build_runtime_configs(deployment=QdrantDeployment.CLOUD)

    logger.info(f"documents_directory=" f"{DOCUMENTS_DIRECTORY}")
    logger.info(f"mongodb_database=" f"{runtime.mongo.database}")
    logger.info(f"mongodb_parent_collection=" f"{runtime.mongo.parent_collection}")
    logger.info(f"qdrant_collection=" f"{runtime.collection_name}")

    logger.info(
        f"dense_provider={runtime.embedding.dense_provider.value} "
        f"qdrant_deployment={runtime.qdrant.deployment.value} "
        f"qdrant_cloud_url_configured="
        f"{bool(runtime.qdrant.cloud_url)} "
        f"qdrant_api_key_configured="
        f"{bool(runtime.qdrant.api_key)} "
        f"collection={runtime.collection_name}"
    )

    # ========================================================
    # LOAD DOCUMENTS
    # ========================================================

    loader = MarkdownCorpusLoader(documents_directory=DOCUMENTS_DIRECTORY)
    documents = loader.load()
    logger.info(f"documents_discovered=" f"{len(documents)}")
    for document in documents:
        logger.debug(
            f"document_id=" f"{document.document_id} " f"source={document.source}"
        )

    # ========================================================
    # APPLICATION
    # ========================================================

    pipeline = StructureAwareRAGDemo(
        chunking_config=runtime.chunking,
        embedding_config=runtime.embedding,
        mongo_config=runtime.mongo,
        hybrid_config=runtime.hybrid,
        qdrant_config=runtime.qdrant,
        collection_name=runtime.collection_name,
        cloud_config=runtime.cloud,
    )

    total_parents = 0
    total_children = 0
    succeeded = 0
    failed = 0

    started = perf_counter()

    try:

        for index, document in enumerate(
            documents,
            start=1,
        ):
            document_started = perf_counter()
            logger.info(
                f"Processing document "
                f"{index}/{len(documents)} "
                f"document_id="
                f"{document.document_id} "
                f"source="
                f"{document.source}"
            )

            try:

                result = pipeline.ingest(
                    document=document,
                    recreate_hybrid_collection=(index == 1),
                )

                elapsed = perf_counter() - document_started
                parent_count = len(result.parents)
                child_count = len(result.children)
                total_parents += parent_count
                total_children += child_count
                succeeded += 1
                logger.info(
                    f"Completed document "
                    f"document_id="
                    f"{document.document_id} "
                    f"parents="
                    f"{parent_count} "
                    f"children="
                    f"{child_count} "
                    f"elapsed_seconds="
                    f"{elapsed:.2f}"
                )
                logger.info(
                    f"dense_provider={runtime.embedding.dense_provider.value} "
                    f"dense_model={cloud_config.embedding_model} "
                    f"dense_dimension={cloud_config.embedding_dimension} "
                    f"qdrant_deployment={runtime.qdrant.deployment.value} "
                    f"collection={runtime.collection_name}"
                )

            except Exception:
                failed += 1
                logger.exception(
                    f"Failed document " f"document_id=" f"{document.document_id}"
                )

                raise

    finally:

        pipeline.close()

    total_elapsed = perf_counter() - started

    logger.info(
        f"Corpus ingestion complete "
        f"documents_discovered="
        f"{len(documents)} "
        f"documents_succeeded="
        f"{succeeded} "
        f"documents_failed="
        f"{failed} "
        f"total_parents="
        f"{total_parents} "
        f"total_children="
        f"{total_children} "
        f"elapsed_seconds="
        f"{total_elapsed:.2f} "
        f"mongodb_collection="
        f"{runtime.mongo.parent_collection} "
        f"qdrant_collection="
        f"{runtime.collection_name}"
    )


if __name__ == "__main__":

    main()
