from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.application_factory import build_retrieval_stack, build_runtime_configs
from app.config.settings import (
    QdrantDeployment,
    QueryTransformationMode,
    QueryingConfig,
)
from app.ingestion.markdown_corpus_loader import MarkdownCorpusLoader
from app.ingestion.pipeline import StructureAwareRAGDemo
from app.utils.logging_config import configure_logging


def _deployment_from_args(args: argparse.Namespace) -> QdrantDeployment:
    return QdrantDeployment.LOCAL if args.local else QdrantDeployment.CLOUD


def _add_deployment_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true", help="Use local providers/Qdrant.")
    group.add_argument("--cloud", action="store_true", help="Use hosted providers/Qdrant.")
    parser.add_argument("--collection", default=None, help="Override Qdrant collection name.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest Markdown documents.")
    _add_deployment_flags(ingest_parser)
    ingest_parser.add_argument("--documents", type=Path, required=True)
    ingest_parser.add_argument("--recreate", action="store_true")

    search_parser = subparsers.add_parser("search", help="Run the advanced retrieval funnel.")
    _add_deployment_flags(search_parser)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--context", default=None)
    search_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in QueryTransformationMode],
        default=QueryTransformationMode.MULTI_QUERY.value,
        help="Query transformation strategy. Default: multi_query.",
    )
    return parser


def run_ingest(args: argparse.Namespace) -> None:
    deployment = _deployment_from_args(args)
    runtime = build_runtime_configs(
        deployment=deployment,
        collection_name=args.collection,
    )
    loader = MarkdownCorpusLoader(documents_directory=args.documents)
    documents = loader.load()
    pipeline = StructureAwareRAGDemo(
        chunking_config=runtime.chunking,
        embedding_config=runtime.embedding,
        mongo_config=runtime.mongo,
        hybrid_config=runtime.hybrid,
        qdrant_config=runtime.qdrant,
        collection_name=runtime.collection_name,
        cloud_config=runtime.cloud,
    )
    try:
        for index, document in enumerate(documents):
            pipeline.ingest(
                document=document,
                recreate_hybrid_collection=args.recreate and index == 0,
            )
    finally:
        pipeline.close()


def run_search(args: argparse.Namespace) -> None:
    deployment = _deployment_from_args(args)
    querying_config = QueryingConfig(
        mode=QueryTransformationMode(args.mode),
    )
    stack = build_retrieval_stack(
        deployment=deployment,
        collection_name=args.collection,
        querying_config=querying_config,
    )
    try:
        result = stack.orchestrator.retrieve(
            args.query,
            mode=querying_config.mode,
            context=args.context,
        )
        print(json.dumps(result.model_dump(mode="json"), indent=2))
    finally:
        stack.close()


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "search":
        run_search(args)


if __name__ == "__main__":
    main()
