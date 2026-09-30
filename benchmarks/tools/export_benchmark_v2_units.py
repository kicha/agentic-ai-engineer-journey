import argparse
import logging
from typing import Any

from pydantic import BaseModel
from qdrant_client import QdrantClient

from app.application_factory import build_runtime_configs
from app.config.settings import QdrantDeployment
from app.repositories.parent_repository import ParentRepository
from app.utils.logging_config import configure_logging
from benchmarks.paths import BENCHMARK_UNITS_PATH

logger = logging.getLogger(__name__)
OUTPUT_PATH = BENCHMARK_UNITS_PATH


class ChildExport(BaseModel):
    child_id: str
    child_key: str
    child_index: int
    parent_id: str
    parent_key: str
    document_id: str
    content: str


class ParentExport(BaseModel):
    parent_id: str
    parent_key: str
    parent_index: int
    document_id: str
    section_path: list[str]
    content: str
    children: list[ChildExport]


class BenchmarkUnitsExport(BaseModel):
    benchmark_version: str
    parent_count: int
    child_count: int
    parents: list[ParentExport]


class BenchmarkV2UnitExporter:
    def __init__(self, *, deployment: QdrantDeployment, collection_name: str | None = None) -> None:
        self.runtime = build_runtime_configs(
            deployment=deployment,
            collection_name=collection_name,
        )
        self.parent_repository = ParentRepository(self.runtime.mongo)

        if deployment == QdrantDeployment.LOCAL:
            self.qdrant_client = QdrantClient(
                url=self.runtime.qdrant.local_url,
                timeout=self.runtime.qdrant.timeout_seconds,
            )
        else:
            if not self.runtime.qdrant.cloud_url or not self.runtime.qdrant.api_key:
                raise ValueError("QDRANT_URL and QDRANT_API_KEY are required for cloud export.")
            self.qdrant_client = QdrantClient(
                url=self.runtime.qdrant.cloud_url,
                api_key=self.runtime.qdrant.api_key,
                cloud_inference=self.runtime.qdrant.cloud_inference,
                timeout=self.runtime.qdrant.timeout_seconds,
            )

    def export(self) -> BenchmarkUnitsExport:
        parents = self._load_parents()
        children = self._load_children()
        children_by_parent_key: dict[str, list[ChildExport]] = {}

        for child in children:
            children_by_parent_key.setdefault(child.parent_key, []).append(child)

        exported_parents: list[ParentExport] = []
        for parent in parents:
            parent_children = sorted(
                children_by_parent_key.get(parent.parent_key, []),
                key=lambda child: child.child_index,
            )
            exported_parents.append(
                ParentExport(
                    parent_id=parent.parent_id,
                    parent_key=parent.parent_key,
                    parent_index=parent.parent_index,
                    document_id=parent.document_id,
                    section_path=parent.section_path,
                    content=parent.content,
                    children=parent_children,
                )
            )

        result = BenchmarkUnitsExport(
            benchmark_version="2.0",
            parent_count=len(exported_parents),
            child_count=len(children),
            parents=exported_parents,
        )
        logger.info(f"parent_count={result.parent_count}")
        logger.info(f"child_count={result.child_count}")
        return result

    def _load_parents(self) -> list[Any]:
        documents = list(self.parent_repository.collection.find({}))
        parents = [
            self.parent_repository._from_document(document) for document in documents
        ]
        parents.sort(key=lambda parent: (parent.document_id, parent.parent_index))
        return parents

    def _load_children(self) -> list[ChildExport]:
        children: list[ChildExport] = []
        next_offset = None
        while True:
            points, next_offset = self.qdrant_client.scroll(
                collection_name=self.runtime.collection_name,
                limit=100,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload
                if payload is None:
                    raise RuntimeError(f"Qdrant point has no payload: point_id={point.id}")
                children.append(
                    ChildExport(
                        child_id=payload["child_id"],
                        child_key=payload["child_key"],
                        child_index=payload["child_index"],
                        parent_id=payload["parent_id"],
                        parent_key=payload["parent_key"],
                        document_id=payload["document_id"],
                        content=payload["content"],
                    )
                )
            if next_offset is None:
                break

        children.sort(
            key=lambda child: (
                child.document_id,
                child.parent_key,
                child.child_index,
            )
        )
        return children

    def close(self) -> None:
        self.parent_repository.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true")
    group.add_argument("--cloud", action="store_true")
    parser.add_argument("--collection", default=None)
    args = parser.parse_args()

    deployment = QdrantDeployment.LOCAL if args.local else QdrantDeployment.CLOUD
    exporter = BenchmarkV2UnitExporter(
        deployment=deployment,
        collection_name=args.collection,
    )
    try:
        result = exporter.export()
        OUTPUT_PATH.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"benchmark_units_written={OUTPUT_PATH}")
    finally:
        exporter.close()


if __name__ == "__main__":
    configure_logging()
    main()
