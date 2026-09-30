from types import SimpleNamespace

from app.embeddings.base import DenseEmbeddingService, DenseVector
from app.models.parent_chunk import ParentChunk
from app.retrieval.parent_mmr_service import ParentMMRService


class FakeDenseEmbedder(DenseEmbeddingService):
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def embedding_dimension(self) -> int:
        return 3

    def embed_query(self, query: str) -> DenseVector:
        self.query_calls.append(query)
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts) -> list[DenseVector]:
        values = list(texts)
        self.document_calls.append(values)
        mapping = {
            "parent one": [1.0, 0.0, 0.0],
            "parent two": [0.0, 1.0, 0.0],
        }
        return [mapping[text] for text in values]


def _parent(parent_id: str, content: str, content_hash: str) -> ParentChunk:
    return ParentChunk.model_construct(
        parent_id=parent_id,
        parent_key=f"key-{parent_id}",
        parent_index=0,
        document_id="DOC-001",
        content=content,
        content_hash=content_hash,
    )


def _reranked(parent: ParentChunk, rank: int, score: float):
    return SimpleNamespace(
        result=SimpleNamespace(parent=parent),
        reranked_rank=rank,
        reranker_score=score,
    )


def test_parent_embeddings_are_reused_across_mmr_calls() -> None:
    embedder = FakeDenseEmbedder()
    service = ParentMMRService(embedder=embedder, lambda_mult=0.7)
    reranked = [
        _reranked(_parent("p1", "parent one", "h1"), 1, 0.9),
        _reranked(_parent("p2", "parent two", "h2"), 2, 0.8),
    ]

    first = service.select(query="query one", reranked_results=reranked, top_k=2)
    second = service.select(query="query two", reranked_results=reranked, top_k=2)

    assert len(first) == 2
    assert len(second) == 2
    assert embedder.document_calls == [["parent one", "parent two"]]
    assert embedder.query_calls == ["query one", "query two"]
