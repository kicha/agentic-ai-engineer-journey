from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, TypeAlias

from pydantic import BaseModel

DenseVector: TypeAlias = list[float]


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class SparseText(BaseModel):
    """Text that Qdrant Cloud should embed server-side with a sparse model."""

    text: str
    model: str


SparseEmbedding: TypeAlias = SparseVector | SparseText


class EmbeddingService(ABC):
    """Base type for all embedding providers."""


class DenseEmbeddingService(EmbeddingService):
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> DenseVector:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[DenseVector]:
        raise NotImplementedError


class SparseEmbeddingService(EmbeddingService):
    @abstractmethod
    def embed_query(self, query: str) -> SparseEmbedding:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[SparseEmbedding]:
        raise NotImplementedError
