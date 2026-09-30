from abc import ABC, abstractmethod
from typing import Sequence

from pydantic import BaseModel

DenseVector = list[float]

class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]

class EmbeddingService(ABC):
    """Base type for all embedding providers."""


class DenseEmbeddingService(EmbeddingService):

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        query: str,
    ) -> DenseVector:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[DenseVector]:
        raise NotImplementedError


class SparseEmbeddingService(EmbeddingService):

    @abstractmethod
    def embed_query(
        self,
        query: str,
    ) -> SparseVector:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[SparseVector]:
        raise NotImplementedError
