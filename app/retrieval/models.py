from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.parent_chunk import ParentChunk


class ChildSearchHit(BaseModel):
    child_id: str
    parent_id: str
    document_id: str
    child_index: int
    content: str
    score: float


class ParentRetrievalResult(BaseModel):
    parent: ParentChunk
    best_score: float
    matched_children: list[ChildSearchHit] = Field(default_factory=list)
