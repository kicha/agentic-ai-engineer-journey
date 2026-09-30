from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from pydantic import BaseModel, Field
from qdrant_client.models import ScoredPoint

from app.embeddings.base import DenseEmbeddingService

from app.querying.multi_query_generator import (
    MultiQueryGenerationResult,
    MultiQueryGenerator,
)

from app.repositories.parent_repository import (
    ParentRepository,
)

from app.reranking.base import Reranker

from app.retrieval.hybrid_child_retriever import (
    HybridChildRetriever,
)

from app.retrieval.models import ChildSearchHit, ParentRetrievalResult

from app.retrieval.parent_mmr_selector import ParentMMRResult
from app.retrieval.parent_mmr_service import ParentMMRService

# ============================================================
# QUERY-SPECIFIC CHILD HIT
# ============================================================


class MultiQueryChildOccurrence(BaseModel):
    query_index: int
    query: str
    rank: int
    hybrid_score: float


# ============================================================
# FUSED CHILD
# ============================================================


class MultiQueryChildCandidate(BaseModel):
    child_id: str
    parent_id: str
    document_id: str
    child_index: int
    content: str
    multi_query_rrf_score: float
    query_hit_count: int
    occurrences: List[MultiQueryChildOccurrence] = Field(default_factory=list)


# ============================================================
# COMPLETE RESULT
# ============================================================


class MultiQueryRetrievalResult(BaseModel):
    generation: MultiQueryGenerationResult
    fused_children: List[MultiQueryChildCandidate]
    candidate_parent_ids: List[str]
    candidate_document_ids: List[str]
    parent_candidates_before_rerank: int
    final_results: List[ParentMMRResult]


# ============================================================
# RETRIEVER
# ============================================================


class MultiQueryHybridRetriever:

    def __init__(
        self,
        query_generator: MultiQueryGenerator,
        hybrid_child_retriever: HybridChildRetriever,
        parent_repository: ParentRepository,
        reranker: Reranker,
        embedder: DenseEmbeddingService,
        lambda_mult: float = 0.7,
        multi_query_rrf_k: int = 60,
        parent_mmr_service: ParentMMRService | None = None,
    ) -> None:

        if multi_query_rrf_k < 0:

            raise ValueError("multi_query_rrf_k cannot be negative.")

        self.query_generator = query_generator
        self.hybrid_child_retriever = hybrid_child_retriever
        self.parent_repository = parent_repository
        self.reranker = reranker
        self.embedder = embedder
        self.multi_query_rrf_k = multi_query_rrf_k
        self.parent_mmr_service = parent_mmr_service or ParentMMRService(
            embedder=embedder,
            lambda_mult=lambda_mult,
        )

    # ========================================================
    # RETRIEVE CHILDREN FOR EACH QUERY
    # ========================================================

    def _retrieve_query_children(
        self,
        queries: List[str],
        child_top_k_per_query: int,
    ) -> List[List[ScoredPoint]]:

        result_lists: List[List[ScoredPoint]] = []

        for query in queries:

            hits = self.hybrid_child_retriever.retrieve_hybrid(
                query=query,
                limit=(child_top_k_per_query),
            )

            result_lists.append(hits)

        return result_lists

    # ========================================================
    # MULTI-QUERY RRF
    # ========================================================

    def _fuse_child_results(
        self,
        queries: List[str],
        result_lists: List[List[ScoredPoint]],
        fused_child_top_k: int,
    ) -> List[MultiQueryChildCandidate]:

        # ----------------------------------------------------
        # child_id -> accumulated data
        # ----------------------------------------------------

        accumulator: Dict[str, dict] = {}

        for (
            query_index,
            (
                query,
                hits,
            ),
        ) in enumerate(
            zip(
                queries,
                result_lists,
            )
        ):

            for rank, hit in enumerate(hits, start=1):
                payload = hit.payload or {}
                child_id = str(
                    payload.get(
                        "child_id",
                        hit.id,
                    )
                )

                parent_id = payload.get("parent_id")

                if not parent_id:

                    raise RuntimeError(
                        "Qdrant child payload "
                        "is missing parent_id. "
                        f"Point ID: {hit.id}"
                    )

                # --------------------------------------------
                # SECOND-LEVEL RRF:
                #
                # We do NOT combine raw Hybrid scores from
                # different query formulations.
                #
                # Only their rank positions contribute.
                # --------------------------------------------

                rrf_contribution = 1.0 / (self.multi_query_rrf_k + rank)

                if child_id not in accumulator:

                    accumulator[child_id] = {
                        "child_id": (child_id),
                        "parent_id": (str(parent_id)),
                        "document_id": (
                            str(
                                payload.get(
                                    "document_id",
                                    "",
                                )
                            )
                        ),
                        "child_index": (
                            int(
                                payload.get(
                                    "child_index",
                                    -1,
                                )
                            )
                        ),
                        "content": (
                            str(
                                payload.get(
                                    "content",
                                    "",
                                )
                            )
                        ),
                        "rrf_score": 0.0,
                        "occurrences": [],
                    }

                accumulator[child_id]["rrf_score"] += rrf_contribution

                accumulator[child_id]["occurrences"].append(
                    MultiQueryChildOccurrence(
                        query_index=(query_index),
                        query=query,
                        rank=rank,
                        hybrid_score=(float(hit.score)),
                    )
                )

        fused: List[MultiQueryChildCandidate] = []

        for item in accumulator.values():

            occurrences = item["occurrences"]

            fused.append(
                MultiQueryChildCandidate(
                    child_id=(item["child_id"]),
                    parent_id=(item["parent_id"]),
                    document_id=(item["document_id"]),
                    child_index=(item["child_index"]),
                    content=(item["content"]),
                    multi_query_rrf_score=(item["rrf_score"]),
                    query_hit_count=(len(occurrences)),
                    occurrences=(occurrences),
                )
            )

        fused.sort(
            key=lambda candidate: (candidate.multi_query_rrf_score),
            reverse=True,
        )

        return fused[:fused_child_top_k]

    # ========================================================
    # CHILDREN -> UNIQUE PARENTS
    # ========================================================

    def _expand_parents(
        self,
        fused_children: List[MultiQueryChildCandidate],
    ) -> List[ParentRetrievalResult]:

        children_by_parent: Dict[
            str,
            List[MultiQueryChildCandidate],
        ] = defaultdict(list)

        for child in fused_children:

            children_by_parent[child.parent_id].append(child)

        parent_results: List[ParentRetrievalResult] = []

        for (
            parent_id,
            fused_parent_children,
        ) in children_by_parent.items():

            parent = self.parent_repository.get_by_id(parent_id)

            if parent is None:

                raise RuntimeError(
                    "Dangling ChildChunk "
                    "reference detected. "
                    f"Missing parent_id="
                    f"{parent_id}"
                )

            # --------------------------------------------
            # Highest multi-query RRF child first.
            # --------------------------------------------

            fused_parent_children.sort(
                key=lambda child: (child.multi_query_rrf_score),
                reverse=True,
            )

            matched_children: List[ChildSearchHit] = []

            for fused_child in fused_parent_children:

                matched_children.append(
                    ChildSearchHit(
                        child_id=(fused_child.child_id),
                        parent_id=(fused_child.parent_id),
                        document_id=(fused_child.document_id),
                        child_index=(fused_child.child_index),
                        content=(fused_child.content),
                        # --------------------------------
                        # At this point score means:
                        # multi-query RRF evidence.
                        #
                        # It is NOT cosine, BM25,
                        # or CrossEncoder probability.
                        # --------------------------------
                        score=(fused_child.multi_query_rrf_score),
                    )
                )

            parent_results.append(
                ParentRetrievalResult(
                    parent=parent,
                    # ------------------------------------
                    # Best fused child is used only to
                    # order candidates BEFORE CE.
                    # ------------------------------------
                    best_score=(fused_parent_children[0].multi_query_rrf_score),
                    matched_children=(matched_children),
                )
            )

        parent_results.sort(
            key=lambda result: (result.best_score),
            reverse=True,
        )

        return parent_results

    # ========================================================
    # PARENT MMR
    # ========================================================

    def _apply_parent_mmr(
        self,
        original_query: str,
        reranked_results,
        final_top_k: int,
    ) -> List[ParentMMRResult]:
        return self.parent_mmr_service.select(
            query=original_query,
            reranked_results=list(reranked_results),
            top_k=final_top_k,
        )

    # ========================================================
    # COMPLETE RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        query: str,
        child_top_k_per_query: int = 15,
        fused_child_top_k: int = 30,
        reranker_top_k: int = 10,
        final_top_k: int = 3,
    ) -> MultiQueryRetrievalResult:

        original_query = query.strip()

        if not original_query:

            raise ValueError("Query cannot be empty.")

        if child_top_k_per_query <= 0:

            raise ValueError("child_top_k_per_query " "must be greater than zero.")

        if fused_child_top_k <= 0:

            raise ValueError("fused_child_top_k must " "be greater than zero.")

        if reranker_top_k <= 0:

            raise ValueError("reranker_top_k must " "be greater than zero.")

        if final_top_k <= 0:

            raise ValueError("final_top_k must " "be greater than zero.")

        if final_top_k > reranker_top_k:

            raise ValueError("final_top_k cannot exceed " "reranker_top_k.")

        # ====================================================
        # 1. GENERATE MULTIPLE QUERY PERSPECTIVES
        # ====================================================

        generation = self.query_generator.generate(original_query)

        # ====================================================
        # 2. HYBRID CHILD RETRIEVAL FOR EVERY QUERY
        # ====================================================

        result_lists = self._retrieve_query_children(
            queries=(generation.all_queries),
            child_top_k_per_query=(child_top_k_per_query),
        )

        # ====================================================
        # 3. CROSS-QUERY CHILD RRF
        # ====================================================

        fused_children = self._fuse_child_results(
            queries=(generation.all_queries),
            result_lists=(result_lists),
            fused_child_top_k=(fused_child_top_k),
        )

        # ====================================================
        # 4. UNIQUE PARENT EXPANSION
        # ====================================================

        parent_candidates = self._expand_parents(fused_children)

        candidate_parent_ids = [
            str(result.parent.parent_id) for result in parent_candidates
        ]

        candidate_document_ids = [
            result.parent.document_id for result in parent_candidates
        ]

        if not parent_candidates:
            return MultiQueryRetrievalResult(
                generation=generation,
                fused_children=fused_children,
                candidate_parent_ids=[],
                candidate_document_ids=[],
                parent_candidates_before_rerank=0,
                final_results=[],
            )

        # ====================================================
        # 5. CROSSENCODER
        #
        # IMPORTANT:
        # use ORIGINAL user query.
        # ====================================================

        reranked_response = self.reranker.rerank(
            query=(original_query),
            candidates=(parent_candidates),
            top_k=(reranker_top_k),
        )

        if not reranked_response.results:
            return MultiQueryRetrievalResult(
                generation=(generation),
                fused_children=(fused_children),
                candidate_parent_ids=[],
                candidate_document_ids=[],
                parent_candidates_before_rerank=(len(parent_candidates)),
                final_results=[],
            )

        # ====================================================
        # 6. ORIGINAL-QUERY PARENT MMR
        # ====================================================

        final_results = self._apply_parent_mmr(
            original_query=(original_query),
            reranked_results=(reranked_response.results),
            final_top_k=(final_top_k),
        )
        return MultiQueryRetrievalResult(
            generation=generation,
            fused_children=fused_children,
            candidate_parent_ids=candidate_parent_ids,
            candidate_document_ids=candidate_document_ids,
            parent_candidates_before_rerank=len(parent_candidates),
            final_results=final_results,
        )
