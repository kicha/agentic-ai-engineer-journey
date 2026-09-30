# AGENTIC AI ENGINEER MENTORSHIP — MASTER CONTINUITY CHECKPOINT

## 1. PURPOSE

This is the canonical, portable continuity document for the Agentic AI Engineer mentorship.

The goal is that this checkpoint + the course roadmap file:

    AI_Engineer_Day_by_Day_Plan_Aug11_Dec31_2026.xlsx

can be given to another ChatGPT account/user and that mentor should be able to reproduce the same mentorship experience, teaching philosophy, roadmap continuity, project architecture, implementation discipline, and current technical state.

This is not merely a summary. It is the mentorship state + technical state + continuity contract.

## 2. MENTORSHIP IDENTITY

- Mentor role: Principal AI Architect / world-class Agentic RAG mentor.
- Goal: make the learner interview-ready for LLM / GenAI / Agentic AI engineering roles.
- Build a production-oriented Agentic RAG project while learning the underlying concepts.
- Follow the day-by-day roadmap in AI_Engineer_Day_by_Day_Plan_Aug11_Dec31_2026.xlsx.

## 3. MANDATORY TEACHING STYLE

For coding topics:
1. Start with a complete, runnable, OOP-based, Pythonic implementation.
2. Prefer Pydantic models over dataclasses.
3. Give complete code first.
4. Then explain architecture, concepts, WHY the design is used, and code file-by-file/class-by-class.
5. Run experiments with the learner and interpret the learner's actual output.
6. Discuss production implications and interview questions where appropriate.

Handholding rules:
- Teach step-by-step.
- Do not jump several architectural layers ahead.
- Do not give fragmented snippets when a complete implementation is appropriate.
- When modifying existing code, identify the exact file and exact change.
- Preserve established architecture unless there is a strong technical reason to change it.

## 4. ROADMAP AUTHORITY

Canonical roadmap:
    AI_Engineer_Day_by_Day_Plan_Aug11_Dec31_2026.xlsx

The spreadsheet is authoritative for daily topic, phase, week, learning mode, implementation, experimentation, integration, testing, recall/interview, weekly review, and proof of learning.

Do not start a new course topic merely because an implementation issue takes longer. Complete the current learning objective properly, then catch up according to the roadmap.

## 5. CURRENT COURSE POSITION

Program: Agentic AI Engineer / Agentic RAG
Current major area: RAG → Chunking + Retrieval → Production RAG
Current detailed topic: **Structure-Aware Parent/Child Chunking — Production Architecture**
Current implementation phase: **Parent/Child persistence and Qdrant integration**

Current unresolved issue:
The parent chunker produces 7 parent chunks in Python, but MongoDB collection/database `parents` shows only one document.

Immediate objective:
Find the persistence problem using the actual current implementation and make parent persistence correct and production-oriented.

## 6. COMPLETED / COVERED TOPICS

- Fixed-size chunking
- Chunk size and overlap experiments
- Token-based chunking concepts
- Recursive character chunking
- Sentence chunking
- Semantic chunking
- Semantic chunking buffers, breakpoints, distances, thresholds
- Mean / standard deviation / IQR / gradient concepts for semantic breakpoints
- Structure-aware chunking
- Structure-Aware Chunking V2
- Structure-Aware Chunking V3
- Production metadata schema design
- Pydantic migration
- Metadata field design and rationale
- Chunk IDs / UUID / idempotency discussion
- Initial Parent/Child chunking implementation
- Parent/Child architectural concepts
- Parent document retrieval concepts
- Qdrant integration and vector search
- HNSW concepts and recall/latency experiments
- SentenceTransformers embedding experiments
- all-MiniLM-L6-v2 (384-dimensional embeddings) in current chunking experiments

## 7. CONFIRMED PROJECT STRUCTURE

The project is organized under an `app/` package.

Confirmed top-level areas:

    app/
    ├── config/
    ├── models/
    ├── chunking/
    ├── repositories/
    └── vector_store/

Known/confirmed modules from the established work:

    app/
    ├── config/
    │   └── settings.py
    ├── models/
    │   ├── document.py
    │   └── parent_chunk.py
    ├── chunking/
    │   ├── structure_detector.py
    │   └── structure_aware_parent.py
    ├── repositories/
    │   └── parent_repository.py
    └── vector_store/
        └── [Qdrant vector-store implementation]

Confirmed negative facts:
- There is NO `app/storage` package.
- There is NO `parent_store.py`.
- `app/repositories/parent_repository.py` was empty at the last explicitly confirmed point unless subsequently changed by the learner.

Project-structure integrity rule:
The latest learner-provided tree/directory listing/source is authoritative and must replace this section when newer evidence is supplied. Never invent missing files.

## 8. CURRENT CODE / ARCHITECTURE AUTHORITY

The latest actual learner-provided source code is authoritative.

There have been multiple Structure-Aware Chunking versions, including V2/V3 and later modular Parent/Child work.

Never reconstruct the latest implementation from memory. If the current file is unavailable, ask for that file or retrieve the supplied project file if available. Do not silently give an older V3 implementation.

## 9. V3 DESIGN CONTEXT

Confirmed V3 concepts/classes included:
- Document
- HeadingContext
- StructuralElement
- ChunkMetadata
- Chunk
- StructureDetector
- StructureAwareChunker
- EmbeddingService
- QdrantVectorStore
- ChunkIdGenerator
- MetadataEnricher
- ContentType

V3 used Pydantic, structural metadata (h1-h6/section path), Qdrant payload metadata, and all-MiniLM-L6-v2 with 384-dimensional embeddings in the demonstrated experiment.

These details describe confirmed V3 context only. They must NOT overwrite newer modular code.

## 10. CURRENT PARENT/CHILD DEBUGGING STATE

Observed:
    Parent chunks generated: 7
    MongoDB parents documents: 1

During debugging verify:
1. Number of ParentChunk objects returned by the chunker.
2. Their IDs and uniqueness.
3. Repository method receiving them.
4. Whether persistence is one-by-one or bulk.
5. Whether all seven are passed to the repository.
6. Insert vs replace/upsert behavior.
7. Whether `_id` is accidentally reused.
8. Serialization/model conversion.
9. Actual MongoDB contents after the write.

Do not assume the root cause before inspecting the actual parent model and repository.

## 11. QDRANT CONTEXT

We discussed:

    def insert_chunks(
        self,
        chunks: List[Chunk],
        embeddings,
    ):

Determine the exact current model/type from the latest project architecture before changing it. Do not blindly replace `Chunk` with `ParentChunk`.

Conceptually, child chunks are normally the retrievable/vectorized units, while parent chunks provide expanded context; the actual current implementation remains authoritative.

## 12. PERSISTENCE / DATABASE ARCHITECTURE PRINCIPLE

Conceptual production flow:

    User Query
        ↓
    Query Embedding
        ↓
    Qdrant retrieves ChildChunk(s)
        ↓
    parent_chunk_id
        ↓
    MongoDB retrieves ParentChunk
        ↓
    Expanded contextual evidence
        ↓
    LLM / Agent

This is an architectural checkpoint, not permission to overwrite current implementation.

## 13. ID / IDEMPOTENCY PRINCIPLES

- UUIDs can uniquely identify chunk instances.
- Deterministic IDs can support idempotent ingestion when derived from stable document/position/content strategy.
- Document changes can change chunk boundaries and therefore identities depending on the chosen strategy.
- Parent/child IDs must have explicit stable relationships.
- Never call an ID strategy idempotent without checking its actual generation method.

## 14. ANTI-DRIFT RULE — CRITICAL

1. Latest learner-provided code overrides remembered code.
2. Latest confirmed project structure overrides remembered structure.
3. If versions conflict, do not guess.
4. Never silently combine pieces from different versions.
5. Never invent packages, files, repositories, classes, methods, collections, or configuration.
6. If current source is unavailable, explicitly request the required file.
7. Before structural changes, state current architecture, problem, proposed change, and why it fits.
8. In a new chat, do not restart the topic unless the learner asks for revision.

This rule exists specifically to prevent the previous continuity failure where a new chat produced an unrelated V3 implementation.

## 15. NEW-CHAT CONTINUATION PROTOCOL

When the learner says "Continue my Agentic AI mentorship":
1. Use this checkpoint as the baseline.
2. Use the roadmap XLSX as the course schedule.
3. Identify exact current topic and implementation phase.
4. Identify exact current bug/task.
5. Review latest available source code.
6. If latest source is unavailable, ask only for the required current file(s).
7. Never recreate old code from memory.
8. Continue from the last confirmed step.
9. Preserve the same teaching style and handholding approach.
10. Update the checkpoint after the session.

Recommended starter:

"Continue my Agentic AI Engineer mentorship from the latest `AGENTIC_RAG_MENTORSHIP_STATE.md` checkpoint and `AI_Engineer_Day_by_Day_Plan_Aug11_Dec31_2026.xlsx`. Use the checkpoint as the continuity and teaching contract. Use my latest actual source code as the implementation authority. Do not recreate an older implementation from memory. Continue from the exact current step."

## 16. DAILY CHECKPOINT HABIT

A checkpoint is required at the end of EVERY mentorship session, when the learner signals that the day's session is finished or at a clearly identifiable end-of-session point.

Every daily update should capture:
- Date
- Roadmap day/week/phase/theme/mode
- Planned objective and completion status
- Concepts learned
- Files/classes changed
- Architecture/model/database/vector-store decisions
- Experiments, parameters, actual results, interpretation
- Bugs, root cause if known, fixes, remaining issues
- Exact point where the session stopped
- Exact next action
- Files needed next
- Interview/retention notes where relevant

Also checkpoint immediately after major bug resolution, architecture change, new module/file, schema change, database/vector-store design change, major topic completion, weekly review, or roadmap catch-up decision.

## 17. DAILY CHECKPOINT TEMPLATE

    ## Daily Checkpoint — YYYY-MM-DD

    ### Roadmap
    - Day:
    - Week:
    - Phase:
    - Theme:
    - Mode:

    ### Planned Objective
    ...

    ### Completed Today
    ...

    ### Concepts Learned
    ...

    ### Code / Project Changes
    ...

    ### Experiments
    ...

    ### Actual Results
    ...

    ### Interpretation
    ...

    ### Bugs / Issues
    ...

    ### Decisions Made
    ...

    ### Current Project State
    ...

    ### Exact Next Step
    ...

    ### Files Relevant to Next Session
    ...

    ### Mentor Notes / Interview Readiness
    ...

    ### Checkpoint Status
    Session checkpoint completed.

## 18. PORTABILITY REQUIREMENT

This checkpoint must remain understandable without access to the original chat.

A new mentor/account receiving this file plus the roadmap XLSX should understand:
- the learner's career objective,
- how the learner must be taught,
- the roadmap governing the course,
- current course position,
- completed topics,
- project architecture,
- authoritative-code rules,
- current bugs/tasks,
- next step,
- and how to continue without resetting the course.

Preferred portable package:

    AGENTIC_RAG_MENTORSHIP_STATE.md
    AI_Engineer_Day_by_Day_Plan_Aug11_Dec31_2026.xlsx
    latest actual project source code (when implementation continuity is required)

## 19. CURRENT CHECKPOINT STATUS

Checkpoint established: 2026-09-02

Current state:
Parent/Child Chunking — parent persistence/synchronization and ChildChunk generation are complete; next is Qdrant child persistence/synchronization.

Next session:
Embed and persist the 19 ChildChunks using the existing EmbeddingService and QdrantVectorStore, then prove and solve stale-child/vector synchronization behavior.

## 20. FINAL CONTINUITY PRINCIPLE

The checkpoint must preserve not only WHAT we studied, but HOW we teach, WHERE we are, WHAT code is authoritative, WHY decisions were made, WHAT experiments showed, and EXACTLY WHAT comes next.

**Continuity over reconstruction.**
**Actual code over remembered code.**
**Roadmap over random topic switching.**
**Daily checkpoint over reliance on one chat window.**


## Daily Checkpoint — 2026-09-03

### Roadmap
- Learning date / roadmap position: Aug 19, 2026
- Week: Week 2 — Chunking + Retrieval
- Phase: Phase 1 — RAG
- Theme: Structure-Aware Parent/Child Chunking — Production Architecture
- Mode: Experiment / production persistence behavior

### Planned Objective
Complete ParentChunk persistence behavior, prove idempotency and synchronization semantics, then implement ChildChunk generation and verify parent-child linkage before Qdrant persistence.

### Completed Today
- Confirmed StructureAwareParentChunker generates 7 unique ParentChunks with 7 unique UUID parent IDs and 7 unique content hashes.
- Implemented ParentRepository for MongoDB persistence.
- Proven 7 generated parents persist as 7 MongoDB documents and deserialize/read back correctly.
- Demonstrated duplicate ingestion problem: random UUID4 application IDs caused repeated insert behavior to grow 7 -> 14 when using plain insert semantics.
- Added idempotent upsert using logical identity `(document_id, content_hash)` plus MongoDB unique compound index.
- Proven repeated ingestion of unchanged source inserts 0 new records and remains at 7.
- Demonstrated changed-document behavior: editing two parent contents changed two hashes and upsert-only semantics grew MongoDB from 7 -> 9, revealing stale-version retention.
- Implemented `sync_parents()` current-state reconciliation: classify incoming/current/stale hashes, insert new parents, preserve unchanged parents, delete stale parents.
- Proven synchronization from 9 -> 7: incoming=7, existing_before=9, inserted=0, unchanged=7, stale_deleted=2, final=7.
- Ran parent-ID stability experiment: after editing only `ef_construct`, synchronization reported inserted=1, unchanged=6, stale_deleted=1, final=7. Six unchanged MongoDB parent IDs were preserved; the changed parent received a new parent ID and content hash.
- Implemented `app/chunking/child_chunker.py`.
- Added child chunk configuration: child chunk size 120 chars and overlap 30 chars.
- ChildChunker splits persisted MongoDB ParentChunks, preserves heading/section metadata, creates child content hashes, and assigns `child.parent_id` to the actual persisted parent ID.
- Generated 19 ChildChunks from 7 persisted parents.
- Referential-integrity check passed: 7 stored parent IDs, 7 referenced parent IDs, invalid parent references = empty set.

### Concepts Learned
- Uniqueness != idempotency != synchronization.
- UUID4 provides instance uniqueness but not logical idempotency.
- `(document_id, content_hash)` is useful as a current experimental logical identity for unchanged content, but is not yet the final universal parent occurrence identity because identical text may appear more than once in one document.
- A full document synchronization operation needs the complete desired parent set; therefore it belongs at ingestion/workflow level and must not be triggered implicitly from every low-level CRUD method.
- Insert-new-before-delete-stale is safer than delete-first when no transaction is used.
- Unchanged persisted parents keep their original MongoDB `_id` / `parent_id` even though fresh chunking generates new UUIDs in memory.
- Children must therefore be generated from synchronized/stored ParentChunks, not blindly from fresh pre-sync ParentChunks.
- Parent-child referential integrity is critical: a child in Qdrant must always point to a parent that currently exists in MongoDB.
- Child overlap preserves semantic continuity at child boundaries.

### Code / Project Changes
- `app/config/settings.py`
  - MongoDB configuration fields added earlier.
  - Child chunk configuration added: `child_chunk_chars=120`, `child_overlap_chars=30`.
- `app/repositories/parent_repository.py`
  - MongoClient-based persistence.
  - Serialization maps `ParentChunk.parent_id` to MongoDB `_id`.
  - Compound indexes retained: `(document_id, parent_index)` and unique `(document_id, content_hash)`.
  - Explicit index names removed during experiment because equivalent auto-named indexes already existed and MongoDB raised IndexOptionsConflict.
  - `upsert_parent`, `upsert_parents`, read/count/delete/ping/close methods.
  - Added `ParentSyncResult` Pydantic model and `sync_parents()`.
- `app/chunking/child_chunker.py`
  - Implemented complete OOP ChildChunker.
  - Public APIs: `chunk_parent()` and `chunk_parents()`.
  - Character-based child splitting with whitespace preference and overlap.
  - Metadata inherited via deep copy; child char count, chunk strategy/version and parent reference updated.
- `main.py`
  - Added persistence/synchronization diagnostics, parent identity diagnostics, child chunking experiment and referential-integrity check.

### Experiments and Actual Results
1. Parent identity generation:
   - Total parents: 7
   - Unique parent IDs: 7
   - Unique content hashes: 7

2. MongoDB bulk persistence/readback:
   - Generated: 7
   - Inserted/read back: 7

3. Duplicate ingestion with non-idempotent insert semantics:
   - First run: 7
   - Second run: 14

4. Idempotent upsert on unchanged source:
   - New parents inserted: 0
   - Parents currently in MongoDB: 7

5. Changed source with upsert-only semantics:
   - Two changed parent hashes
   - New parents inserted: 2
   - MongoDB: 7 -> 9

6. Parent synchronization from stale state:
   - Incoming parents: 7
   - Existing before sync: 9
   - New parents inserted: 0
   - Unchanged parents: 7
   - Stale parents deleted: 2
   - Parents after sync: 7

7. Parent-ID stability after one-parent edit:
   - Existing before sync: 7
   - New parents inserted: 1
   - Unchanged parents: 6
   - Stale parents deleted: 1
   - Final: 7
   - Six unchanged stored parent IDs preserved.
   - Edited `ef_construct` parent got a new stored parent ID/hash.

8. ChildChunk experiment:
   - Stored parents: 7
   - Children: 19
   - Stored parent IDs: 7
   - Referenced parent IDs: 7
   - Invalid parent references: empty set

### Bugs / Issues Resolved
- Original 7-parent/1-MongoDB-document concern was not reproduced by the uploaded current source because the repository had originally been empty; persistence was implemented explicitly and proven 7 -> 7.
- MongoDB `IndexOptionsConflict`: same index key pattern already existed under auto-generated name (`document_id_1_parent_index_1`) while code attempted a different explicit name. Resolution: omit explicit names and reuse existing equivalent index definitions for this experiment.
- PyMongo `InvalidOperation: Cannot use MongoClient after close`: readback code executed after `repository.close()`. Resolution: all repository operations remain inside `try`; `close()` runs only in `finally` after DB work is complete.

### Decisions Made
- ParentChunks are persisted in MongoDB; ChildChunks are the future vectorized/retrievable units in Qdrant.
- ParentRepository remains storage-specific and separate from StructureAwareParentChunker.
- `parent_id` remains application/persistent parent identity and maps to MongoDB `_id`.
- `(document_id, content_hash)` remains the current experimental idempotency/sync key; explicitly not yet considered the final universal stable occurrence identity.
- `sync_parents()` is the normal full-document persistence behavior to be orchestrated automatically by the ingestion layer later, not automatically invoked by every CRUD method.
- ChildChunks must be generated from parents read back after synchronization so their `parent_id` values reference the actual persisted parent records.
- Do not move to retrieval until Qdrant child persistence and stale-child synchronization semantics are understood.

### Current Project State
Current proven ingestion path:

    Source Document
        -> StructureDetector
        -> StructureAwareParentChunker
        -> fresh ParentChunks
        -> ParentRepository.sync_parents()
        -> MongoDB current parent state
        -> read persisted ParentChunks
        -> ChildChunker
        -> 19 ChildChunks with valid persisted parent references

Conceptual next path:

    ChildChunks
        -> EmbeddingService
        -> QdrantVectorStore
        -> child synchronization / stale vector cleanup
        -> ParentChildRetriever

### Exact Stopping Point
Parent persistence, idempotency, current-state synchronization, parent-ID stability and ChildChunk generation/linkage are proven. Qdrant persistence of the 19 children has NOT yet been executed in the current Parent/Child production flow.

### Exact Next Step
Use the existing `EmbeddingService` and existing `QdrantVectorStore.insert_children()` from the latest uploaded project source to:
1. embed the 19 ChildChunks,
2. persist them in Qdrant,
3. verify point count and payload parent IDs,
4. re-ingest unchanged source and observe child ID behavior,
5. edit one parent and demonstrate stale-child/vector behavior,
6. design and implement Qdrant child synchronization so obsolete children cannot be retrieved.

### Files Relevant to Next Session
- `app/models/child_chunk.py`
- `app/chunking/child_chunker.py`
- `app/embeddings/embedding_service.py`
- `app/vectorstore/qdrant_store.py`
- `app/repositories/parent_repository.py`
- `app/ingestion/pipeline.py`
- `main.py`
- latest project ZIP/source if code changed after the uploaded archive

### Mentor Notes / Interview Readiness
The learner can now explain and experimentally demonstrate the difference between uniqueness, idempotency and synchronization; explain why unchanged parents retain persisted IDs after reconciliation; explain why children must be created from persisted parents; and explain the dangling-reference failure mode that occurs when stale children reference deleted parents.

### Checkpoint Status
Session checkpoint completed.

### Open Conceptual Question at Session End
Learner asked whether Parent/Child chunking provides enough benefit to justify maintaining both MongoDB parents and Qdrant children, noting that if query matches children under Parent A and Parent B then both full parents may be sent to the LLM anyway. This should be answered with concrete retrieval scenarios and tradeoffs before closing the session.


## Pending Future Improvements — 2026-09-06

The following improvements are intentionally deferred and must remain visible in future mentorship sessions until completed:

1. **Differential Qdrant child synchronization**
   - Current `sync_children()` behavior is correct but coarse-grained: delete all Qdrant children for a `document_id`, then insert the full current child set.
   - Future improvement: introduce stable/deterministic logical ChildChunk identity and differential synchronization so unchanged children are preserved, new children are inserted, and stale children are deleted without rewriting the entire document.
   - Desired future behavior for unchanged ingestion: inserted=0, unchanged=N, stale_deleted=0.
   - Desired future behavior for partial edits: preserve unchanged child vectors, insert only new/changed children, delete only stale children.
   - This is an ingestion efficiency/production-hardening improvement; current-state correctness is already proven by the 57 -> 19 synchronization experiment.

2. **Reranking after vector retrieval**
   - Current ParentChildRetriever uses child vector similarity, score thresholding, parent grouping/deduplication, and parent ranking by best child score.
   - Retrieval experiment showed that high cosine similarity does not always imply answer relevance (for example, the broad Vector Databases parent also scored highly for the metadata query).
   - Future improvement: add a reranking stage after candidate retrieval so final evidence is ranked by query-answer relevance rather than embedding similarity alone.
   - Reranking should be implemented in the later advanced retrieval/reranking roadmap stage, not as a blocker for completing the current Parent/Child architecture lesson.

### Current Decision
- Proceed forward with the roadmap.
- Treat both differential Qdrant synchronization and reranking as **explicit pending future improvements**, not completed features.
- Do not silently mark either item complete in future chats without an implementation and experiment proving it.


## Daily Checkpoint — 2026-09-06 (Session End)

### Roadmap
- Phase: Phase 1 — RAG
- Week 2 — Chunking + Retrieval: closure completed
- Next roadmap area: Week 3 — Reranking + Advanced Retrieval
- Immediate next lesson: Cross-Encoder Reranking

### Completed Today
- ParentChildRetriever end-to-end retrieval validated with live Qdrant child search + MongoDB parent expansion.
- Added child similarity threshold and parent_top_k retrieval controls.
- Ran four retrieval-quality experiments:
  - ef_construct query: correct parent rank #1, weak parents filtered.
  - M parameter query: correct parent rank #1; broader HNSW parent also relevant.
  - metadata query: correct Metadata parent rank #1, but broad Vector Databases parent still scored highly, demonstrating that embedding similarity is not identical to answer relevance.
  - HNSW search query: two complementary HNSW parents retrieved, both useful.
- Added/ran ParentChildRetriever behavior tests; all passed:
  - correct parent ranked first,
  - empty query rejected,
  - score threshold filters weak parents,
  - duplicate child hits deduplicate to unique parents,
  - parent_top_k limits final context,
  - zero-result retrieval is allowed,
  - dangling child -> missing parent reference raises RuntimeError.
- Week 2 recall/interview review completed. Learner correctly explained child precision vs parent context, persisted parent IDs, child_top_k / score threshold / parent_top_k, parent deduplication, dangling references, and abstention rationale; refined concept that dense embedding cosine similarity measures semantic relatedness, not fixed keyword weighting or guaranteed answer relevance.

### Current Retrieval Architecture
Query -> query embedding -> Qdrant ChildChunk retrieval -> child score threshold -> group/deduplicate by parent_id -> MongoDB ParentChunk lookup -> rank parents by best child score -> parent_top_k -> final context.

### Pending Future Improvements (must remain open until explicitly implemented and proven)
1. Differential Qdrant child synchronization
   - Current sync_children() remains coarse-grained: delete all children for document_id, then insert current set.
   - Future improvement requires stable/deterministic ChildChunk identity plus differential keep/insert/delete semantics.
2. Reranking after vector retrieval
   - Next roadmap stage begins Cross-Encoder Reranking.
   - Motivation proven by metadata experiment: high cosine similarity can still produce a broad passage that is semantically related but not answer-focused.

### Framework / LangChain Decision
- LangChain is intentionally part of the mentorship/course and WILL be integrated.
- Teaching approach: fundamentals-first, framework-second. First understand and build the core mechanics directly (chunking, embedding, vector search, parent/child retrieval, synchronization, reranking concepts), then introduce LangChain abstractions to reduce boilerplate, compare behavior/tradeoffs, and integrate them into the same running project.
- Do not replace conceptual understanding with framework magic. The learner should be able to explain what LangChain abstractions are doing underneath and know when direct SDK/custom code is preferable.
- Future course work should explicitly include LangChain retrieval/chains/tooling where it fits, followed later by LangGraph for agent workflows/state/loops.
- When introducing LangChain, preserve the current project architecture and integrate rather than starting a disconnected demo project.

### Next Session
Start Week 3 — Cross-Encoder Reranking.
Planned progression:
1. Conceptual distinction: bi-encoder retrieval vs cross-encoder reranking.
2. Add a clean OOP/Pydantic reranking module to the existing project.
3. Rerank ParentChildRetriever candidates and compare vector-only vs vector + reranker on the same queries.
4. Continue advanced retrieval topics in sequence: MMR, query rewriting, multi-query retrieval, HyDE, and hybrid search, with conceptual coverage and implementation/experiments where appropriate.
5. Introduce LangChain abstractions at the appropriate point to reduce boilerplate and compare with the custom implementation.

### Checkpoint Status
Session checkpoint completed.


---

## Daily Checkpoint — 2026-09-11 (Session Boundary Before Hybrid Search)

### Roadmap Position
- Phase: Phase 1 — RAG
- Week 3 — Reranking + Advanced Retrieval
- Cross-Encoder Reranking: completed and experimentally validated
- MMR: completed at conceptual, from-scratch implementation, real Parent/Child integration, and pipeline-comparison level
- Immediate next topic: **Hybrid Search**
- Next-window instruction: start Hybrid Search from this checkpoint; do not repeat basic CrossEncoder/MMR implementation unless needed for comparison or recall.

### Framework Decision — New Default Going Forward
The learner explicitly requested a production-framework-first approach from this point forward.

New coding policy:
1. Prefer **LangChain** for production RAG components where a solid maintained abstraction exists (retrievers, vector-store integrations, MMR, reranking wrappers/integration, hybrid retrieval, document pipelines, standard RAG composition).
2. Prefer **LangGraph** later for stateful agent orchestration, loops, branching, retries, HITL, and multi-step agent workflows.
3. Fall back to direct Python/provider SDKs only when the framework lacks the required capability, introduces an unsuitable abstraction, or custom production behavior is necessary.
4. Preserve fundamentals-understood / framework-first execution: the learner should know what the abstraction is doing underneath and be able to debug it.
5. Do not start a disconnected LangChain toy project. Integrate LangChain into the existing canonical Agentic RAG project architecture.
6. Frameworks reduce boilerplate and maintenance surface but are not assumed error-free; version churn, defaults, and abstraction-specific behavior should still be inspected.

### Cross-Encoder Reranking — Completed
Implemented `app/reranking/cross_encoder_reranker.py` with model:
- `cross-encoder/ms-marco-MiniLM-L6-v2`

Key behavior proven:
- Broad vector candidate retrieval first; CrossEncoder then performs stronger query-document relevance ranking.
- Metadata query: CrossEncoder corrected broad vector-similarity ordering.
- M-parameter query: direct answer parent strongly separated from broad HNSW context.
- HNSW efficiency query exposed query-phrasing sensitivity: the useful search-process parent scored very low for `How does HNSW search efficiently?` but scored ~0.9942 for the rewritten query `Why does HNSW avoid comparing the query against every vector in the database?`
- Therefore CrossEncoder sigmoid outputs are not treated as universally calibrated probabilities.

### MMR — Completed
Implemented and experimentally studied Maximal Marginal Relevance from scratch.

Core formula:
`MMR(d) = lambda * Sim(query,d) - (1-lambda) * max Sim(d, selected)`

Controlled corpus experiment proved lambda behavior:
- lambda=1.0 -> B, A, C (pure relevance / same as similarity ranking)
- lambda=0.7 -> B, C, A (mild diversity)
- lambda=0.5 -> B, C, D (stronger diversity)
- lambda=0.2 -> B, E, D (diversity dominates and can promote weak/off-topic content)

Production lesson: MMR is not a relevance oracle. It optimizes relevance + non-redundancy and can promote weak items when lambda is too low.

### Real Parent/Child MMR Integration
Added/used these retrieval components during the lesson:
- `app/retrieval/mmr_retriever.py`
- `app/retrieval/mmr_parent_child_retriever.py`
- `app/retrieval/mmr_rerank_retriever.py`
- `app/retrieval/parent_mmr_selector.py`
- `app/retrieval/rerank_mmr_retriever.py`
- `scripts/test_mmr.py`
- `scripts/compare_retrieval_strategies.py`

Qdrant search was extended with an optional `with_vectors` flag so MMR could reuse stored child vectors instead of re-embedding retrieved child text. Existing search behavior remains default-compatible with `with_vectors=False`.

### Retrieval Strategy Comparison — Query
Query used repeatedly:
`How does HNSW search efficiently?`

#### 1. Plain similarity retrieval
Top parents:
1. Parent 2 — HNSW overview — best child score 0.6820
2. Parent 3 — HNSW search process — 0.5333
3. Parent 6 — Metadata — 0.3931

Conclusion: good top 2, noisy third result.

#### 2. Child-level MMR
With lambda=0.7, child-level MMR selected diverse children that collapsed to:
1. Parent 2 — HNSW overview
2. Parent 3 — HNSW search process
3. Parent 1 — Qdrant

Conclusion: MMR removed Metadata but introduced Qdrant because Qdrant was less redundant. Diversity improved, relevance was not guaranteed.

#### 3. CrossEncoder only
Top parents:
1. Parent 2 — HNSW overview — CrossEncoder ~0.9994
2. Parent 3 — HNSW search process — ~0.0032
3. Parent 1 — Qdrant — ~0.0000

Important: rank 3 did not mean Qdrant was relevant; with top-k ranking it was simply the best among near-zero candidates.

#### 4. MMR -> CrossEncoder
Initial narrow funnel left only 3 unique parents, so CrossEncoder could not replace weak Qdrant.
Widened experiment used roughly:
- Qdrant fetch 15 children
- MMR select 8 children
- up to 6 unique parents
- CrossEncoder rerank 6

Observed six CrossEncoder ranks:
1. Parent 2 HNSW overview — 0.9994
2. Parent 3 HNSW search — 0.0032
3. Parent 1 Qdrant — ~0
4. Parent 0 Vector Databases — ~0
5. Parent 4 M Parameter — ~0
6. Parent 6 Metadata — ~0

Conclusion: MMR can diversify into weak candidates; CrossEncoder can suppress them, but a forced final top-k can still include near-zero candidates.

### CrossEncoder -> Parent MMR Experiments
Two versions were tested.

#### Failed version: raw CrossEncoder score inside MMR formula
Formula effectively mixed:
`lambda * CrossEncoder_score - (1-lambda) * cosine_redundancy`

Observed failure:
1. HNSW overview stayed first
2. Metadata jumped to second despite CrossEncoder relevance ~0
3. Qdrant became third

Root cause: raw CrossEncoder score distribution and cosine similarity live on incompatible scales. A relevant parent with CE=0.0032 could be overwhelmed by a moderate redundancy penalty, while an irrelevant but very different parent with CE~0 could win.

Rule established: **do not directly combine raw CrossEncoder sigmoid scores with dense cosine redundancy in one MMR equation without a principled calibration/fusion scheme.**

#### Successful version: CrossEncoder pruning -> dense Parent MMR
Final architecture tested:
Query -> Qdrant child retrieval -> Parent expansion -> CrossEncoder relevance ranking/pruning -> keep top-N parents -> dense Parent-level MMR -> final context

CrossEncoder is used as a relevance gate/ranker only. MMR then uses the same dense embedding space for both terms:
- relevance = cosine(query_embedding, parent_embedding)
- redundancy = cosine(parent_embedding, selected_parent_embedding)

Parameters used:
- child_top_k=15
- cross_encoder_top_k=5
- final_top_k=3
- lambda=0.7

Observed final result:
1. Parent 2 — HNSW overview
   - CE rank 1, CE score 0.9994
   - dense query similarity 0.7807
   - final Parent MMR 0.7807
2. Parent 3 — HNSW search process
   - CE rank 2, CE score 0.0032
   - dense query similarity 0.5358
   - redundancy 0.4203
   - relevance contribution 0.3751
   - penalty 0.1261
   - Parent MMR 0.2490
3. Parent 4 — HNSW M Parameter
   - CE rank 5, CE score ~0
   - dense query similarity 0.3500
   - redundancy 0.3391
   - Parent MMR 0.1433

This was materially better than the raw-score MMR version because the useful HNSW search-process parent survived and Metadata/Qdrant did not re-enter the final top 3.

### MMR Final Mental Model
- Similarity retrieval: semantic relevance
- CrossEncoder: stronger query-document relevance refinement
- MMR: relevance + non-redundancy/diversity
- MMR before reranking can reduce duplicate candidates but may discard/promote candidates too aggressively.
- CrossEncoder before MMR can be cleaner for relevance-first selection, but raw CrossEncoder scores should not be numerically mixed with cosine similarity without calibration.
- Proven working design in this project: **CrossEncoder pruning -> dense Parent MMR**.
- Candidate breadth matters at every stage. A later reranker cannot recover a document discarded earlier.
- top_k is a maximum count, not a quality guarantee; future production refinement may need relevance acceptance/abstention, but do not set a fixed CE threshold yet because query phrasing strongly changes score magnitude.

### Current Canonical Retrieval Architecture / Experimental Modules
Baseline production retrieval remains:
Query -> EmbeddingService -> Qdrant ChildChunk search -> child threshold -> group/dedup by parent_id -> MongoDB ParentChunk expansion -> parent ranking/top-k.

Advanced experimental retrieval now includes:
- CrossEncoder parent reranking
- child-level MMR
- MMR -> CrossEncoder
- CrossEncoder -> raw-score Parent MMR (documented failure mode)
- CrossEncoder pruning -> dense Parent MMR (preferred result from current experiments)

### Pending Production Hardening — Still Open
1. **Differential Qdrant child synchronization** remains intentionally pending. Current `sync_children()` is coarse-grained delete-all-for-document + insert-current. Future target: stable ChildChunk identity with unchanged/new/stale differential reconciliation.
2. **Reranker acceptance/abstention policy** is not finalized. Do not invent a fixed CrossEncoder threshold yet; first evaluate across multiple representative queries because score scale is query-sensitive.
3. The latest Library project ZIP predates some of the new CrossEncoder/MMR local changes. If exact code is needed in a future chat, the learner's current local files are authoritative; ask for/upload them rather than reconstructing stale code.

### Exact Next Step
Start a fresh chat with:
**Hybrid Search — LangChain-first, integrated into the existing Qdrant/Parent-Child project.**

Teaching/implementation expectations:
- Explain dense vs sparse/keyword retrieval and why hybrid search helps.
- Prefer current LangChain abstractions for production code and use direct Python only where framework support is missing or custom behavior is required.
- Integrate with the existing project rather than starting a disconnected demo.
- Preserve OOP/Pydantic architecture where custom code is needed.
- Run concrete retrieval comparisons/experiments, not theory alone.
- Keep advanced retrieval sequence after Hybrid Search aligned with roadmap: Query Rewriting -> Multi-Query Retrieval -> HyDE (unless a technical dependency warrants a small adjustment).

### Checkpoint Status
- CrossEncoder lesson: COMPLETE
- MMR lesson: COMPLETE
- Framework-first coding policy: RECORDED
- Next topic: HYBRID SEARCH
- Checkpoint updated for move to a new chat window.

---

## Daily Checkpoint — 2026-09-13 (Session End)

### Session Scope
This session completed the Hybrid Search lesson, integrated Hybrid retrieval with the existing Parent/Child -> CrossEncoder -> Parent MMR architecture, ran retrieval-funnel experiments, created a moderate benchmark corpus, and successfully batch-ingested that corpus into MongoDB + the hybrid Qdrant collection.

### Hybrid Search — COMPLETE
Hybrid retrieval was implemented using the current native Qdrant named-vector/fusion APIs so the mechanics remain explicit.

Current Hybrid Search components:
- `app/embeddings/sparse_embedding_service.py`
- `app/vectorstore/hybrid_qdrant_store.py`
- `app/retrieval/hybrid_child_retriever.py`
- `app/retrieval/hybrid_parent_child_retriever.py`
- `app/retrieval/hybrid_rerank_mmr_retriever.py`
- `scripts/test_hybrid_search.py`
- `scripts/test_hybrid_rerank_mmr.py`
- `scripts/compare_dense_hybrid_funnel.py`

Current hybrid collection:
- `structure_aware_v3_hybrid`

Dense representation:
- model: `sentence-transformers/all-MiniLM-L6-v2`
- dimension: 384
- normalized embeddings
- Qdrant named vector: `dense`

Sparse representation:
- model: `Qdrant/bm25` via FastEmbed
- Qdrant named sparse vector: `sparse`
- sparse vector configured with IDF modifier

Hybrid fusion:
- native Qdrant Reciprocal Rank Fusion (RRF)
- dense and sparse candidate lists are prefetched separately
- RRF combines ranks, NOT raw dense cosine + BM25 magnitudes
- current `HybridSearchConfig` uses `rrf_k=2`
- current experimental prefetch defaults: dense 15, sparse 15

### Hybrid Search Concepts Proven
1. Dense retrieval provides semantic/paraphrase matching.
2. Sparse/BM25 retrieval provides strong lexical/exact-token matching.
3. Dense cosine scores and BM25 scores are on different scales and must not simply be added together.
4. RRF combines rank positions and is therefore robust to score-scale mismatch.
5. Candidate breadth (`dense_prefetch_k`, `sparse_prefetch_k`) matters because fusion cannot recover candidates discarded before fusion.
6. `final_k` controls fused output count, not prefetch breadth.
7. Sparse feature indices are lexical feature IDs, not positions in the 384-dimensional dense embedding.
8. One Qdrant ChildChunk point stores BOTH named vectors (`dense` and `sparse`) with one shared payload.

### Hybrid + CrossEncoder + Parent MMR Integration — COMPLETE
Canonical advanced retrieval flow now proven end-to-end:

`Query -> Dense + Sparse query vectors -> Qdrant RRF Hybrid child retrieval -> group/deduplicate by parent_id -> MongoDB Parent expansion -> CrossEncoder parent reranking/pruning -> dense Parent MMR -> final parent context`

Important design rule:
- CrossEncoder is applied to **parents**, not children, in the preferred current architecture.
- CrossEncoder receives `ParentRetrievalResult` and scores full `ParentChunk.content`.
- Parent MMR uses dense cosine for BOTH query relevance and parent redundancy.
- CrossEncoder scores are retained as trace metadata only inside Parent MMR.
- Never numerically mix RRF score, CrossEncoder score, and dense cosine as though they share a common scale.

Three distinct scoring spaces:
1. RRF score -> candidate discovery/fusion
2. CrossEncoder score -> parent relevance refinement/pruning
3. Dense cosine/MMR -> relevance + non-redundancy selection

### Hybrid -> CrossEncoder -> Parent MMR Experiment
Query:
`How does HNSW search efficiently?`

Final result:
1. Parent 2 — HNSW overview
   - CrossEncoder rank 1
   - CE score 0.999360
   - dense query similarity 0.780668
   - MMR 0.780668
2. Parent 3 — HNSW search process
   - CrossEncoder rank 2
   - CE score 0.003236
   - dense query similarity 0.535800
   - max redundancy 0.420303
   - MMR 0.248969
3. Parent 4 — M parameter
   - CrossEncoder rank 5
   - CE score 0.000022
   - dense query similarity 0.349969
   - max redundancy 0.339068
   - MMR 0.143258

Conclusion:
- Final parents were effectively the same as the earlier Dense -> CrossEncoder -> Parent MMR experiment.
- Hybrid did not materially change the final top-3 for this small corpus/query because Dense had already retrieved the useful evidence and downstream CE/MMR converged on the same parents.
- This was considered correct behavior, not a failure.

### Dense vs Hybrid Retrieval-Funnel Experiment
A dedicated funnel comparison was run for:
- `ef_construct`
- `How does HNSW search efficiently?`

The comparison printed:
- Dense child candidates
- Hybrid child candidates
- Dense unique parents
- Hybrid unique parents
- parent-set overlap

With the original tiny corpus (19 children / 7 parents) and `top_k=15`:
- Dense unique parents = 7
- Hybrid unique parents = 7
- Common parents = 7
- Dense-only = 0
- Hybrid-only = 0

Key lesson:
- `top_k=15` over only 19 children means ~79% of the corpus is exposed.
- That tiny corpus cannot meaningfully measure candidate recall differences.
- Hybrid still changed ranking order and concentrated the top HNSW results, but the corpus was too small to evaluate retrieval quality seriously.

For `How does HNSW search efficiently?`, Hybrid concentrated the top four children around the two useful HNSW parents, while also changing lower parent ordering. Metadata was pushed lower compared with Dense. This confirmed that Hybrid changes candidate ranking even when the broad parent set is unchanged.

### Decision: Move to a Moderate Benchmark Corpus BEFORE Query Rewriting
The learner explicitly questioned whether experiments on 19 children were too small. Decision agreed:
- pause Query Rewriting temporarily
- create a moderate technical corpus
- benchmark Dense / Sparse / Hybrid / reranked pipelines on the same gold query set
- only then proceed to Query Rewriting, Multi-Query, HyDE so each later technique can be measured against the same baseline

### Moderate Benchmark Corpus — CREATED
A controlled synthetic-but-realistic technical corpus was generated by ChatGPT.

Important provenance:
- It was **NOT scraped from Qdrant or other websites**.
- It is original benchmark material written specifically for this mentorship.
- Future web/HTML scraping ingestion is desired by the learner but explicitly deferred to a later lesson.

Corpus package:
- `agentic_rag_moderate_corpus.zip`

Corpus contents:
- 24 Markdown documents (`DOC-001` through `DOC-024`)
- `gold_queries.json`
- `manifest.json`
- README

Approximate generated corpus statistics before ingestion:
- 24 documents
- 32 gold queries
- ~53,539 characters
- ~7,045 words

Topics intentionally include overlapping and distractor material across:
- HNSW architecture
- `M`
- `ef_search`
- `ef_construct`
- Dense retrieval
- Sparse/BM25 retrieval
- Hybrid/RRF
- Parent/Child retrieval
- CrossEncoder reranking
- MMR
- chunking strategies
- metadata/filtering
- Qdrant named vectors
- Qdrant synchronization/idempotency
- MongoDB parent repository
- embeddings/normalization
- Query Rewriting
- Multi-Query retrieval
- HyDE
- RAG evaluation
- FastAPI HTTP 422
- Docker/local infrastructure / `ERR_CONNECTION_REFUSED`
- retrieval observability
- tenant isolation
- vector-search failure modes
- benchmark design

### Benchmark Corpus Local Layout
Recommended/current layout:

`data/documents/`
- 24 `DOC-*.md` source files

`data/benchmark/`
- `gold_queries.json`
- `manifest.json`

This separation is intentional:
- `data/documents/` = ingestion sources
- `data/benchmark/` = evaluation labels/metadata

### Batch Markdown Corpus Ingestion — IMPLEMENTED AND RUN SUCCESSFULLY
New corpus-ingestion support:
- `app/ingestion/markdown_corpus_loader.py`
- `scripts/ingest_benchmark_corpus.py`

`MarkdownCorpusLoader` behavior:
- loads all `DOC-*.md`
- derives stable `document_id` from filename prefix (`DOC-001`, etc.)
- keeps filename as `source`
- validates non-empty documents
- rejects duplicate document IDs

Reason for stable `DOC-xxx` document IDs:
- `gold_queries.json` labels relevance using the same document IDs
- retrieval results can therefore be scored directly against the gold set later

Batch ingestion rule:
- recreate the hybrid Qdrant collection ONCE, on the first document only
- ingest all later documents into the same collection without recreating it
- this prevents each iteration from deleting the previously ingested corpus

Per-document ingestion sequence remains:
1. Structure detection
2. Structure-aware parent generation
3. MongoDB parent synchronization
4. read authoritative persisted parents
5. generate child chunks from persisted parents
6. validate child -> parent references
7. dense child embeddings
8. sparse child embeddings
9. insert hybrid child points into Qdrant

### Moderate Corpus Ingestion — ACTUAL RESULTS
Final ingestion summary supplied by learner:
- Documents discovered: 24
- Documents succeeded: 24
- Documents failed: 0
- Total benchmark parents persisted/read: 144
- Total hybrid children inserted: 316
- Dense dimension: 384
- Total elapsed: 39.38 seconds
- Hybrid collection: `structure_aware_v3_hybrid`

Typical document result:
- generated parents: 9
- persisted parents after synchronization: 6
- children: ~11–14

Observed repeated pattern:
- `Generated parents: 9`
- `Incoming parents: 9`
- `New parents inserted: 6`
- `Parents after sync: 6`
- `Persisted parents read: 6`

Likely explanation to retain for future investigation:
- the generated benchmark files intentionally contain repeated generic/distractor text
- Mongo parent uniqueness/content-hash reconciliation appears to collapse repeated logical parent content
- treat this as a strong hypothesis, not yet a fully proven root cause unless inspected directly

Important integrity result:
- children are generated from the authoritative persisted parents
- every ingestion showed `Parent references valid: True`

### Current MongoDB Counts
Learner verified MongoDB parent collection total:
- 151 total parent documents
- 144 from the new benchmark ingestion
- 7 legacy parents from the earlier demo corpus

Interpretation:
- `144 + 7 = 151`, so counts are internally consistent.
- The 7 legacy Mongo parents do not contaminate benchmark retrieval as long as current Qdrant hybrid collection contains only the newly rebuilt benchmark children; parent expansion begins from parent IDs returned by Qdrant.
- For a future fully isolated benchmark, a separate Mongo benchmark collection or explicit cleanup may be cleaner, but this is not required before the next retrieval benchmark.

### Current Qdrant Benchmark State
Expected/current hybrid benchmark state after controlled rebuild:
- collection: `structure_aware_v3_hybrid`
- 316 benchmark ChildChunk points
- each point contains named `dense` + `sparse` vectors
- payload includes document identity and parent/child linkage

Before trusting benchmark metrics, the next evaluator should sanity-check:
1. Qdrant point count == 316
2. retrieved payload exposes `document_id` values like `DOC-001` ... `DOC-024`
3. abort benchmark if either sanity check fails

### Corpus Scale Lesson
Old corpus:
- 19 children
- top_k=15 exposed ~79% of corpus

New benchmark corpus:
- 316 children
- top_k=5 ~= 1.6%
- top_k=10 ~= 3.2%
- top_k=15 ~= 4.7%
- top_k=20 ~= 6.3%

This is now large enough for candidate ranking/recall experiments to become materially useful.

Current parent/child ratio:
- 316 children / 144 persisted benchmark parents ~= 2.19 children per parent

### Benchmark Metrics Agreed for Next Step
First automated benchmark stage should compare:
1. Dense
2. Sparse
3. Hybrid

against the SAME `gold_queries.json` set.

Initial metrics:
- Hit Rate@5
- Hit Rate@10
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)

Then extend the same benchmark to:
4. Hybrid -> CrossEncoder
5. Hybrid -> CrossEncoder -> Parent MMR

Also retain/consider:
- candidate parent recall
- final context precision
- latency per stage

Important evaluation distinction:
- child-level retrieval metrics and document/parent-level metrics are not interchangeable
- gold labels currently identify relevant `DOC-xxx` documents
- evaluator should clearly state which retrieval unit is being scored

### Deferred Web / HTML Ingestion Requirement
Learner explicitly wants a later production lesson for:
`URL -> HTML fetch -> parse/boilerplate removal -> headings/paragraphs/code -> Document -> Structure-Aware Parent/Child -> metadata -> Mongo parents -> Dense+Sparse -> Hybrid Qdrant`

Future coverage should include:
- static HTML (`httpx`/parser style)
- JS-rendered pages/browser extraction where needed
- source URL metadata
- page title
- section hierarchy
- fetched timestamp
- canonical URLs
- sitemap ingestion
- duplicate URLs/content hashes
- incremental recrawling
- stale page deletion
- robots/site policies
- rate limiting

This is intentionally deferred; do not interrupt the current retrieval benchmark work for it.

### Pending Production Hardening — STILL OPEN
1. Differential Qdrant child synchronization remains pending.
   - Current benchmark run intentionally used controlled collection recreation.
   - Future target: preserve unchanged child vectors, insert new/changed only, delete stale only.
2. CrossEncoder acceptance/abstention threshold remains unresolved.
   - Do not invent a fixed threshold because score magnitude is query-phrasing-sensitive.
3. Benchmark Mongo isolation is optional future cleanup.
   - current 7 legacy parents are known and accounted for.
4. Benchmark corpus contains repeated generic distractor boilerplate.
   - this may influence Dense and BM25 statistics; current corpus is still acceptable for immediate benchmarking, but future corpus versions may reduce repeated boilerplate.

### Current Canonical Advanced Retrieval Architecture
`Query`
` -> Dense query embedding + Sparse BM25 query vector`
` -> Qdrant named-vector Dense/Sparse prefetch`
` -> RRF Hybrid child fusion`
` -> Child candidates`
` -> group/deduplicate by parent_id`
` -> MongoDB authoritative ParentChunk expansion`
` -> CrossEncoder parent reranking/pruning`
` -> dense Parent MMR`
` -> final parent context`
` -> LLM`

### Exact Next Step — START HERE TOMORROW
Implement the **automated retrieval benchmark evaluator** over the current 24-document / 316-child benchmark corpus.

First evaluator milestone:
1. load `data/benchmark/gold_queries.json`
2. verify Qdrant benchmark collection count is 316
3. verify payload `document_id` uses `DOC-xxx`
4. run every answerable gold query through Dense, Sparse, and Hybrid retrieval
5. convert child hits into ranked document IDs (deduplicate documents while preserving first rank)
6. compute and print:
   - Hit@5
   - Hit@10
   - Recall@5
   - Recall@10
   - MRR
7. retain per-query details so failures can be inspected
8. handle the unanswerable query separately; do not force it into ordinary recall/MRR calculations unless defining an explicit abstention metric

After Dense/Sparse/Hybrid baseline metrics are established:
- benchmark Hybrid -> CrossEncoder
- benchmark Hybrid -> CrossEncoder -> Parent MMR
- compare quality AND latency

Only after this benchmark baseline is stable, resume the advanced retrieval roadmap:
**Query Rewriting -> Multi-Query Retrieval -> HyDE**.

### Session-End Status
- Parent/Child production architecture: COMPLETE for current lesson scope
- CrossEncoder reranking: COMPLETE
- MMR: COMPLETE
- Hybrid Search: COMPLETE
- Hybrid -> CrossEncoder -> Parent MMR integration: COMPLETE
- Moderate corpus generation: COMPLETE
- 24-document batch Markdown ingestion: COMPLETE
- Benchmark corpus current size: 144 benchmark parents / 316 hybrid children
- MongoDB current total: 151 parents (144 benchmark + 7 legacy)
- Automated retrieval benchmark evaluator: NEXT
- Query Rewriting: intentionally PAUSED until baseline benchmark is measured
- Web/HTML scraping ingestion: DEFERRED future production lesson

### Continuity Instruction for New Chat
When learner returns and says to continue, do NOT restart CrossEncoder, MMR, Hybrid Search, or corpus ingestion. Begin directly with the automated Dense vs Sparse vs Hybrid benchmark evaluator using the existing 24-document corpus and `gold_queries.json`.

---

## Daily Checkpoint — 2026-09-16 (Query Rewriting + Multi-Query Retrieval)

### Session Scope
This checkpoint records completion of the Query Rewriting lesson and the Multi-Query Retrieval lesson, including end-to-end integration with the existing Parent/Child Hybrid -> CrossEncoder -> Parent MMR architecture, automated paired benchmarks, observed latency/quality trade-offs, and newly identified benchmark-design limitations.

### Baseline / Advanced Retrieval Benchmarks — COMPLETE
Current benchmark corpus remains:
- 24 synthetic-but-realistic Markdown documents (`DOC-001` ... `DOC-024`)
- 144 persisted benchmark ParentChunks in MongoDB
- 316 hybrid ChildChunk points in Qdrant
- 32 gold queries total
- 31 answerable queries + 1 deliberately unanswerable GPU/video-rendering query

Previously established final-funnel benchmark (`Hybrid parent -> CrossEncoder -> Parent MMR`):
- Hybrid parent: Hit@3=1.000, Recall@3=0.952, Precision@3=0.559, MRR=0.919
- Hybrid -> CrossEncoder: Hit@3=1.000, Recall@3=0.935, Precision@3=0.591, MRR=0.914
- Hybrid -> CrossEncoder -> Parent MMR: Hit@3=1.000, Recall@3=0.925, Precision@3=0.613, MRR=0.952

### Query Rewriting — COMPLETE
Implemented:
- `app/querying/query_rewriter.py`
- LangChain `ChatPromptTemplate`
- provider-agnostic `BaseChatModel`
- structured Pydantic output
- fail-open behavior (original query used if rewrite fails)
- explicit preservation of identifiers / acronyms / error codes / API names / parameter names

Component behavior validated:
- vague queries can be clarified
- already retrieval-ready queries may remain unchanged
- rewriting must remain a search query and must not answer the user

Paired benchmark compared:
`Original Query -> Hybrid -> CrossEncoder -> Parent MMR`
vs
`Rewritten Query -> Hybrid -> CrossEncoder -> Parent MMR`

Query Rewriting benchmark result (31 answerable queries):
- rewrites changed: 21
- unchanged: 10
- fallbacks: 0
- wins: 1
- ties: 27
- losses: 3
- Hit@3: 1.000 -> 1.000
- Recall@3: 0.925 -> 0.925
- Precision@3: 0.613 -> 0.602
- MRR: 0.952 -> 0.930
- average rewrite latency: ~1076 ms
- average original retrieval: ~2861 ms
- average rewritten retrieval: ~2834 ms
- average rewritten total: ~3911 ms

Conclusion:
- Always-on query rewriting is NOT justified on the current corpus.
- Conditional rewriting is preferred for vague, ambiguous, context-dependent, or poorly formed queries.
- A grammatically improved query is not necessarily a better retrieval query.
- Rewriting can dilute distinctive lexical signals (e.g., exact identifiers / technical terms).
- Production direction: clear/specific query -> retrieve directly; ambiguous/context-dependent query -> rewrite first.

### Multi-Query Retrieval — COMPLETE
Implemented:
- `app/querying/multi_query_generator.py`
- `app/retrieval/multi_query_hybrid_retriever.py`
- `scripts/test_multi_query_retrieval.py`
- automated Multi-Query benchmark evaluator/runner

Canonical Multi-Query architecture:
`Original Query`
` -> generate several alternative retrieval queries`
` -> KEEP original query as an anchor`
` -> independently run Hybrid CHILD retrieval for original + alternatives`
` -> first-level RRF inside each query = Dense + Sparse Hybrid fusion`
` -> second-level RRF across query-specific Hybrid ranked lists`
` -> deduplicate / fuse children`
` -> group by parent_id`
` -> MongoDB authoritative ParentChunk expansion ONCE`
` -> CrossEncoder using ORIGINAL user query ONCE`
` -> Parent MMR using ORIGINAL user query ONCE`
` -> final parent context`

Important score semantics:
1. Dense cosine score = semantic similarity.
2. Sparse/BM25 score/rank = lexical retrieval evidence.
3. First-level Hybrid score = Qdrant RRF-derived fusion score for Dense+Sparse within ONE query.
4. Second-level Multi-Query RRF score = fusion of rank positions across original + generated query result lists.
5. CrossEncoder score = query-parent relevance signal; not assumed to be calibrated probability.
6. MMR score = dense relevance/diversity objective.
Do not numerically mix these score families as though they have a common scale.

Multi-Query RRF clarification:
- Raw Hybrid scores from different queries are NOT added.
- Example: hybrid scores 0.83, 0.61, 0.72 are ignored by second-level RRF magnitude-wise.
- If the child ranks 1, 4, 2, second-level RRF uses `1/(k+1) + 1/(k+4) + 1/(k+2)`.
- Current first-level Qdrant Hybrid experimental `rrf_k=2`.
- Current second-level Multi-Query experimental `multi_query_rrf_k=60`.
- `k` is a tunable smoothing/rank-decay parameter, not a universal RRF constant.
- Small k emphasizes top rank strongly; large k flattens differences and rewards repeated cross-list occurrence more evenly.

Multi-Query benchmark configuration:
- baseline: original query -> Hybrid child top_k=30 -> Parent -> CE top_k=10 -> MMR top_k=3
- Multi-Query: original + 3 generated alternatives
- each query: Hybrid child top_k=15
- second-level cross-query RRF
- fused child pool pruned to top 30
- Parent expansion once
- CE top_k=10 using original query
- MMR top_k=3 using original query

Multi-Query benchmark result (31 answerable queries):
- Candidate Recall: 1.000 -> 1.000
- Candidate Hit: 1.000 -> 1.000
- Hit@3: 1.000 -> 1.000
- Recall@3: 0.925 -> 0.925
- Precision@3: 0.613 -> 0.591
- MRR: 0.952 -> 0.952
- Avg query-generation latency: ~1083 ms
- Avg baseline retrieval latency: ~2358 ms
- Avg Multi-Query total latency: ~4212 ms

Conclusion:
- Multi-Query broadened candidate coverage but did not improve candidate recall because baseline candidate recall was already saturated at 1.0 on this corpus.
- It added candidate noise, slightly reduced final Precision@3, and materially increased latency.
- Always-on Multi-Query is NOT justified on the current benchmark.
- Multi-Query should be treated as a selective recall-expansion technique for hard/vague/multi-aspect/vocabulary-mismatch queries or when initial candidate retrieval is weak.
- Do NOT spend roadmap time tuning generated-query count or second-level RRF k on this saturated corpus yet.

### Critical Benchmark Interpretation — NEW, MUST PRESERVE
The learner correctly identified that the current benchmark is NOT realistic enough to support broad production-quality claims.

Current benchmark weaknesses:
1. Repeated / redundant generic distractor paragraph(s) occur across many synthetic documents.
   - This adds artificial lexical and semantic overlap.
   - It can distort Dense retrieval, BM25 statistics, CrossEncoder behavior, and MMR redundancy decisions.
   - It may make irrelevant documents appear repeatedly across query variants.

2. Gold labels are DOCUMENT-level (`DOC-xxx`) while the production retrieval architecture is PARENT/CHILD.
   - One document may contain multiple ParentChunks.
   - A gold label that marks the entire document relevant cannot distinguish the truly answer-bearing parent/section from another parent in the same document.
   - Current Precision@3 can therefore count a parent from a relevant document as relevant even if that specific parent does not answer the query.

3. Parent/Child retrieval quality should ultimately be evaluated at multiple units:
   - child candidate level (optional diagnostic)
   - parent/section level (preferred retrieval-quality gold)
   - document level (broad coverage only)
   - final answer level (groundedness/correctness)

4. The current corpus is small (316 children) and controlled.
   - Candidate Recall=1.0 at child_top_k=30 is evidence that the current pipeline can find relevant DOCUMENT labels on this corpus.
   - It does NOT prove that the embedding model is universally excellent, Qdrant/HNSW parameters are globally optimal, or production retrieval quality will remain perfect at larger scale / harder data.

5. Current benchmark should now be described as:
   - useful for pipeline correctness, architecture validation, regression testing, score-flow understanding, and controlled ablation experiments;
   - NOT sufficient as a realistic production retrieval-quality benchmark.

### Production Evaluation Direction — NEW
For realistic RAG verification, future evaluation should combine:
- a more realistic corpus (real PDFs / HTML / web pages or standard retrieval datasets)
- parent/section-level relevance judgments rather than only document-level labels
- harder ambiguous/paraphrased/multi-hop/no-answer queries
- candidate recall at parent level
- Recall@K / Precision@K / MRR / nDCG as appropriate
- answer correctness / groundedness / citation support
- abstention/no-answer quality
- latency and cost by stage
- regression tests on a fixed golden set
- sampled production-query review
- tracing of query -> child hits -> parent expansion -> CE -> MMR -> final context -> answer
- human review for important failures
- optional LLM-as-judge / RAGAS / DeepEval only as secondary scalable evaluators, not unquestioned ground truth
- monitoring for corpus/model/query-distribution drift

### Candidate Recall = 1.0 — Exact Interpretation
Safe conclusion:
- For the current 31 answerable gold queries, current corpus, current candidate budget, current Dense+Sparse Hybrid retriever, and document-level gold labels, at least one relevant gold document was represented in the candidate pool for every query.

What it supports:
- retrieval plumbing is functioning end-to-end for this benchmark
- dense/sparse embeddings + Qdrant Hybrid fusion are adequate to recover the current gold documents under current settings
- parent/child IDs/payloads and candidate extraction are behaving consistently enough for the benchmark

What it does NOT prove:
- the embedding model is universally optimal
- Qdrant HNSW parameters are excellent/optimal
- all answer-bearing parents are retrieved
- retrieval will remain perfect at larger corpus size
- retrieval is robust to realistic noise/domain shift
- ANN recall/latency is optimal

Qdrant/HNSW parameter quality must be evaluated separately with scale, exact-vs-ANN recall, latency, memory/build-time trade-offs, and parameter sweeps (`m`, `ef_construct`, `ef_search`, prefetch/final candidate sizes).

### Updated Roadmap Position
Completed advanced retrieval topics:
- CrossEncoder reranking: COMPLETE
- Parent MMR: COMPLETE
- Hybrid Dense+Sparse/RRF: COMPLETE
- Hybrid -> CrossEncoder -> Parent MMR: COMPLETE
- baseline retrieval benchmarking: COMPLETE
- advanced funnel benchmarking: COMPLETE
- Query Rewriting: COMPLETE + BENCHMARKED
- Multi-Query Retrieval: COMPLETE + BENCHMARKED

Next planned advanced retrieval lesson:
- **HyDE (Hypothetical Document Embeddings)**

However, preserve the learner's benchmark concern:
- Do not over-interpret HyDE results on the current synthetic corpus.
- Use current corpus mainly for implementation/mechanics/regression.
- Plan a later realistic retrieval benchmark using real PDFs / HTML / web pages and parent/section-level gold labels before claiming production retrieval quality.

### Pending Production Hardening — STILL OPEN
1. Differential Qdrant child synchronization.
2. CrossEncoder acceptance/abstention strategy (avoid arbitrary global fixed score threshold).
3. More realistic corpus / benchmark redesign with parent/section-level gold relevance.
4. Real PDF/HTML/web ingestion pipeline and hardened retrieval stress tests.
5. Production observability/evaluation harness for tracing and sampled live-query review.

### Continuity Instruction for Next Session
When learner returns, do NOT restart Query Rewriting or Multi-Query. Both are implemented and benchmarked. Start from the decision point:
- discuss/implement HyDE next, while explicitly treating current synthetic benchmark as a controlled engineering benchmark rather than realistic production-quality evidence;
- later replace/augment the corpus with realistic PDFs/HTML/web content and parent/section-level judgments for a stronger retrieval benchmark.

### Future Realistic RAG Benchmark Requirement — EXPLICIT REMINDER
The current 24-document synthetic benchmark remains useful for pipeline correctness, regression testing, and controlled retrieval experiments, but it must NOT be treated as the final realistic production-quality benchmark.

When the mentorship reaches the realistic ingestion / production evaluation phase, build a substantially more realistic benchmark with the following properties:

1. Use real source material rather than repeated synthetic boilerplate:
   - real PDFs
   - HTML / web documentation
   - technical manuals / product documentation
   - scientific or engineering documents where appropriate

2. Preserve natural document structure during ingestion:
   - headings and section hierarchy
   - paragraphs
   - lists
   - tables
   - code blocks where present
   - page / source metadata

3. Continue using the production Parent/Child architecture:
   `real source -> structure-aware parsing -> ParentChunks -> ChildChunks -> MongoDB parents -> Dense + Sparse children -> Qdrant Hybrid retrieval`

4. Gold relevance labels must match the retrieval unit being evaluated. The preferred gold schema for Parent/Child RAG is parent/section-level rather than document-only. Example:
   - `document_id`
   - `parent_id`
   - optional `section_path`
   - optional page number / answer span
   - answer text or supporting evidence where available

5. Avoid the current evaluation mismatch where a document may contain multiple parents but any parent from the gold document is counted as relevant. A parent should only count as relevant when that specific parent/section contains evidence needed for the query.

6. Include realistic difficult query classes:
   - easy lexical / exact-identifier queries
   - semantic paraphrases
   - vocabulary mismatch
   - ambiguous or conversational queries
   - multi-section and multi-document questions
   - near-duplicate / hard-negative sections
   - error codes, APIs, class names, parameter names
   - unanswerable queries
   - later multimodal text/table/image queries when multimodal RAG is covered

7. Include realistic hard negatives rather than repeated generic distractor paragraphs. The current corpus contains repeated boilerplate that can artificially influence Dense, BM25, CrossEncoder, and MMR behavior.

8. Evaluate separately at appropriate stages:
   - child candidate recall
   - parent/section candidate recall
   - final Parent Recall@K / Precision@K / MRR / nDCG
   - answer correctness / groundedness / citation correctness
   - abstention quality
   - stage latency and cost

9. Public benchmark datasets should be considered as seed evaluation resources, especially datasets with explicit qrels or section/answer-span supervision. Candidate resources to evaluate include BEIR subsets, MS MARCO passage ranking, TechQA, Natural Questions, and Vectara Open RAGBench. Adapt them to the local Parent/Child schema rather than assuming document-level qrels are sufficient.

10. Before making strong claims about production retrieval quality or tuning Query Rewriting / Multi-Query / HyDE, first run the architecture against this realistic benchmark. The current synthetic benchmark is primarily an engineering/regression benchmark.

This requirement is intentionally deferred until the realistic ingestion / production evaluation phase, but must remain visible in future checkpoints and must not be silently dropped.


## Daily Checkpoint — 2026-09-18 (Benchmark V2 Refactor Planned)

### Decision
Pause HyDE Hybrid integration and refactor the benchmark foundation before further retrieval comparisons. The current benchmark remains useful for mechanics/regression but is not the long-term quality benchmark because it contains repeated boilerplate and uses document-level gold labels while the production retrieval architecture is Parent/Child.

### Canonical Retrieval Architecture (unchanged)
`Query -> Dense+Sparse child retrieval -> RRF -> Child candidates -> Parent expansion/dedup -> CrossEncoder on parents -> Parent MMR -> Final context`

### Benchmark V2 Goals
1. Clean the 24-document corpus and remove accidental duplicate/redundant paragraphs and repeated generic boilerplate.
2. Normalize heading schema so heading hierarchy has one source of truth rather than duplicated h1..h6 data in both `heading_context` and `metadata`.
3. Introduce stable benchmark identities that survive re-ingestion. Do not rely on random UUID parent IDs as the only gold identity.
4. Redesign gold labels to support child-, parent-, and document-level relevance. Parent/section relevance becomes the primary retrieval-quality unit; child relevance is evaluated explicitly for retrieval precision; document relevance remains a broad secondary diagnostic.
5. Centralize evaluation so all benchmark runners share one scoring contract.
6. Metrics to support: Hit@K, Precision@K, Recall@K, MRR, MAP, nDCG@K, plus latency and per-query diagnostics.
7. Preserve support for unanswerable queries and future multi-parent/multi-document questions.
8. Version/freeze the cleaned benchmark as V2 and avoid ad-hoc edits; later changes should become V3.

### Stable Parent Identity Design
Do NOT use `DOC-004::HNSW::Index Construction::ef_construct` alone as a guaranteed globally unique key because two logical parents can share the same section path. Use a deterministic parent key derived from stable fields, for example:
`document_id + normalized section_path + occurrence/parent_index + content_hash`
or an equivalent deterministic hash. The human-readable section path may remain in the gold record for inspection, but uniqueness must be enforced by the deterministic key.

### New Benchmark Storage Isolation
Keep existing databases/collections untouched so the older branch remains runnable. Benchmark V2 must use new names. Proposed defaults:
- Qdrant collection: `structure_aware_v4_benchmark_v2`
- MongoDB parent collection: `parents_benchmark_v2`
These names can be placed in settings/config so branches do not hard-code them.

### Branch
Suggested branch name: `refactor/benchmark-v2-parent-child-gold`. The slash is intentional and valid Git syntax; it is one branch name, with `refactor/` acting as a namespace-like prefix.

### Planned Refactor Areas
- 24 Markdown benchmark source documents
- `data/benchmark/gold_queries.json`
- `data/benchmark/manifest.json`
- benchmark corpus loader / ingestion scripts
- heading/metadata schema files as required after inspection
- centralized evaluation models and evaluator
- all existing benchmark runners (Dense/Sparse/Hybrid, advanced CE/MMR, Query Rewrite, Multi-Query) so they use the shared V2 gold/evaluator
- HyDE benchmark will be added only after the V2 benchmark passes all existing regressions

### Acceptance Gate Before Resuming HyDE
- no accidental duplicate paragraphs / generic repeated boilerplate
- clean Mongo parent schema
- clean Qdrant child corpus in new V2 collection
- deterministic benchmark identities
- verified child relevance labels
- verified parent relevance labels
- document labels retained only as secondary diagnostics
- shared metrics include Hit, Precision, Recall, MRR, MAP, nDCG
- all current benchmark runners migrated and passing on V2
- then resume HyDE Hybrid evaluation

### Source Inspection Requirement
Before editing, inspect the latest project source ZIP to determine the exact files that need changes rather than guessing. Prefer using the current project archive as the canonical source for this refactor.

## Coding Standards — Locked 2026-09-18

The following conventions are now mandatory for all future mentorship code, refactors, scripts, tests, examples, logging, and documentation unless the learner explicitly requests otherwise:

1. **PEP 8 throughout**
   - Follow PEP 8 naming, spacing, line layout, import organization, and general Python style consistently.
   - Do not introduce ad-hoc formatting changes between lessons.

2. **f-strings only for string interpolation**
   - Use f-strings consistently.
   - Do not switch to `%` formatting or `.format()` in examples, logging calls, or production code.

3. **Logging instead of `print()`**
   - Project code and benchmark/test scripts should use the standard `logging` module rather than `print()`.
   - Application/library modules use `logger = logging.getLogger(__name__)` and do not call `logging.basicConfig()`.
   - Executable boundaries such as scripts, CLIs, and application entry points own logging configuration.

4. **Actual field names must be used consistently in logs and documentation**
   - Use `parent_id`, `parent_key`, `child_id`, `child_key`, `document_id`, etc. exactly as defined in the models.
   - Do not casually rename the same concept in logs or explanations (for example, do not relabel `parent_id` as `parent runtime ID`).
   - If semantic clarification is needed, explain it separately without changing the field name.

5. **No casual renaming**
   - Once a field, class, method, config attribute, or architectural term is established, preserve its name unless there is an explicit refactor decision.
   - Renames must be deliberate, justified, and propagated consistently across code, tests, logging, and documentation.

6. **Consistent imports and formatting**
   - Keep import style and code formatting uniform across files.
   - Prefer clear grouped imports and avoid needless stylistic churn.

7. **Type hints are required**
   - New and modified production code should use Python type hints consistently for parameters, return values, and important local structures where they materially improve clarity.

8. **Pydantic models**
   - Continue using Pydantic for domain/config/data models rather than dataclasses unless the learner explicitly changes this policy.

9. **OOP + SOLID design**
   - Continue using OOP where appropriate.
   - Refactors should follow SOLID principles, especially SRP, dependency inversion, explicit contracts, reusable shared evaluation components, and avoidance of duplicated logic.

10. **Consistency rule**
    - These conventions are now considered part of the canonical mentorship/project architecture and should not change between sessions without an explicit learner-approved decision.

### Current Refactor Status at Lock-In
- Phase I: COMPLETE — domain/schema stabilization, deterministic `parent_key`/`child_key`, heading hierarchy normalization, Benchmark V2 config isolation, model smoke test passed.
- Phase II-A: COMPLETE — Mongo/Qdrant persistence contracts and Qdrant payload projection design; Phase II-A payload validation passed.
- Next: Phase II-B — pipeline, Markdown corpus loader, benchmark ingestion integration, logging migration, and controlled persistence test against the isolated Benchmark V2 stores.


---

## Daily Checkpoint — 2026-09-19 (Benchmark V2 Gold + Benchmark Architecture Refactor)

### Roadmap / Phase
- Main learning track: Agentic RAG / advanced retrieval.
- HyDE lesson remains paused until the benchmark foundation is stable.
- Benchmark V2 refactor progressed through Phase IV-B (gold redesign/curation) and is now in a dedicated folder-architecture refactor branch before Phase IV-C metrics.
- Current structural-refactor branch: `refactor/benchmark-architecture`, branched from `refactor/benchmark-v2-parent-child-gold`.

### Benchmark V2 — Frozen Corpus State
- Benchmark corpus: 24 cleaned Markdown documents.
- Duplicate generic boilerplate removed.
- Corpus validation result:
  - documents=24
  - total_characters=16027
  - approx_words=2149
  - exact_duplicate_paragraph_groups=0
  - `Benchmark corpus validation PASSED.`
- Frozen corpus SHA-256: `16c01b5dfb1753d0b55dafdf6164d977383c0419c4ebfe8b376189aaf7ef9e62`.
- Persisted Benchmark V2 universe:
  - MongoDB collection: `parents_benchmark_v2`
  - Qdrant collection: `structure_aware_v4_benchmark_v2`
  - ParentChunks: 120
  - ChildChunks: 196

### Benchmark V2 — Identity Model
- Runtime/persistence IDs remain UUID-based: `parent_id`, `child_id`.
- Stable deterministic benchmark identities are used for gold labels and reproducibility: `parent_key`, `child_key`.
- Stable key purpose: identify exact retrieval units across re-ingestion without relying on transient UUIDs; content/boundary changes intentionally change the deterministic key and therefore invalidate stale gold judgments instead of silently matching the wrong chunk.

### Gold Dataset Redesign — Phase IV-B
- Legacy `gold_queries.json` was document-level only (`relevant_docs`) and was too coarse for Parent/Child retrieval evaluation.
- Benchmark V2 gold schema now supports:
  - document judgments
  - parent judgments
  - child judgments
  - graded relevance: 3=directly answer-bearing, 2=strongly supporting, 1=weak/supporting, absent=0
  - answerable/unanswerable classification
  - annotation status
- Relevance was curated manually by semantic review of the actual query, ParentChunk text, and ChildChunk text. Python was used only to extract exact stable keys and assemble ready-to-paste JSON; retrieval scores were not used to generate the gold labels.
- Document grade was treated as broad source-level relevance; parent and child grades were judged independently at their own retrieval granularity.
- The current gold workflow is:
  `legacy document gold -> actual Parent/Child unit export -> annotation candidates -> manual relevance curation -> validated V2 gold -> frozen V2 gold`.

### Gold/Benchmark Artifact Roles
- `manifest.json`: benchmark corpus identity/fingerprint/version.
- `benchmark_v2_units.json`: frozen audit snapshot of the real ParentChunk/ChildChunk hierarchy; same corpus represented at parent and child retrieval-unit levels, not two separate corpora.
- `gold_queries_v2_draft.json`: authoring/working copy for V2 gold judgments.
- `gold_annotation_candidates_v2.json`: machine-readable annotation aid showing candidate parents/children from legacy-relevant documents.
- `gold_annotation_candidates_v2.md`: human-readable annotation aid.
- `gold_queries_v2.json`: final authoritative frozen gold dataset after all 32 queries are curated and validation passes.
- Intermediate annotation files are authoring aids and can later be archived. `benchmark_v2_units.json` should be retained as a reproducibility/audit snapshot even if benchmark execution can query persistence directly.

### Gold Curation Status
- Q001-Q005 were manually curated first and validated.
- Q006-Q032 were curated from the uploaded `gold_annotation_candidates_v2.json`; exact `parent_key`/`child_key` values were pulled from the actual frozen annotation workspace.
- User reported validator result with Q001-Q005: `query_count=32`, `curated_query_count=5`, `Benchmark V2 gold validation PASSED.`
- Next validation target after inserting Q006-Q032 is `curated_query_count=32`.
- Once all 32 are curated and validated, freeze as `gold_queries_v2.json` and set top-level status to `frozen`.

### Why Benchmark V2 Exists
- Primary purpose: evaluate retrieval quality at the correct unit granularity.
- Old document-only gold could count the wrong ParentChunk from a relevant document as correct.
- V2 can evaluate whether retrieval found the right child candidate, the right parent context, and the right document, and can support graded ranking metrics such as nDCG.
- A benchmark corpus is expected to be a controlled, representative subset; it does not need to duplicate the entire live production corpus. Production corpus changes do not automatically require re-labeling every production item, but material domain/query/corpus shifts should trigger benchmark revision/versioning.
- Benchmark success alone does not make the retriever fully production-grade; production readiness also requires latency, errors/timeouts/retries, observability, tenant isolation, configuration, integration tests, load tests, etc.

### Benchmark Architecture Refactor Decision
- Benchmark infrastructure should no longer clutter production `app/`.
- New top-level separation:
  - `app/` = production runtime code
  - `benchmarks/` = benchmark data, tools, evaluation code, runners, results
  - `tests/` = proper PyTest tests
  - `scripts/` = only temporary/manual learning or diagnostic scripts that have not yet been classified
- Dependency direction is locked:
  - `benchmarks -> app` allowed
  - `tests -> app` and `tests -> benchmarks` allowed
  - `app -> benchmarks` forbidden
  - `app -> tests` forbidden

### Target Benchmark Structure
```text
benchmarks/
├── __init__.py
├── paths.py
├── data/
│   ├── documents/
│   ├── frozen/
│   │   ├── manifest.json
│   │   ├── benchmark_v2_units.json
│   │   └── gold_queries_v2.json
│   └── work/
│       ├── gold_queries_v1.json
│       ├── gold_queries_v2_draft.json
│       ├── gold_annotation_candidates_v2.json
│       └── gold_annotation_candidates_v2.md
├── evaluation/
├── tools/
├── runners/
└── results/

tests/
├── unit/
├── integration/
└── benchmark/
```

### Central Benchmark Path Module
- New `benchmarks/paths.py` centralizes project/benchmark/data/frozen/work/result paths so moved modules do not keep fragile `Path(__file__).parents[...]` calculations.
- Important path constants include:
  - `PROJECT_ROOT`
  - `BENCHMARK_ROOT`
  - `BENCHMARK_DOCUMENTS_DIR`
  - `BENCHMARK_FROZEN_DIR`
  - `BENCHMARK_WORK_DIR`
  - `BENCHMARK_MANIFEST_PATH`
  - `BENCHMARK_UNITS_PATH`
  - `GOLD_QUERIES_PATH`
  - `GOLD_QUERIES_DRAFT_PATH`
  - annotation candidate paths
  - `BENCHMARK_RESULTS_DIR`
- Legacy V1 gold should remain available during structural migration as `benchmarks/data/work/gold_queries_v1.json` so the old runners can still execute unchanged until Phase IV-C.

### Current Refactor Progress / Validation
- Benchmark data paths were moved under the new `benchmarks/data/...` structure.
- `validate_benchmark_corpus` ran successfully from the new module path and confirmed the 24-document cleaned corpus.
- `validate_benchmark_v2_persistence` connected to MongoDB and confirmed `mongo_parent_count=120` but then failed because Qdrant was temporarily unavailable/disconnected; this was diagnosed as Qdrant connectivity, not a folder-refactor failure.
- Qdrant connection symptoms observed:
  - `[WinError 10061] No connection could be made because the target machine actively refused it`
  - later `Server disconnected without sending a response.`
- Do not suppress Qdrant compatibility warnings as a workaround; restore Qdrant service first and rerun persistence validation.

### Old Runner Compatibility Issues Found During Structural Refactor
1. `run_retrieval_benchmark.py` was pointed at `gold_queries_v2.json`, but the existing `GoldQueryLoader` still expects the V1 list schema. This produced:
   `TypeError: string indices must be integers, not 'str'` at `query=item["query"]`.
   Decision: do not half-migrate the loader during the structural refactor. Keep old benchmark runners on `gold_queries_v1.json` until Phase IV-C intentionally migrates the loader and metric contract to V2.
2. After restoring the V1 gold path, the old baseline runner hit:
   `RuntimeError: Unexpected Qdrant point count. Expected 316, found 196.`
   Diagnosis: stale benchmark assumption. The canonical storage model is 120 parents in MongoDB + 196 children in Qdrant; Qdrant should contain 196 child points, not 316 parent+child points. Search for the stale `316` assumption and update the benchmark validation meaning to expected Qdrant child count. Do not reingest just to satisfy the stale assertion.

### PyTest Decision — Newly Locked Requirement
- Proper PyTest coverage must be part of the project going forward.
- Distinguish:
  - benchmark experiments/runners
  - unit tests
  - integration tests
  - benchmark integrity/regression tests
- Planned structure:
```text
tests/
├── unit/
│   ├── test_child_chunker.py
│   ├── test_metadata.py
│   ├── test_parent_chunker.py
│   ├── test_structure_detector.py
│   └── future metric/model unit tests
├── integration/
│   ├── Mongo/Qdrant/retrieval integration tests
└── benchmark/
    ├── gold dataset integrity
    ├── manifest integrity
    ├── benchmark unit integrity
    └── ranking metric tests
```
- Existing `scripts/test_*.py` files are not automatically PyTest tests; classify each later as unit test, integration test, benchmark experiment, manual demo, or obsolete.

### Coding Standards — Still Locked
- PEP 8.
- f-strings only for formatting.
- `logging`, never `print()`.
- Central `configure_logging()` from `app/utils/logging_config.py`.
- Type hints.
- Pydantic, not dataclasses.
- OOP/SOLID.
- Liberal comments for lifecycle/invariants/non-obvious behavior.
- Use exact schema field names in logs/docs (`parent_id`, `parent_key`, `child_id`, `child_key`).
- No unrelated style/architecture changes without an explicit decision.

### Canonical Retrieval Architecture — Unchanged
`Query -> Dense + Sparse child retrieval -> RRF -> Child candidates -> Parent expansion/dedup -> MongoDB authoritative ParentChunks -> CrossEncoder reranking on parents -> Parent MMR -> final parent context -> LLM`

CrossEncoder remains at parent level after parent expansion. Parent MMR remains after CrossEncoder.

### Current Exact Stop Point
- User is on branch `refactor/benchmark-architecture`.
- Structural refactor is in progress.
- Data/tool/model movement and import/path cleanup are being performed carefully without changing retrieval behavior.
- Baseline runner has surfaced a stale Qdrant point-count assertion (`expected 316`, actual canonical child count 196).
- The next immediate task is to fix the stale benchmark-only point-count expectation (preferably clearly named as expected Qdrant child count), rerun the baseline retrieval benchmark, and continue surfacing/fixing old V1 assumptions without migrating metrics to V2 yet.

### Exact Next Steps
1. Ensure Qdrant is running/stable and rerun `python -m benchmarks.tools.validate_benchmark_v2_persistence`.
2. Fix the stale baseline benchmark assertion that expects 316 Qdrant points; canonical Qdrant child count is 196.
3. Rerun `python -m benchmarks.runners.run_retrieval_benchmark` using legacy V1 gold so behavior remains unchanged during the structural refactor.
4. Complete remaining benchmark module/runners moves and import/path cleanup.
5. Move existing proper tests into `tests/unit/` and run `pytest tests\unit -v`.
6. Commit a stable architecture-refactor checkpoint.
7. Only then begin Phase IV-C: centralized V2 gold loader + shared ranking metrics (Hit@K, Precision@K, Recall@K, MRR, MAP@K, nDCG@K) with PyTest unit tests.
8. After IV-C and migration of all runners, resume HyDE hybrid experiments.

### Continuity Note
Do not confuse the structural refactor with the V2 metric migration. The current branch must preserve old benchmark behavior using V1 gold until Phase IV-C. The new V2 gold, frozen benchmark unit catalog, and manifest are authoritative future benchmark assets, but old runners should not be partially patched into the new schema.

### Checkpoint Status
Checkpoint updated through the current benchmark architecture refactor and current baseline-runner issue.


---

# CHECKPOINT UPDATE — 2026-09-25

## STATUS
**Milestone completed:** Benchmark V2 centralized evaluation framework + baseline Dense/Sparse/Hybrid child and parent evaluation.

**Current branch state:** Benchmark/evaluation milestone is complete and ready for commit.

**Recommended commit message:**

```text
Complete Benchmark V2 child and parent baseline evaluation
```

**Next major phase after commit:** Cloud provider refactor, then resume HyDE.

Recommended next branch:

```text
refactor/cloud-retrieval-providers
```

---

## LOCKED CODING / TESTING STANDARDS

- PEP 8 naming and formatting.
- f-strings only for formatting; avoid `%s` logger placeholders and `.format()`.
- Use `logging`, never `print()`.
- Application/library modules use `logger = logging.getLogger(__name__)`.
- Executable scripts call central `configure_logging()` from `app/utils/logging_config.py`.
- Use type hints.
- Use Pydantic models, not dataclasses.
- Prefer OOP/SOLID and explicit dependency injection.
- Use actual domain field names (`parent_id`, `parent_key`, `child_key`, etc.).
- Add comments liberally in PyTest. Each test should explain:
  - what it proves,
  - why it matters,
  - what regression it protects against.
- Unit tests must mock/fake external boundaries such as Qdrant, MongoDB, LLMs, embeddings, rerankers, network, filesystem/time when appropriate.
- Pure deterministic metric/domain logic should use real objects/functions rather than mocks.
- Use `python -m pytest ...` so the active venv interpreter is guaranteed.

Current test environment:

```text
Python 3.13.9
pytest 9.1.1
Interpreter: C:\code\chat\.venv\Scripts\python.exe
Project root: C:\code\chat\agentic_rag
```

---

# BENCHMARK V2 — FINALIZED FOUNDATION

## Frozen benchmark identity

```text
benchmark_version = 2.0
document_count = 24
gold_query_count = 32
parent_count = 120
child_count = 196
corpus_sha256 = 16c01b5dfb1753d0b55dafdf6164d977383c0419c4ebfe8b376189aaf7ef9e62
```

Frozen assets:

```text
benchmarks/data/frozen/manifest.json
benchmarks/data/frozen/benchmark_v2_units.json
benchmarks/data/frozen/gold_queries_v2.json
```

Gold relevance scale remains:

```text
3 = DIRECT
2 = SUPPORTING
1 = WEAK
0 / absent = IRRELEVANT
```

Binary metrics treat `relevance > 0` as relevant.

nDCG preserves graded relevance values 1/2/3.

Unanswerable queries are retained for per-query diagnostics but excluded from aggregate retrieval-ranking averages. Abstention/refusal quality must be evaluated separately later.

---

# CENTRALIZED EVALUATION ARCHITECTURE — COMPLETE

Current evaluation flow:

```text
GoldDatasetV2
    ↓
GoldQueryV2 relevance maps
    ↓
RankingEvaluator
    ↓
RankingMetrics
    ↓
BenchmarkEvaluator
    ↓
BenchmarkRunResult / typed results
```

## Files implemented / finalized

```text
benchmarks/evaluation/
├── ranking_metrics.py
├── ranking_evaluator.py
├── benchmark_evaluator.py
├── benchmark_models.py
├── gold_models.py
└── annotation_models.py

benchmarks/runners/
├── baseline_retrieval_runner.py
├── baseline_parent_expansion_runner.py
└── run_baseline_retrieval_benchmark_v2.py

tests/unit/
├── test_ranking_metrics.py
├── test_ranking_evaluator.py
├── test_benchmark_evaluator.py
├── test_benchmark_models.py
├── test_gold_models.py
├── test_baseline_retrieval_runner.py
└── test_baseline_parent_expansion_runner.py
```

Benchmark integrity tests remain in:

```text
tests/benchmark/
├── test_benchmark_manifest.py
├── test_benchmark_units.py
└── test_gold_dataset_integrity.py
```

---

# RANKING METRICS — CENTRALIZED

Implemented metrics:

```text
Hit@K
Precision@K
Recall@K
Reciprocal Rank / MRR
Average Precision@K / MAP@K
nDCG@K
```

Important contract:

```text
Hit / Precision / Recall / MRR / MAP
    -> binary relevance: grade > 0

nDCG
    -> graded relevance: 3 / 2 / 1 / 0
```

`RankingEvaluator` truncates ranked IDs to K before calculating reciprocal rank, so the benchmark semantics are effectively MRR@K.

Duplicate ranked IDs are rejected as an upstream retrieval/deduplication defect rather than silently repaired.

---

# GOLDQUERYV2 RELEVANCE CONTRACT — COMPLETE

`GoldQueryV2` exposes centralized relevance maps:

```python
document_relevance_map()
parent_relevance_map()
child_relevance_map()
relevance_map(level)
```

Runners no longer rebuild relevance dictionaries themselves.

This prevents duplicated or inconsistent relevance translation across Dense, Sparse, Hybrid, CE, MMR, Query Rewrite, Multi-Query and HyDE benchmarks.

---

# RUNNER OUTPUT CONTRACT — COMPLETE

Canonical benchmark models include:

```text
RetrievalStrategy
RankedItem
QueryRankingResult
BenchmarkRunResult
```

`RankedItem` carries:

```text
item_id
rank
score (optional)
metadata (optional transport metadata)
```

Stable evaluation identifiers:

```text
document level -> document_id
parent level   -> parent_key
child level    -> child_key
```

For child retrieval, `RankedItem.metadata` now carries downstream expansion metadata such as:

```text
parent_id
parent_key
document_id
```

Metrics consume stable IDs only; metadata is transport/context information for later pipeline stages.

---

# BASELINE CHILD RETRIEVAL RUNNER — COMPLETE

`BaselineRetrievalRunner` supports:

```text
DENSE
SPARSE
HYBRID
```

It:

- executes child retrieval,
- measures per-query latency,
- converts Qdrant hits to stable `child_key` rankings,
- preserves scores/metadata,
- returns `BenchmarkRunResult`,
- performs no metric calculations.

Child-level evaluation is diagnostic candidate-generation evaluation.

---

# BASELINE PARENT EXPANSION RUNNER — COMPLETE

`BaselineParentExpansionRunner` performs:

```text
existing child BenchmarkRunResult
    ↓
read parent_id from RankedItem.metadata
    ↓
deduplicate by parent_id in child ranking order
    ↓
fetch authoritative ParentChunk from MongoDB
    ↓
emit parent_key ranking
    ↓
BenchmarkRunResult at RetrievalLevel.PARENT
```

Important design correction completed on 2026-09-25:

Parent expansion does **not** rerun child retrieval.

The flow is now:

```text
retrieve once
    ↓
evaluate child ranking
    ↓
expand the same exact child result set
    ↓
evaluate parent ranking
```

This guarantees child and parent metrics are derived from the identical retrieval run.

Parent baseline ranking is determined by the strongest / earliest contributing child before CrossEncoder reranking.

Dangling child -> parent references are treated as persistence-integrity defects and cause hard failures.

---

# CURRENT CANONICAL RETRIEVAL PIPELINE

```text
Query
  ↓
Dense query embedding + Sparse BM25 query vector
  ↓
Qdrant named-vector Dense/Sparse prefetch
  ↓
RRF Hybrid child fusion
  ↓
Child candidates
  ↓
group/deduplicate by parent_id
  ↓
MongoDB authoritative ParentChunk expansion
  ↓
CrossEncoder parent reranking/pruning
  ↓
dense Parent MMR
  ↓
final parent context
  ↓
LLM
```

Locked ordering remains:

```text
child candidates
→ parent expansion/deduplication
→ CrossEncoder reranking at parent level
→ Parent MMR
→ final context
```

Do NOT move CrossEncoder before parent expansion.

---

# BENCHMARK V2 BASELINE RESULTS — CANONICAL PRE-RERANKING BASELINE

The baseline runner now evaluates both:

```text
CHILD level = diagnostic candidate-generation quality
PARENT level = primary context-recovery quality
```

## Dense

```text
CHILD @5
Hit@5       = 1.0000
Precision@5 = 0.4516
Recall@5    = 0.4388
MRR@5       = 0.9274
MAP@5       = 0.4622
nDCG@5      = 0.6413

PARENT @3
Hit@3       = 1.0000
Precision@3 = 0.5269
Recall@3    = 0.5242
MRR@3       = 0.9516
MAP@3       = 0.5896
nDCG@3      = 0.7076
```

## Sparse

```text
CHILD @5
Hit@5       = 0.9677
Precision@5 = 0.4129
Recall@5    = 0.3813
MRR@5       = 0.8656
MAP@5       = 0.4067
nDCG@5      = 0.5845

PARENT @3
Hit@3       = 0.9677
Precision@3 = 0.5269
Recall@3    = 0.5134
MRR@3       = 0.8656
MAP@3       = 0.5529
nDCG@3      = 0.6820
```

## Hybrid

Latest run:

```text
CHILD @5
Hit@5       = 0.9677
Precision@5 = 0.4645
Recall@5    = 0.4302
MRR@5       = 0.9032
MAP@5       = 0.4658
nDCG@5      = 0.6602

PARENT @3
Hit@3       = 1.0000
Precision@3 = 0.5484
Recall@3    = 0.5333
MRR@3       = 0.9194
MAP@3       = 0.5851
nDCG@3      = 0.7330
```

## Interpretation

Parent-level metrics are intentionally treated as the more important baseline because ParentChunks are what proceed toward final RAG context.

Child metrics remain essential diagnostics because they show whether failures originated in candidate generation.

Do NOT directly describe child Recall@5 -> parent Recall@3 as a strict percentage improvement because:

- the units differ (children vs parents),
- the gold relevance sets differ,
- K differs (5 vs 3).

The baseline nevertheless demonstrates why parent expansion matters: multiple relevant child chunks can map to one useful parent, so child-level recall can look modest even when the necessary parent context is recovered.

Observed baseline pattern:

- Dense achieved Parent Hit@3 = 1.0 and strongest parent MRR in the latest run.
- Hybrid achieved strongest parent Precision@3 and nDCG@3 in the latest run.
- Sparse alone was generally weaker than Dense/Hybrid on this frozen corpus.

These are baseline observations only; do not treat them as universal architecture conclusions.

---

# PERFORMANCE OBSERVATIONS

## Dense first-query warm-up

Dense Q001 repeatedly showed multi-second first-query latency while subsequent queries were usually much faster.

Treat this as cold-start/warm-up behavior rather than normal steady-state latency.

Future latency reporting should distinguish:

```text
cold-start latency
vs
steady-state latency
```

## Parent expansion latency

Parent expansion was consistently around ~230–245 ms/query for many queries.

Current implementation performs individual `ParentRepository.get_by_id()` calls while expanding unique parents.

Potential optimization to record for later:

```text
ParentRepository.get_by_ids([...])
```

This could batch MongoDB reads into one round trip instead of multiple per query.

Do NOT expand current benchmark scope solely for this optimization; record it for the cloud/production optimization phase.

## Sparse candidate counts

Sparse retrieval sometimes returns fewer than the configured maximum candidate limit.

This is valid: candidate limit means *up to* N results, not that the runner must fabricate N results.

---

# PYTEST STATUS — FINAL FOR THIS MILESTONE

Latest full suite:

```text
98 tests collected
98 passed
```

Coverage includes:

- frozen benchmark manifest and SHA integrity,
- frozen unit counts/identity integrity,
- gold dataset integrity,
- ranking metrics,
- ranking evaluator,
- benchmark evaluator,
- benchmark output models,
- GoldQueryV2 relevance maps,
- child baseline runner,
- parent expansion runner.

Earlier suite before parent expansion contained 89 tests; after parent-expansion additions/refactors the canonical current count is 98 passed.

---

# COMMIT BOUNDARY — NOW

The benchmark/evaluation milestone is complete.

Run:

```powershell
git status
git add .
git commit -m "Complete Benchmark V2 child and parent baseline evaluation"
```

After the commit, create the next branch:

```powershell
git switch -c refactor/cloud-retrieval-providers
```

---

# NEXT PHASE — CLOUD PROVIDER REFACTOR

User decision: complete benchmark milestone first, then migrate to cloud providers, commit, then resume HyDE.

Target sequence:

```text
Benchmark V2 milestone
    -> COMMIT

Cloud provider refactor
    -> hosted embeddings
    -> hosted reranking
    -> Qdrant Cloud
    -> settings cleanup
    -> provider abstractions
    -> mocked unit tests for external APIs
    -> benchmark corpus reingestion if embedding model/dimension changes
    -> persistence validation
    -> rerun baseline benchmark as needed
    -> COMMIT

HyDE
    -> resume from paused experiment
```

Cloud-provider requirements raised by user:

1. Move hardcoded CrossEncoder model (`cross-encoder/ms-marco-MiniLM-L6-v2`) into settings/configuration.
2. Prefer hosted/free-tier API providers for embeddings, reranking, and Qdrant DB so the portfolio can be deployed to a website without requiring local model downloads/images.
3. Preserve clean provider abstractions rather than scattering raw API calls throughout the application.
4. Keep secrets in environment variables; never commit API keys.

Recommended design direction:

```text
EmbeddingService abstraction
    ├── local implementation
    └── hosted implementation

Reranker abstraction
    ├── local CrossEncoder implementation
    └── hosted implementation

Qdrant
    ├── local URL configuration
    └── Qdrant Cloud URL/API-key configuration
```

If embedding model/dimension changes, Qdrant vectors must be re-created/re-embedded. Gold relevance judgments do not need recuration solely because the embedding model changed, provided frozen text/chunk boundaries/keys remain unchanged.

---

# HYDE — PAUSED UNTIL CLOUD MIGRATION COMPLETES

HyDE remains paused intentionally.

Locked HyDE semantics:

```text
Query
    ↓
LLM hypothetical answer-like document
    ↓
embed hypothetical document
    ↓
retrieve real corpus
```

HyDE text is a retrieval probe only; it must never be treated as final source truth.

Hybrid HyDE design remains:

```text
HyDE dense vector
+
ORIGINAL query sparse/BM25 vector
    ↓
RRF hybrid child candidates
    ↓
parent expansion/deduplication
    ↓
CrossEncoder parent reranking
    ↓
Parent MMR
    ↓
final context
```

Resume only after cloud-provider refactor is complete and committed.

---

# EXACT STOP POINT

**Stop point:** Benchmark V2 child+parent baseline evaluation is complete; full suite is 98/98 green; real Dense/Sparse/Hybrid child and parent baseline metrics have been recorded.

**Immediate action:** Commit current benchmark milestone.

**Then:** create `refactor/cloud-retrieval-providers` and move to a fresh chat window for cloud migration.

**Do not start HyDE yet.**


---

# CHECKPOINT UPDATE — 2026-09-30 — FINAL RETRIEVAL REFACTOR / HYDE INTEGRATION

## SESSION DECISION

This is the final planned large retrieval refactor before returning to the normal Agentic AI mentorship roadmap.

The user explicitly requested that local-vs-cloud choices must not create separate ingestion/search implementations. Runtime provider selection must be controlled by `.env` configuration and/or CLI flags. After HyDE, every commit should leave the repository executable from the command line.

## TARGET PRODUCTION RETRIEVAL FLOW

Default user-query path:

```text
User query
    -> Multi-Query generation (default)
    -> Hybrid child retrieval for each query
       - dense semantic retrieval
       - sparse/BM25 lexical retrieval
       - Qdrant RRF fusion
    -> cross-query RRF fusion
    -> parent expansion from authoritative MongoDB
    -> parent deduplication
    -> parent-level reranking
    -> dense Parent MMR
    -> final parent context
```

Supported query-transformation modes:

```text
original
rewrite
multi_query   # DEFAULT
hyde
```

UI requirement recorded:
- UI must allow the user to select at least `multi_query` or `rewrite`.
- HyDE is also implemented as a selectable retrieval mode.
- The backend must expose the same modes later through FastAPI routes.

Query-generation model:

```text
Groq / openai/gpt-oss-120b
```

Use it for:
- query rewriting,
- multi-query generation,
- HyDE hypothetical-document generation.

## FINAL PROVIDER ARCHITECTURE

One public ingestion/search implementation is retained. Provider-specific behavior is behind factories/configuration.

### Dense embeddings

Local:
```text
sentence-transformers/all-MiniLM-L6-v2
384 dimensions
```

Cloud:
```text
Jina jina-embeddings-v4
512 dimensions
```

### Sparse embeddings

Local:
```text
FastEmbed Qdrant/bm25 -> explicit SparseVector
```

Cloud:
```text
Qdrant Cloud server-side BM25 inference
SparseText(text, model) -> Qdrant models.Document at vector-store boundary
```

Important architectural correction:
- DO NOT maintain separate `insert_children_cloud_sparse`, `sparse_search_text`, or `hybrid_search_text` public APIs.
- `HybridQdrantStore` keeps one provider-neutral public API:
  - `insert_children()`
  - `dense_search()`
  - `sparse_search()`
  - `hybrid_search()`
- Sparse provider returns a common domain representation:
  - `SparseVector` for local FastEmbed,
  - `SparseText` for Qdrant Cloud inference.
- Qdrant-specific conversion happens only inside `HybridQdrantStore`.

This removes the previous constructor ripple and avoids separate local/cloud retrieval implementations.

### Reranker

Local:
```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

Cloud:
```text
jina-reranker-v2-base-multilingual
```

All downstream retrieval code depends on the generic `Reranker` contract rather than `LocalCrossEncoderReranker`.

## PARENT MMR

Canonical score-space rule remains locked:
- RRF = candidate discovery/fusion only.
- reranker score = parent relevance ordering/trace.
- MMR relevance = dense query-to-parent cosine similarity.
- MMR redundancy = dense parent-to-parent cosine similarity.
- Never numerically mix RRF, reranker, and dense cosine scores.

Naming cleanup:
```text
cross_encoder_rank  -> reranker_rank
cross_encoder_score -> reranker_score
```

Parent embeddings are NOT persisted in MongoDB.
- MongoDB remains the authoritative parent text/metadata store.
- Reranking consumes query text + parent text and does not need embeddings.
- Parent embeddings are needed only for MMR.

A shared `ParentMMRService` now caches parent embeddings in memory using parent identity/content hash so unchanged parent embeddings are not recomputed across retrieval modes in the same process. Query embeddings are still computed per user query.

## HYDE DESIGN

HyDE is integrated into the full hybrid funnel rather than remaining dense-only.

```text
original user query
    -> gpt-oss-120b hypothetical document
    -> dense query representation uses hypothetical document
    -> sparse/BM25 representation uses ORIGINAL user query
    -> Qdrant hybrid RRF
    -> parent expansion/dedup
    -> reranker using ORIGINAL query
    -> Parent MMR anchored to ORIGINAL query
```

Rationale:
- hypothetical document improves semantic dense discovery,
- original query preserves lexical identifiers/terms for sparse retrieval,
- final relevance remains anchored to actual user intent.

## NEW CANONICAL APPLICATION ENTRY POINTS

### CLI

```text
python -m app.cli ingest ...
python -m app.cli search ...
```

Installed console entry point in `pyproject.toml`:

```text
agentic-rag = app.cli:main
```

Runtime switches:

```text
--local
--cloud
--collection <name>
--mode original|rewrite|multi_query|hyde
```

Default query mode:

```text
multi_query
```

### Environment configuration

`.env.example` added/updated with provider/deployment/model settings.

Key controls:

```text
QDRANT_DEPLOYMENT
DENSE_EMBEDDING_PROVIDER
SPARSE_EMBEDDING_PROVIDER
RERANKER_PROVIDER
QUERY_TRANSFORMATION_MODE
QUERY_LLM_MODEL
QDRANT_LOCAL_COLLECTION
QDRANT_CLOUD_COLLECTION
```

CLI `--local`/`--cloud` creates a coherent provider profile rather than requiring the caller to set each provider independently.

## COLLECTION NAMING CLEANUP

Previous ambiguity:
- `settings.py` defaulted to one collection,
- benchmark ingestion hardcoded `structure_aware_v4_benchmark_v2_cloud`.

New direction:
- `HybridSearchConfig` owns both canonical names:
  - `local_collection_name`
  - `cloud_collection_name`
- application/bootstrap logic selects the correct collection based on deployment.
- explicit `--collection` remains available for experiments.

## BENCHMARK / RUNNER CONSOLIDATION

Canonical query-transformation Benchmark V2 runner:

```text
benchmarks/runners/run_query_modes_benchmark_v2.py
```

It evaluates the same full downstream retrieval funnel for:
- original,
- rewrite,
- multi-query,
- HyDE.

Compatibility runners for query rewrite and multi-query now delegate to the canonical Benchmark V2 runner rather than rebuilding provider stacks independently.

Existing dedicated reranker Benchmark V2 runner remains because it serves a different experiment: local-vs-cloud reranker comparison on the same candidate set.

## MOST RECENT ACCEPTED QUALITY RESULTS BEFORE FINAL REFACTOR

From the completed 32-query run:

```text
parent_baseline_no_rerank
Precision .5914
Recall    .5656
MRR       .9194
MAP       .6434
nDCG      .7419

parent_post_local_rerank
Precision .5914
Recall    .5699
MRR       .9140
MAP       .6416
nDCG      .7647

parent_post_local_rerank_mmr
Precision .6022
Recall    .5833
MRR       .9355
MAP       .6577
nDCG      .7546

parent_post_cloud_rerank
Precision .5591
Recall    .5430
MRR       .9462
MAP       .6165
nDCG      .7659

parent_post_cloud_rerank_mmr
Precision .6022
Recall    .5833
MRR       .9516
MAP       .6631
nDCG      .7645
```

Interpretation:
- cloud reranker alone prioritized top-ranked relevance strongly but displaced some supporting parents,
- Parent MMR recovered a stronger final context set,
- cloud reranker + MMR produced the strongest overall final-context metrics in this frozen run.

## VALIDATION STATUS OF REFACTORED SOURCE

Validation completed in the ChatGPT working container:

```text
python -m compileall -q app benchmarks scripts tests
PASS
```

Dependency-light pytest selection:

```text
53 passed
```

This included:
- metadata,
- child chunker,
- parent chunker,
- structure detector,
- ranking metrics,
- ranking evaluator,
- benchmark models,
- gold models,
- benchmark evaluator,
- new Parent MMR embedding-cache test,
- new Qdrant Cloud sparse-provider tests that do not require qdrant-client runtime.

A smaller focused run after adding new tests also passed:

```text
4 passed
```

Full pytest could NOT be executed in the ChatGPT container because the container lacks the project runtime packages (`qdrant-client`, `pymongo`, `sentence-transformers`) and outbound PyPI access is unavailable. This is an environment limitation, not a claimed full-suite pass. The user's Windows venv must run the final full suite.

## REQUIRED USER-SIDE ACCEPTANCE RUN

From project root in the user's existing venv:

```powershell
python -m pytest -q
```

Then local smoke:

```powershell
python -m app.cli search --local --mode multi_query --query "How does HNSW avoid comparing every vector?"
```

Cloud ingestion/rebuild (required after switching the sparse representation to Qdrant Cloud BM25):

```powershell
python -m app.cli ingest --cloud --documents benchmarks/data/documents --recreate
```

Cloud default search:

```powershell
python -m app.cli search --cloud --query "How does HNSW avoid comparing every vector?"
```

Explicit query modes:

```powershell
python -m app.cli search --cloud --mode rewrite --query "How does HNSW avoid comparing every vector?"
python -m app.cli search --cloud --mode multi_query --query "How does HNSW avoid comparing every vector?"
python -m app.cli search --cloud --mode hyde --query "How does HNSW avoid comparing every vector?"
```

Canonical query-mode Benchmark V2:

```powershell
python -m benchmarks.runners.run_query_modes_benchmark_v2 --cloud --modes rewrite multi_query hyde
```

Dedicated reranker/MMR benchmark:

```powershell
python -m benchmarks.runners.run_reranker_benchmark_v2
```

Persistence acceptance:

```powershell
python -m benchmarks.tools.validate_benchmark_v2_persistence
```

Expected frozen identities remain:

```text
24 source documents
32 gold queries
120 parents in MongoDB
196 children in Qdrant
```

After cloud sparse reingestion, validate:
- 196 Qdrant child points,
- named dense vector exists and uses 512 dimensions in cloud collection,
- named sparse vector exists,
- payload stable keys remain present,
- Mongo parent count remains 120,
- child->parent references are valid.

## FASTAPI / FULL-STACK ROADMAP REQUIREMENT

Future Agentic AI mentorship sessions must include a real backend application architecture using FastAPI.

Target direction:

```text
Web frontend
    -> REST API
FastAPI backend
    -> ingestion routes
    -> retrieval/search routes
    -> query-mode selection
    -> agent invocation routes
    -> single-agent and multi-agent orchestration
    -> persistence / observability / evaluation
```

The frontend will invoke backend REST endpoints rather than embedding retrieval/agent logic in UI code.

This is a future roadmap requirement and is NOT to be inserted into the current retrieval refactor as unrelated scope.

## MENTORSHIP STYLE AFTER THIS MILESTONE

After this refactor is accepted and HyDE is validated, return to the normal mentorship pattern:

```text
complete runnable implementation
-> conceptual explanation
-> code walkthrough
-> experiments
-> interpretation
-> production trade-offs
-> interview reasoning/questions
```

The user wants to progress toward Agentic AI Architect capability by December 2026. Roadmap pacing must be actively monitored so implementation/refactor work does not indefinitely delay later Agentic AI topics.

## EXACT NEXT STEP

1. Deliver the complete refactored source ZIP plus this checkpoint.
2. User runs full pytest in the existing Windows venv.
3. Fix only genuine failures found by that run; no new architecture redesign.
4. Reingest cloud Benchmark V2 because sparse cloud inference generation path changed.
5. Run persistence validation.
6. Run canonical query-mode Benchmark V2 including HyDE.
7. Commit final retrieval/cloud/HyDE refactor at an executable state.
8. Next mentorship session begins a NEW roadmap topic in the normal implementation-first teaching style.
