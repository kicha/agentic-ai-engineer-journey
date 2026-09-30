# Agentic RAG Foundation Cleanup — 2026-09-30

## Objective

Remove legacy/duplicate configuration and retrieval paths so the project has one canonical local/cloud architecture before continuing the Agentic AI roadmap.

## Canonical configuration ownership

- `ChunkingConfig`: parent/child chunk sizes and ingestion/chunking versions only.
- `EmbeddingConfig`: dense/sparse provider selection and model names.
- `MongoConfig`: MongoDB URI, database, and authoritative ParentChunk collection.
- `HybridSearchConfig`: local/cloud Qdrant collection names, vector names, prefetch sizes, Qdrant RRF settings.
- `QdrantConfig`: deployment, local/cloud endpoint, API key, cloud inference, timeout.
- `RerankingConfig`: local/cloud reranker provider/model configuration.
- `CloudConfig`: hosted dense embedding endpoint/model/dimension, hosted reranker endpoint, hosted-provider credentials/retry policy.
- `QueryingConfig`: query transformation mode, gpt-oss-120b configuration, multi-query RRF, retrieval candidate sizes, Parent MMR settings.

`QDRANT_COLLECTION` has been removed. Runtime collection selection happens exactly once in `build_runtime_configs()`:

- local -> `QDRANT_LOCAL_COLLECTION`
- cloud -> `QDRANT_CLOUD_COLLECTION`
- CLI `--collection` -> explicit experiment override

## Removed legacy architecture

The following superseded dense-only retrieval implementation was removed from the canonical source tree:

- `app/vectorstore/qdrant_store.py`
- `app/retrieval/parent_child_retriever.py`
- `app/retrieval/mmr_parent_child_retriever.py`
- `app/retrieval/rerank_mmr_retriever.py`
- `app/retrieval/mmr_rerank_retriever.py`

Shared retrieval result models now live in:

- `app/retrieval/models.py`

Current hybrid/reranker code no longer imports legacy retrievers merely to obtain data models.

Several stale manual comparison scripts and unused benchmark evaluator implementations tied to the old architecture were also removed. Git history remains the historical archive.

## Removed configuration debt

Removed from `ChunkingConfig`:

- embedding provider/model settings
- sparse provider/model settings
- Qdrant host/port/collection
- generic `top_k`
- old child/parent retrieval defaults
- MongoDB settings
- unused `min_chunk_chars`

Also removed the unused `CloudEmbeddingProvider` enum (Jina/Voyage/Cohere) because only Jina is currently implemented. Unsupported providers should not be advertised in foundational configuration.

Renamed:

- `CloudConfig.cloud_url` -> `CloudConfig.embedding_url`

This prevents ambiguity with `QdrantConfig.cloud_url`.

## Isolated refactor persistence defaults

The cleaned branch defaults to isolated acceptance targets:

```env
QDRANT_LOCAL_COLLECTION=structure_aware_v4_benchmark_v2
QDRANT_CLOUD_COLLECTION=structure_aware_v4_benchmark_v2_cloud_refactor
MONGODB_PARENT_COLLECTION=parents_benchmark_v2_cloud_refactor
```

This prevents the refactor acceptance run from overwriting the previously accepted Benchmark V2 cloud persistence.

## Validation performed in ChatGPT container

```text
python -m compileall -q app benchmarks scripts tests
PASS
```

Dependency-light tests:

```text
102 passed
```

Architecture-specific settings tests:

```text
5 passed
```

Legacy reference scan confirmed no active references to:

- `QDRANT_COLLECTION`
- `app.vectorstore.qdrant_store`
- `app.retrieval.parent_child_retriever`
- `app.retrieval.mmr_parent_child_retriever`
- `app.retrieval.rerank_mmr_retriever`
- `app.retrieval.mmr_rerank_retriever`
- `cross_encoder_top_k`
- `cross_encoder_score`
- `cross_encoder_rank`

Full pytest could not be collected in the ChatGPT container because this environment does not have `sentence-transformers`, `qdrant-client`, or `pymongo`. Your project venv is the authoritative environment for the full suite.

## Acceptance commands on Windows

From the refactor branch and activated project venv:

```powershell
python -m compileall -q app benchmarks scripts tests
python -m pytest -q
```

Then confirm `.env` contains:

```env
QDRANT_CLOUD_COLLECTION=structure_aware_v4_benchmark_v2_cloud_refactor
MONGODB_PARENT_COLLECTION=parents_benchmark_v2_cloud_refactor
```

Cloud ingestion:

```powershell
python -m app.cli ingest --cloud --documents benchmarks/data/documents --recreate
```

Persistence acceptance:

```powershell
python -m benchmarks.tools.validate_benchmark_v2_persistence --cloud
```

Default multi-query retrieval:

```powershell
python -m app.cli search --cloud --query "How does HNSW avoid comparing every vector?"
```

Explicit query-mode acceptance:

```powershell
python -m app.cli search --cloud --mode rewrite --query "How does HNSW avoid comparing every vector?"
python -m app.cli search --cloud --mode multi_query --query "How does HNSW avoid comparing every vector?"
python -m app.cli search --cloud --mode hyde --query "How does HNSW avoid comparing every vector?"
```

Final query-mode benchmark:

```powershell
python -m benchmarks.runners.run_query_modes_benchmark_v2 --cloud --modes rewrite multi_query hyde
```

## Freeze rule

After the user's full project-environment tests, only concrete execution/test defects should be fixed. No additional architectural redesign should be introduced before resuming the Agentic AI roadmap.
