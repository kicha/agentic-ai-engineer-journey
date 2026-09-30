# Agentic RAG

Production-oriented Parent/Child RAG learning project with provider-neutral local/cloud execution.

## Canonical retrieval flow

Default search mode is `multi_query`:

```text
User query
  -> Multi-Query generation (gpt-oss-120b)
  -> Hybrid child retrieval (dense + sparse/BM25)
  -> Qdrant RRF
  -> Cross-query RRF
  -> Parent expansion/deduplication (MongoDB)
  -> Parent reranker
  -> Dense Parent MMR
  -> Final parent context
```

Alternative query modes:

- `rewrite`
- `multi_query` (default)
- `hyde`
- `original`

HyDE uses the hypothetical document for the dense semantic branch and preserves the original user query for sparse/BM25 retrieval, reranking, and final Parent MMR relevance.

## Local vs cloud

There is one ingestion implementation and one retrieval implementation. Provider selection happens through configuration/factories.

### Local

- Dense: `sentence-transformers/all-MiniLM-L6-v2`
- Sparse: FastEmbed `Qdrant/bm25`
- Qdrant: local
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`

### Cloud

- Dense: Jina `jina-embeddings-v4`
- Sparse: Qdrant Cloud BM25 server-side inference
- Qdrant: Qdrant Cloud
- Reranker: Jina hosted reranker
- Query rewriting / multi-query / HyDE: Groq `openai/gpt-oss-120b`

## Setup

Copy `.env.example` to `.env` and fill in the keys required by the selected profile. The refactor defaults intentionally target isolated MongoDB/Qdrant collections so the previously accepted Benchmark V2 persistence is not overwritten.

Install dependencies in your existing project venv:

```powershell
python -m pip install -e .
```

## CLI

### Rebuild cloud benchmark collection

```powershell
python -m app.cli ingest --cloud --documents benchmarks/data/documents --recreate
```

### Default search: multi-query

```powershell
python -m app.cli search --cloud --query "How does HNSW avoid comparing every vector?"
```

### Explicit query mode

```powershell
python -m app.cli search --cloud --mode rewrite --query "How does HNSW avoid comparing every vector?"
python -m app.cli search --cloud --mode multi_query --query "How does HNSW avoid comparing every vector?"
python -m app.cli search --cloud --mode hyde --query "How does HNSW avoid comparing every vector?"
```

### Local fallback

```powershell
python -m app.cli search --local --mode multi_query --query "How does HNSW avoid comparing every vector?"
```

`--local` and `--cloud` select coherent provider profiles. `--collection` can override the canonical collection name for experiments.

## Tests

```powershell
python -m pytest -q
```

## Benchmark V2

Query transformation comparison:

```powershell
python -m benchmarks.runners.run_query_modes_benchmark_v2 --cloud --modes rewrite multi_query hyde
```

Reranker + Parent MMR comparison:

```powershell
python -m benchmarks.runners.run_reranker_benchmark_v2
```

Persistence acceptance:

```powershell
python -m benchmarks.tools.validate_benchmark_v2_persistence --cloud
```

Frozen Benchmark V2 identity:

- 24 documents
- 32 gold queries
- 120 MongoDB parents
- 196 Qdrant children
