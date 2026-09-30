# Validation Report — 2026-09-30

## Completed in ChatGPT working container

### Static compilation

```text
python -m compileall -q app benchmarks scripts tests
PASS
```

### Dependency-light test selection

```text
53 passed
```

Covered:
- metadata models
- child chunker
- parent chunker
- structure detector
- ranking metrics
- ranking evaluator
- benchmark models
- gold models
- benchmark evaluator
- Parent MMR embedding reuse/cache
- cloud sparse provider domain representation

### Additional focused run

```text
4 passed
```

Covered new Parent MMR cache + cloud sparse provider tests.

## Full-suite limitation in this container

`python -m pytest -q` could not complete collection because this container does not contain the project runtime dependencies `qdrant-client`, `pymongo`, and `sentence-transformers`. Attempting `pip install -e .` failed because outbound PyPI/DNS access is unavailable in this environment.

This is an environment limitation. It is **not** recorded as a full-suite pass.

## Required user-side acceptance

Run from the project root in the existing Windows venv:

```powershell
python -m pytest -q
```

Then rebuild the cloud collection because the sparse generation path changed to Qdrant Cloud BM25:

```powershell
python -m app.cli ingest --cloud --documents benchmarks/data/documents --recreate
```

Validate persistence:

```powershell
python -m benchmarks.tools.validate_benchmark_v2_persistence --cloud
```

Run default multi-query search:

```powershell
python -m app.cli search --cloud --query "How does HNSW avoid comparing every vector?"
```

Run all advanced query modes:

```powershell
python -m benchmarks.runners.run_query_modes_benchmark_v2 --cloud --modes rewrite multi_query hyde
```
