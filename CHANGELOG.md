# Changelog

## [0.2.0] - 2026-03-04

### Performance
- Multi-threaded Tokio runtime (16 workers by default, configurable via `TOKIO_WORKERS`)
- LRU embedding cache (10K entries, ~120MB) — eliminates redundant Gemini API calls
- Parallel batch embedding with configurable concurrency (`EMBED_CONCURRENCY=8`)
- Optimized reqwest connection pool (max 32 idle, TCP keepalive 60s)
- Brain stats cache (60s TTL) — eliminates expensive full-collection scans
- Qdrant HNSW tuning: `ef_construct` and `m` configurable via env (`HNSW_EF=256`, `HNSW_M=32`)
- Qdrant payload field indexing on `type`, `topics`, `people`, `created_at`, `importance`
- Qdrant `MAX_SEARCH_THREADS=16` via Docker Compose config

### Features
- New MCP tool: `remember_batch` — store N memories in one call with parallel embedding + batch upsert
- New env vars: `TOKIO_WORKERS`, `EMBED_CACHE_SIZE`, `EMBED_CONCURRENCY`, `HNSW_EF`, `HNSW_M`, `STATS_CACHE_TTL_SECS`

### Breaking Changes
- None — full API compatibility with 0.1.0

## [0.1.0] - 2026-03-02

### Initial Release
- Axum HTTP server with MCP SSE transport
- Qdrant vector storage with Gemini embeddings (dim=3072, Cosine)
- MCP tools: `remember`, `recall`, `list_recent`, `brain_stats`
- REST endpoints: `/store`, `/recall`, `/stats`, `/health`
- Bulk migration tool (`src/bin/migrate.rs`) for importing `.md` files
