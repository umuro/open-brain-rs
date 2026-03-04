# open-brain-rs

AI memory layer in Rust — MCP server backed by Qdrant vector database and Gemini embeddings.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  open-brain-rs (Rust)                           │
│                                                 │
│  src/                                           │
│    main.rs       — Axum HTTP server             │
│    config.rs     — env vars (QDRANT_URL etc.)   │
│    embed.rs      — Gemini embedContent client   │
│    store.rs      — Qdrant write path            │
│    recall.rs     — Qdrant search + scroll       │
│    mcp/                                         │
│      mod.rs      — module exports + /health     │
│      tools.rs    — remember/recall/list/stats   │
│      transport.rs — SSE + JSON-RPC transport    │
│  bin/                                           │
│    migrate.rs    — bulk import from .md files   │
└─────────────────────────────────────────────────┘

External:
  Qdrant  — vector DB (HNSW, Cosine, dim=3072)
  Gemini  — embeddings (models/gemini-embedding-001)
```

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/umuro/open-brain-rs
cd open-brain-rs
cp .env.example .env
# Edit .env — set GEMINI_API_KEY
```

### 2. Start with Docker Compose

```bash
docker-compose up -d
```

### 3. Verify

```bash
curl http://localhost:3737/health
# {"status":"ok","service":"open-brain-rs"}
```

### 4. Connect via MCP (Claude Desktop / Cursor)

Add to your MCP config:

```json
{
  "mcpServers": {
    "open-brain": {
      "url": "http://localhost:3737/sse"
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a memory (content, type, topics, people, importance) |
| `remember_batch` | Store multiple memories in one call (parallel embedding + batch upsert) |
| `recall` | Semantic search by query, optional type filter |
| `list_recent` | List memories from the last N days |
| `brain_stats` | Total count and breakdown by type |

## Building from Source

```bash
# Requires Rust 1.75+
cargo build --release

# Run server (needs QDRANT_URL + GEMINI_API_KEY in env)
RUST_LOG=info ./target/release/open-brain

# Migrate from markdown files
SOURCE_DIR=./docs cargo run --bin migrate
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | required | Google Gemini API key |
| `QDRANT_URL` | `http://localhost:6334` | Qdrant gRPC endpoint |
| `PORT` | `3737` | HTTP server port |
| `RUST_LOG` | `info` | Log level |

## Performance Tuning

All performance parameters are configurable via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKIO_WORKERS` | `16` | Async worker threads — set to your core count |
| `EMBED_CACHE_SIZE` | `10000` | LRU cache entries for embeddings (~120MB RAM at dim=3072) |
| `EMBED_CONCURRENCY` | `8` | Parallel Gemini API calls in flight |
| `HNSW_EF` | `256` | HNSW search ef parameter — higher = better recall, slower |
| `HNSW_M` | `32` | HNSW graph connectivity — higher = better recall, more RAM |
| `STATS_CACHE_TTL_SECS` | `60` | Brain stats cache TTL (avoids full-collection scan) |

### Resource Requirements

| Tier | CPU | RAM | Notes |
|------|-----|-----|-------|
| Minimum | 4 cores | 8 GB | Development / small collections |
| Recommended | 16+ cores | 96 GB | Production — all vectors fit in RAM, zero disk I/O for search |

With 96 GB RAM and a 3072-dim collection:
- ~1M vectors ≈ 12 GB of vector data — fits entirely in RAM
- Qdrant `mmap` pages stay hot → sub-millisecond search latency
- 16 Tokio workers saturate all cores for concurrent MCP sessions

## Why Rust + Qdrant

- **HNSW indexing** — sub-millisecond semantic search vs brute-force
- **Pre-filter** — Qdrant payload filtering before vector search, not after
- **No GC** — memory-safe, zero-copy operations, no pause times
- **qdrant-client** — first-class Rust client, same language as the DB
- **LRU embedding cache** — eliminates redundant Gemini API calls for repeated queries
- **Batch upsert** — single Qdrant round-trip for N memories

Phase 1 (Node.js + sqlite-vec) proved the concept with 36/36 semantic recall tests.
Phase 2 (this repo) is the production-grade implementation.
