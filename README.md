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
git clone https://github.com/umurozkul/open-brain-rs
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

## Why Rust + Qdrant

- **HNSW indexing** — sub-millisecond semantic search vs brute-force
- **Pre-filter** — Qdrant payload filtering before vector search, not after
- **No GC** — memory-safe, zero-copy operations, no pause times
- **qdrant-client** — first-class Rust client, same language as the DB

Phase 1 (Node.js + sqlite-vec) proved the concept with 36/36 semantic recall tests.
Phase 2 (this repo) is the production-grade implementation.
