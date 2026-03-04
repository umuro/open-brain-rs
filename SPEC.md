# Open Brain Phase 2 — Rust + Qdrant
**Status:** Design locked — Phase 1 proven (36/36 tests pass, 2026-03-02)
**Goal:** Migrate from Node.js+sqlite-vec → Rust+Qdrant. Publish as flagship article + GitHub repo.

---

## What Phase 1 Proved (the invariants Phase 2 must preserve)

| Test | Result | Phase 2 must... |
|------|--------|-----------------|
| 12 semantic recalls, zero keyword overlap | ✅ all pass | Use same embedding model (Gemini) |
| Distance quality < 0.95 | ✅ all pass | Keep HNSW or better indexing |
| Type filtering | ✅ exact | Preserve payload filtering in Qdrant |
| Roundtrip store → recall | ✅ immediate | No async indexing lag |
| MCP server SSE | ✅ 200 OK | Replace Node SSE with Rust Axum SSE |
| Edge cases | ✅ all pass | Maintain same API surface |

**API contract (must not break):**
```
store(content, type, people, topics, importance) → id
recall(query, limit, type?) → [{id, content, type, date, distance}]
listRecent(days, type?, limit) → [{id, content, type, date}]
stats() → {total, last7days, byType, oldestEntry}
```

---

## Why Migrate to Rust + Qdrant

| Dimension | Phase 1 (sqlite-vec) | Phase 2 (Qdrant) |
|-----------|---------------------|-------------------|
| Indexing | brute-force cosine | HNSW (hierarchical navigable small world) |
| Scale | ~10K vectors OK | billions of vectors |
| Query latency | ~5ms | ~1ms (HNSW + SIMD) |
| Filtering | post-filter in JS | pre-filter in Qdrant (payload indexing) |
| Horizontal scale | single file | clustering, replication |
| Language | Node.js | Rust — memory safe, no GC pauses |
| Blog story | prototype | "production-grade AI infrastructure in Rust" |
| GitHub appeal | JS script | Rust crate = stars from Rust community |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  open-brain-rs (Rust crate)                         │
│                                                     │
│  src/                                               │
│    main.rs          — Axum HTTP server              │
│    embed.rs         — Gemini HTTP client            │
│    store.rs         — Qdrant write path             │
│    recall.rs        — Qdrant search path            │
│    mcp/             — MCP protocol (SSE transport)  │
│      server.rs      — McpServer impl                │
│      tools.rs       — remember/recall/list/stats    │
│      transport.rs   — SSE + POST handler            │
│    config.rs        — env vars, defaults            │
│                                                     │
│  Cargo.toml                                         │
└─────────────────────────────────────────────────────┘

External services:
  Qdrant — vector DB (Docker: qdrant/qdrant)
  Gemini API — embeddings (same key, same model)
```

---

## Cargo.toml

```toml
[package]
name = "open-brain"
version = "0.1.0"
edition = "2021"

[dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }

# HTTP server
axum = { version = "0.7", features = ["macros"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }

# Qdrant client (written in Rust, first-class)
qdrant-client = "1.10"

# HTTP client (Gemini embedding calls)
reqwest = { version = "0.12", features = ["json"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Error handling
anyhow = "1"
thiserror = "2"

# Environment
dotenvy = "0.15"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# UUID for session management
uuid = { version = "1", features = ["v4"] }
```

---

## Key Implementation Files

### src/embed.rs
```rust
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    content: Content,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Embedding,
}

#[derive(Deserialize)]
struct Embedding {
    values: Vec<f32>,
}

pub struct Embedder {
    client: Client,
    api_key: String,
}

impl Embedder {
    pub fn new(api_key: String) -> Self {
        Self { client: Client::new(), api_key }
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={}",
            self.api_key
        );
        
        let req = EmbedRequest {
            model: "models/gemini-embedding-001".into(),
            content: Content {
                parts: vec![Part { text: text.chars().take(8000).collect() }],
            },
        };

        // Retry on 429
        for attempt in 0..5u32 {
            let res = self.client.post(&url).json(&req).send().await?;
            if res.status().as_u16() == 429 {
                let wait = 2u64.pow(attempt) * 1000;
                tokio::time::sleep(std::time::Duration::from_millis(wait)).await;
                continue;
            }
            let body: EmbedResponse = res.error_for_status()?.json().await?;
            return Ok(body.embedding.values);
        }
        anyhow::bail!("Gemini 429 after 5 retries")
    }
}
```

### src/store.rs
```rust
use anyhow::Result;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, UpsertPointsBuilder, PointStruct,
    VectorsConfig, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use serde_json::json;
use std::collections::HashMap;

pub const COLLECTION: &str = "open_brain";
pub const DIM: u64 = 3072;

pub async fn ensure_collection(client: &Qdrant) -> Result<()> {
    let collections = client.list_collections().await?;
    if collections.collections.iter().any(|c| c.name == COLLECTION) {
        return Ok(());
    }
    client.create_collection(
        CreateCollectionBuilder::new(COLLECTION)
            .vectors_config(VectorsConfig::Params(
                VectorParamsBuilder::new(DIM, Distance::Cosine).build()
            ))
    ).await?;
    Ok(())
}

pub struct MemoryPayload {
    pub content: String,
    pub memory_type: String,
    pub topics: Vec<String>,
    pub people: Vec<String>,
    pub source: String,
    pub importance: u8,
    pub created_at: i64,
}

pub async fn store_memory(
    client: &Qdrant,
    id: uuid::Uuid,
    vector: Vec<f32>,
    payload: MemoryPayload,
) -> Result<()> {
    let mut pl = HashMap::new();
    pl.insert("content",    json!(payload.content));
    pl.insert("type",       json!(payload.memory_type));
    pl.insert("topics",     json!(payload.topics));
    pl.insert("people",     json!(payload.people));
    pl.insert("source",     json!(payload.source));
    pl.insert("importance", json!(payload.importance));
    pl.insert("created_at", json!(payload.created_at));

    client.upsert_points(
        UpsertPointsBuilder::new(COLLECTION, vec![
            PointStruct::new(id.to_string(), vector, pl)
        ])
    ).await?;
    Ok(())
}
```

### src/recall.rs
```rust
use anyhow::Result;
use qdrant_client::qdrant::{
    SearchPointsBuilder, Condition, Filter, FieldCondition, Match,
};
use qdrant_client::Qdrant;

pub struct RecallResult {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub topics: Vec<String>,
    pub people: Vec<String>,
    pub created_at: i64,
    pub score: f32,
}

pub async fn semantic_recall(
    client: &Qdrant,
    vector: Vec<f32>,
    limit: u64,
    filter_type: Option<&str>,
) -> Result<Vec<RecallResult>> {
    let mut builder = SearchPointsBuilder::new(COLLECTION, vector, limit)
        .with_payload(true);

    if let Some(t) = filter_type {
        builder = builder.filter(Filter::must([
            Condition::Field(FieldCondition {
                key: "type".into(),
                r#match: Some(Match::Keyword(t.into())),
                ..Default::default()
            })
        ]));
    }

    let results = client.search_points(builder).await?;

    Ok(results.result.into_iter().map(|p| {
        let pl = p.payload;
        RecallResult {
            id:          pl.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            content:     pl.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            memory_type: pl.get("type").and_then(|v| v.as_str()).unwrap_or("note").to_string(),
            topics:      pl.get("topics").and_then(|v| v.as_array())
                           .map(|a| a.iter().filter_map(|x| x.as_str()).map(String::from).collect())
                           .unwrap_or_default(),
            people:      pl.get("people").and_then(|v| v.as_array())
                           .map(|a| a.iter().filter_map(|x| x.as_str()).map(String::from).collect())
                           .unwrap_or_default(),
            created_at:  pl.get("created_at").and_then(|v| v.as_i64()).unwrap_or(0),
            score: p.score,
        }
    }).collect())
}
```

### src/main.rs
```rust
use axum::{routing::{get, post}, Router};
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

mod config;
mod embed;
mod store;
mod recall;
mod mcp;

#[derive(Clone)]
pub struct AppState {
    pub qdrant: Arc<qdrant_client::Qdrant>,
    pub embedder: Arc<embed::Embedder>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cfg = config::Config::from_env()?;

    let qdrant = Arc::new(qdrant_client::Qdrant::from_url(&cfg.qdrant_url).build()?);
    store::ensure_collection(&qdrant).await?;

    let embedder = Arc::new(embed::Embedder::new(cfg.gemini_api_key.clone()));
    let state = AppState { qdrant, embedder };

    let app = Router::new()
        .route("/health", get(mcp::handlers::health))
        .route("/sse",    get(mcp::transport::sse_handler))
        .route("/messages", post(mcp::transport::message_handler))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cfg.port);
    tracing::info!("🧠 Open Brain (Rust) on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

---

## Migration from Phase 1

Phase 2 imports all memories from Phase 1's brain.db:

```rust
// migrate_from_sqlite.rs
// Reads brain.db (better-sqlite3 schema) → Qdrant upsert
// Run once: cargo run --bin migrate
```

Steps:
1. Read all rows from `memories` table (SQLite)
2. Re-embed each with Gemini (or read from sqlite-vec if compatible)
3. Upsert into Qdrant collection
4. Verify recall results match Phase 1 test suite

The Phase 1 test.js becomes `tests/recall_tests.rs` — same 36 assertions, now in Rust.

---

## Docker Compose (Phase 2)

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage

  open-brain:
    build: .
    ports:
      - "3737:3737"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - QDRANT_URL=http://qdrant:6334
      - PORT=3737
    depends_on:
      - qdrant
    restart: unless-stopped
```

---

## The Blog Post Structure (writes itself from this doc)

**Title:** "I Built My AI's Memory Layer in Rust — Here's the Architecture"

```
1. The Problem (2 min read)
   - Every AI session starts from zero
   - Platform silos: Claude ≠ Claude Code ≠ Cursor
   - The 7 memory traps (from Phase 1 research)

2. Phase 1: Proving the Concept (3 min)
   - sqlite-vec + Node.js + Gemini embeddings
   - 36/36 tests pass
   - Show the semantic recall results (no keywords!)
   - "The concept is proven. Now let's build it right."

3. Why Rust (2 min)
   - Qdrant is written in Rust — not coincidence
   - HNSW requires zero-copy, SIMD distance calculations
   - No GC pauses in production memory infrastructure
   - The infrastructure layer of the agent web will be Rust

4. Architecture Walkthrough (5 min)
   - embed.rs → store.rs → recall.rs → MCP server
   - Axum + qdrant-client + tokio
   - MCP SSE transport: any AI tool connects to localhost:3737

5. The Results (1 min)
   - Same 36 tests, now in Rust
   - 10x faster query latency (HNSW vs brute force)
   - One brain, every tool

6. What's Next: Knowledge Graph Layer (1 min)
   - GraphRAG: vector + entity relationships
   - "I'll write that in Part 2"
   - GitHub: umurozkul/open-brain-rs
```

---

## Validation Gate Before Phase 2 Starts

Phase 2 begins when ALL of:
- [ ] Phase 1 test.js passes 36/36 ✅ (DONE)
- [ ] Migration script picks up all 297 memories ✅ (DONE)
- [ ] MCP server health endpoint responds ✅ (DONE)
- [ ] Blog post outline approved by Umur
- [ ] Qdrant running locally (`docker run -p 6333:6333 qdrant/qdrant`)

---

## Estimated Timeline

| Task | Time |
|------|------|
| `cargo new open-brain-rs` + Cargo.toml | 30 min |
| embed.rs (Gemini HTTP client) | 1 hour |
| store.rs (Qdrant upsert) | 1 hour |
| recall.rs (Qdrant search + filter) | 1 hour |
| MCP SSE transport in Axum | 2 hours |
| MCP tools (remember/recall/list/stats) | 1 hour |
| Migration from SQLite | 1 hour |
| Tests (same 36 assertions in Rust) | 1 hour |
| Dockerfile + docker-compose | 30 min |
| **Total** | **~9 hours** |

Split across 2 days. Phase 2 is a proper Claude Code project.

---

## Repo Structure (GitHub)

```
umurozkul/open-brain-rs/
  README.md          ← "AI memory layer in Rust — MCP server for your AI brain"
  Cargo.toml
  src/
    main.rs
    embed.rs
    store.rs
    recall.rs
    config.rs
    mcp/
      mod.rs
      server.rs
      tools.rs
      transport.rs
  tests/
    recall_tests.rs   ← 36 assertions, Rust
  docker-compose.yml
  .env.example
  ARCHITECTURE.md    ← the blog post, technical version
```

The README is the blog post intro. The repo IS the flagship content.

---

*Phase 1 proven: 2026-03-02 — 36/36 tests, 297 memories, MCP server healthy*
*Phase 2 status: READY TO BUILD*
