use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde_json::Value;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing_subscriber::EnvFilter;

mod config;
mod embed;
mod mcp;
mod recall;
mod store;

use mcp::transport::SessionMap;

#[derive(Clone)]
pub struct AppState {
    pub qdrant: Arc<qdrant_client::Qdrant>,
    pub embedder: Arc<embed::Embedder>,
    pub sessions: SessionMap,
    pub stats_cache: Arc<tokio::sync::RwLock<Option<(serde_json::Value, std::time::Instant)>>>,
    pub stats_cache_ttl_secs: u64,
}

async fn rest_store(State(state): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
    match mcp::tools::handle_tool_call("remember", body, &state).await {
        Ok(v) => Json(v),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

async fn rest_recall(State(state): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
    match mcp::tools::handle_tool_call("recall", body, &state).await {
        Ok(v) => Json(v),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

async fn rest_stats(State(state): State<AppState>) -> Json<Value> {
    match mcp::tools::handle_tool_call("brain_stats", serde_json::json!({}), &state).await {
        Ok(v) => Json(v),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

fn main() -> anyhow::Result<()> {
    let workers = std::env::var("TOKIO_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16_usize);

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()?
        .block_on(async_main())
}

async fn async_main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cfg = config::Config::from_env()?;

    let qdrant = Arc::new(qdrant_client::Qdrant::from_url(&cfg.qdrant_url).build()?);
    store::ensure_collection(&qdrant, &cfg).await?;

    let embedder = Arc::new(embed::Embedder::new(
        cfg.gemini_api_key.clone(),
        cfg.embed_cache_size,
        cfg.embed_concurrency,
    ));
    let sessions: SessionMap = Arc::new(dashmap::DashMap::new());
    let stats_cache_ttl_secs = cfg.stats_cache_ttl_secs;

    let state = AppState {
        qdrant,
        embedder,
        sessions,
        stats_cache: Arc::new(tokio::sync::RwLock::new(None)),
        stats_cache_ttl_secs,
    };

    let app = Router::new()
        // REST convenience endpoints
        .route("/health", get(mcp::handlers::health))
        .route("/store", post(rest_store))
        .route("/recall", post(rest_recall))
        .route("/stats", get(rest_stats))
        // MCP SSE transport (legacy)
        .route("/sse", get(mcp::transport::sse_handler))
        .route("/messages", post(mcp::transport::message_handler))
        // MCP Streamable HTTP transport (2025-03-26 spec)
        .route("/mcp", post(mcp::transport::streamable_mcp_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cfg.port);
    tracing::info!("Open Brain (Rust) listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
