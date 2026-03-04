use axum::{extract::State, routing::{get, post}, Json, Router};
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
}

async fn rest_store(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    match mcp::tools::handle_tool_call("remember", body, &state).await {
        Ok(v) => Json(v),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

async fn rest_recall(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cfg = config::Config::from_env()?;

    let qdrant = Arc::new(
        qdrant_client::Qdrant::from_url(&cfg.qdrant_url)
            .build()?,
    );
    store::ensure_collection(&qdrant).await?;

    let embedder = Arc::new(embed::Embedder::new(cfg.gemini_api_key.clone()));
    let sessions: SessionMap = Arc::new(dashmap::DashMap::new());

    let state = AppState {
        qdrant,
        embedder,
        sessions,
    };

    let app = Router::new()
        // REST convenience endpoints
        .route("/health", get(mcp::handlers::health))
        .route("/store",  post(rest_store))
        .route("/recall", post(rest_recall))
        .route("/stats",  get(rest_stats))
        // MCP SSE transport
        .route("/sse",      get(mcp::transport::sse_handler))
        .route("/messages", post(mcp::transport::message_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cfg.port);
    tracing::info!("Open Brain (Rust) listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
