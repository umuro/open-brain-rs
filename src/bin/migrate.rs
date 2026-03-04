//! Migration tool: reads SOURCE_DIR/**/*.md, chunks by ## header,
//! embeds with Gemini, upserts to Qdrant.
//!
//! Usage:
//!   SOURCE_DIR=./docs QDRANT_URL=http://localhost:6334 GEMINI_API_KEY=xxx cargo run --bin migrate

use anyhow::{Context, Result};
use chrono::Utc;
use qdrant_client::Qdrant;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

const COLLECTION: &str = "open_brain";
const DIM: u64 = 3072;
const MIN_CHUNK_LEN: usize = 30;

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    content: EmbedContent,
}

#[derive(Serialize)]
struct EmbedContent {
    parts: Vec<EmbedPart>,
}

#[derive(Serialize)]
struct EmbedPart {
    text: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: EmbedValues,
}

#[derive(Deserialize)]
struct EmbedValues {
    values: Vec<f32>,
}

async fn embed_text(client: &Client, api_key: &str, text: &str) -> Result<Vec<f32>> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={}",
        api_key
    );
    let req = EmbedRequest {
        model: "models/gemini-embedding-001".into(),
        content: EmbedContent {
            parts: vec![EmbedPart {
                text: text.chars().take(8000).collect(),
            }],
        },
    };

    for attempt in 0..5u32 {
        let res = client.post(&url).json(&req).send().await?;
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

fn chunk_markdown(content: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in content.lines() {
        if line.starts_with("## ") && !current.trim().is_empty() {
            let trimmed = current.trim().to_string();
            if trimmed.len() >= MIN_CHUNK_LEN {
                chunks.push(trimmed);
            }
            current = line.to_string();
            current.push('\n');
        } else {
            current.push_str(line);
            current.push('\n');
        }
    }

    let trimmed = current.trim().to_string();
    if trimmed.len() >= MIN_CHUNK_LEN {
        chunks.push(trimmed);
    }

    chunks
}

async fn collect_md_files(source_dir: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut stack = vec![PathBuf::from(source_dir)];

    while let Some(dir) = stack.pop() {
        let mut entries = tokio::fs::read_dir(&dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().map(|e| e == "md").unwrap_or(false) {
                files.push(path);
            }
        }
    }
    Ok(files)
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let source_dir = std::env::var("SOURCE_DIR").context("SOURCE_DIR env var required")?;
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
    let gemini_api_key =
        std::env::var("GEMINI_API_KEY").context("GEMINI_API_KEY env var required")?;

    tracing::info!("Migrating from SOURCE_DIR={}", source_dir);

    // Setup Qdrant
    let qdrant = Qdrant::from_url(&qdrant_url).build()?;

    // Ensure collection exists
    use qdrant_client::qdrant::{
        vectors_config, CreateCollectionBuilder, Distance, VectorParamsBuilder, VectorsConfig,
    };
    let collections = qdrant.list_collections().await?;
    if !collections.collections.iter().any(|c| c.name == COLLECTION) {
        qdrant
            .create_collection(CreateCollectionBuilder::new(COLLECTION).vectors_config(
                VectorsConfig {
                    config: Some(vectors_config::Config::Params(
                        VectorParamsBuilder::new(DIM, Distance::Cosine).build(),
                    )),
                },
            ))
            .await?;
        tracing::info!("Created collection '{}'", COLLECTION);
    }

    let http_client = Client::new();

    let md_files = collect_md_files(&source_dir).await?;
    tracing::info!("Found {} markdown files", md_files.len());

    let mut total_chunks = 0usize;
    let mut upserted = 0usize;

    for file_path in &md_files {
        let content = tokio::fs::read_to_string(file_path).await?;
        let source = file_path.to_string_lossy().to_string();
        let chunks = chunk_markdown(&content);

        tracing::info!("{}: {} chunks", source, chunks.len());

        for chunk in chunks {
            total_chunks += 1;
            let vector = embed_text(&http_client, &gemini_api_key, &chunk).await?;
            let id = Uuid::new_v4();

            use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder};
            use serde_json::json;
            use std::collections::HashMap;

            let mut payload: HashMap<String, serde_json::Value> = HashMap::new();
            payload.insert("content".into(), json!(chunk));
            payload.insert("type".into(), json!("note"));
            payload.insert("topics".into(), json!([]));
            payload.insert("people".into(), json!([]));
            payload.insert("source".into(), json!(source));
            payload.insert("importance".into(), json!(5u8));
            payload.insert("created_at".into(), json!(Utc::now().timestamp()));

            qdrant
                .upsert_points(UpsertPointsBuilder::new(
                    COLLECTION,
                    vec![PointStruct::new(id.to_string(), vector, payload)],
                ))
                .await?;

            upserted += 1;

            // Small delay to avoid hammering Gemini API
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    tracing::info!(
        "Migration complete: {} files, {} chunks processed, {} upserted",
        md_files.len(),
        total_chunks,
        upserted
    );

    Ok(())
}
