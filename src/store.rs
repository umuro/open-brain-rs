use anyhow::Result;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, UpsertPointsBuilder, VectorParamsBuilder,
    VectorsConfig, vectors_config,
};
use qdrant_client::Qdrant;
use serde_json::json;
use std::collections::HashMap;

pub const COLLECTION: &str = "open_brain";
pub const DIM: u64 = 3072;

pub async fn ensure_collection(client: &Qdrant) -> Result<()> {
    let collections = client.list_collections().await?;
    if collections.collections.iter().any(|c| c.name == COLLECTION) {
        tracing::info!("Collection '{}' already exists", COLLECTION);
        return Ok(());
    }
    tracing::info!("Creating collection '{}' with dim={}", COLLECTION, DIM);
    client
        .create_collection(
            CreateCollectionBuilder::new(COLLECTION).vectors_config(VectorsConfig {
                config: Some(vectors_config::Config::Params(
                    VectorParamsBuilder::new(DIM, Distance::Cosine).build(),
                )),
            }),
        )
        .await?;
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
    let mut pl: HashMap<String, serde_json::Value> = HashMap::new();
    pl.insert("content".into(), json!(payload.content));
    pl.insert("type".into(), json!(payload.memory_type));
    pl.insert("topics".into(), json!(payload.topics));
    pl.insert("people".into(), json!(payload.people));
    pl.insert("source".into(), json!(payload.source));
    pl.insert("importance".into(), json!(payload.importance));
    pl.insert("created_at".into(), json!(payload.created_at));

    client
        .upsert_points(UpsertPointsBuilder::new(
            COLLECTION,
            vec![PointStruct::new(id.to_string(), vector, pl)],
        ))
        .await?;
    Ok(())
}
