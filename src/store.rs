use anyhow::Result;
use qdrant_client::qdrant::{
    vectors_config, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance,
    FieldType, HnswConfigDiffBuilder, PointStruct, UpsertPointsBuilder, VectorParamsBuilder,
    VectorsConfig,
};
use qdrant_client::Qdrant;
use serde_json::json;
use std::collections::HashMap;
use uuid::Uuid;

use crate::config::Config;

pub const COLLECTION: &str = "open_brain";
pub const DIM: u64 = 3072;

pub async fn ensure_collection(client: &Qdrant, cfg: &Config) -> Result<()> {
    let collections = client.list_collections().await?;
    if collections.collections.iter().any(|c| c.name == COLLECTION) {
        tracing::info!("Collection '{}' already exists", COLLECTION);
        return Ok(());
    }
    tracing::info!("Creating collection '{}' with dim={}", COLLECTION, DIM);
    client
        .create_collection(
            CreateCollectionBuilder::new(COLLECTION)
                .hnsw_config(
                    HnswConfigDiffBuilder::default()
                        .m(cfg.hnsw_m)
                        .ef_construct(cfg.hnsw_ef)
                        .build(),
                )
                .vectors_config(VectorsConfig {
                    config: Some(vectors_config::Config::Params(
                        VectorParamsBuilder::new(DIM, Distance::Cosine).build(),
                    )),
                }),
        )
        .await?;

    tracing::info!("Creating payload field indices");
    client
        .create_field_index(CreateFieldIndexCollectionBuilder::new(
            COLLECTION,
            "type",
            FieldType::Keyword,
        ))
        .await?;
    client
        .create_field_index(CreateFieldIndexCollectionBuilder::new(
            COLLECTION,
            "topics",
            FieldType::Keyword,
        ))
        .await?;
    client
        .create_field_index(CreateFieldIndexCollectionBuilder::new(
            COLLECTION,
            "people",
            FieldType::Keyword,
        ))
        .await?;
    client
        .create_field_index(CreateFieldIndexCollectionBuilder::new(
            COLLECTION,
            "created_at",
            FieldType::Integer,
        ))
        .await?;
    client
        .create_field_index(CreateFieldIndexCollectionBuilder::new(
            COLLECTION,
            "importance",
            FieldType::Integer,
        ))
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

fn build_point(id: Uuid, vector: Vec<f32>, payload: MemoryPayload) -> PointStruct {
    let mut pl: HashMap<String, serde_json::Value> = HashMap::new();
    pl.insert("content".into(), json!(payload.content));
    pl.insert("type".into(), json!(payload.memory_type));
    pl.insert("topics".into(), json!(payload.topics));
    pl.insert("people".into(), json!(payload.people));
    pl.insert("source".into(), json!(payload.source));
    pl.insert("importance".into(), json!(payload.importance));
    pl.insert("created_at".into(), json!(payload.created_at));
    PointStruct::new(id.to_string(), vector, pl)
}

pub async fn store_memory(
    client: &Qdrant,
    id: Uuid,
    vector: Vec<f32>,
    payload: MemoryPayload,
) -> Result<()> {
    client
        .upsert_points(UpsertPointsBuilder::new(
            COLLECTION,
            vec![build_point(id, vector, payload)],
        ))
        .await?;
    Ok(())
}

pub async fn store_memories_batch(
    client: &Qdrant,
    memories: Vec<(Uuid, Vec<f32>, MemoryPayload)>,
) -> Result<()> {
    if memories.is_empty() {
        return Ok(());
    }
    let points: Vec<PointStruct> = memories
        .into_iter()
        .map(|(id, vector, payload)| build_point(id, vector, payload))
        .collect();
    client
        .upsert_points(UpsertPointsBuilder::new(COLLECTION, points))
        .await?;
    Ok(())
}
