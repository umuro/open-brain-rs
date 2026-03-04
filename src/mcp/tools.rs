use anyhow::Result;
use chrono::Utc;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::store::{store_memory, MemoryPayload};
use crate::recall::{semantic_recall, list_recent, brain_stats};
use crate::AppState;

pub async fn handle_tool_call(name: &str, args: Value, state: &AppState) -> Result<Value> {
    match name {
        "remember" => tool_remember(args, state).await,
        "recall" => tool_recall(args, state).await,
        "list_recent" => tool_list_recent(args, state).await,
        "brain_stats" => tool_brain_stats(state).await,
        _ => Ok(json!({ "error": format!("Unknown tool: {}", name) })),
    }
}

async fn tool_remember(args: Value, state: &AppState) -> Result<Value> {
    let content = args["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let memory_type = args["type"]
        .as_str()
        .unwrap_or("note")
        .to_string();
    let topics: Vec<String> = args["topics"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).map(String::from).collect())
        .unwrap_or_default();
    let people: Vec<String> = args["people"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).map(String::from).collect())
        .unwrap_or_default();
    let source = args["source"]
        .as_str()
        .unwrap_or("mcp")
        .to_string();
    let importance = args["importance"]
        .as_u64()
        .unwrap_or(5) as u8;

    if content.is_empty() {
        return Ok(json!({ "error": "content is required" }));
    }

    let vector = state.embedder.embed(&content).await?;
    let id = Uuid::new_v4();
    let created_at = Utc::now().timestamp();

    store_memory(
        &state.qdrant,
        id,
        vector,
        MemoryPayload {
            content: content.clone(),
            memory_type,
            topics,
            people,
            source,
            importance,
            created_at,
        },
    )
    .await?;

    Ok(json!({
        "id": id.to_string(),
        "content": content,
        "created_at": created_at,
        "status": "stored"
    }))
}

async fn tool_recall(args: Value, state: &AppState) -> Result<Value> {
    let query = args["query"].as_str().unwrap_or("").to_string();
    let limit = args["limit"].as_u64().unwrap_or(10);
    let filter_type = args["type"].as_str().map(String::from);

    if query.is_empty() {
        return Ok(json!({ "error": "query is required" }));
    }

    let vector = state.embedder.embed(&query).await?;
    let results = semantic_recall(
        &state.qdrant,
        vector,
        limit,
        filter_type.as_deref(),
    )
    .await?;

    Ok(json!(results.iter().map(|r| json!({
        "id": r.id,
        "content": r.content,
        "type": r.memory_type,
        "topics": r.topics,
        "people": r.people,
        "created_at": r.created_at,
        "distance": r.score
    })).collect::<Vec<_>>()))
}

async fn tool_list_recent(args: Value, state: &AppState) -> Result<Value> {
    let days = args["days"].as_i64().unwrap_or(7);
    let filter_type = args["type"].as_str().map(String::from);
    let limit = args["limit"].as_u64().unwrap_or(50);

    let results = list_recent(
        &state.qdrant,
        days,
        filter_type.as_deref(),
        limit,
    )
    .await?;

    Ok(json!(results.iter().map(|r| json!({
        "id": r.id,
        "content": r.content,
        "type": r.memory_type,
        "topics": r.topics,
        "people": r.people,
        "created_at": r.created_at
    })).collect::<Vec<_>>()))
}

async fn tool_brain_stats(state: &AppState) -> Result<Value> {
    let stats = brain_stats(&state.qdrant).await?;
    Ok(json!({
        "total": stats.total,
        "by_type": stats.by_type
    }))
}

pub fn tools_list() -> Value {
    json!([
        {
            "name": "remember",
            "description": "Store a memory in the brain",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "The content to remember" },
                    "type": { "type": "string", "description": "Memory type (note, fact, task, event, etc.)", "default": "note" },
                    "topics": { "type": "array", "items": { "type": "string" }, "description": "Topics/tags" },
                    "people": { "type": "array", "items": { "type": "string" }, "description": "People mentioned" },
                    "source": { "type": "string", "description": "Source of the memory", "default": "mcp" },
                    "importance": { "type": "integer", "description": "Importance 1-10", "default": 5 }
                },
                "required": ["content"]
            }
        },
        {
            "name": "recall",
            "description": "Semantically recall memories by query",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Semantic search query" },
                    "limit": { "type": "integer", "description": "Max results", "default": 10 },
                    "type": { "type": "string", "description": "Filter by memory type" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "list_recent",
            "description": "List recent memories",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "days": { "type": "integer", "description": "Number of days back", "default": 7 },
                    "type": { "type": "string", "description": "Filter by memory type" },
                    "limit": { "type": "integer", "description": "Max results", "default": 50 }
                }
            }
        },
        {
            "name": "brain_stats",
            "description": "Get brain statistics",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }
    ])
}
