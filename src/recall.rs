use anyhow::Result;
use qdrant_client::qdrant::{
    Condition, Filter, PointId, Range, ScrollPointsBuilder, SearchPointsBuilder,
    point_id::PointIdOptions,
    value::Kind,
};
use qdrant_client::Qdrant;
use std::collections::HashMap;

use crate::store::COLLECTION;

fn point_id_to_string(id: PointId) -> String {
    match id.point_id_options {
        Some(PointIdOptions::Uuid(u)) => u,
        Some(PointIdOptions::Num(n)) => n.to_string(),
        None => String::new(),
    }
}

pub struct RecallResult {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub topics: Vec<String>,
    pub people: Vec<String>,
    pub created_at: i64,
    pub score: f32,
}

pub struct BrainStats {
    pub total: u64,
    pub by_type: HashMap<String, u64>,
}

fn val_str<'a>(
    payload: &'a HashMap<String, qdrant_client::qdrant::Value>,
    key: &str,
    default: &'a str,
) -> &'a str {
    payload
        .get(key)
        .and_then(|v| v.kind.as_ref())
        .and_then(|k| {
            if let Kind::StringValue(s) = k { Some(s.as_str()) } else { None }
        })
        .unwrap_or(default)
}

fn val_i64(payload: &HashMap<String, qdrant_client::qdrant::Value>, key: &str) -> i64 {
    payload
        .get(key)
        .and_then(|v| v.kind.as_ref())
        .and_then(|k| {
            if let Kind::IntegerValue(n) = k { Some(*n) } else { None }
        })
        .unwrap_or(0)
}

fn val_strs(payload: &HashMap<String, qdrant_client::qdrant::Value>, key: &str) -> Vec<String> {
    payload
        .get(key)
        .and_then(|v| v.kind.as_ref())
        .and_then(|k| {
            if let Kind::ListValue(lv) = k { Some(lv) } else { None }
        })
        .map(|lv| {
            lv.values
                .iter()
                .filter_map(|v| {
                    if let Some(Kind::StringValue(s)) = &v.kind { Some(s.clone()) } else { None }
                })
                .collect()
        })
        .unwrap_or_default()
}

pub async fn semantic_recall(
    client: &Qdrant,
    vector: Vec<f32>,
    limit: u64,
    filter_type: Option<&str>,
) -> Result<Vec<RecallResult>> {
    let mut builder = SearchPointsBuilder::new(COLLECTION, vector, limit).with_payload(true);

    if let Some(t) = filter_type {
        builder = builder.filter(Filter::must([Condition::matches("type", t.to_string())]));
    }

    let results = client.search_points(builder).await?;

    Ok(results
        .result
        .into_iter()
        .map(|p| {
            let pl = &p.payload;
            RecallResult {
                id: p.id.map(point_id_to_string).unwrap_or_default(),
                content: val_str(pl, "content", "").to_string(),
                memory_type: val_str(pl, "type", "note").to_string(),
                topics: val_strs(pl, "topics"),
                people: val_strs(pl, "people"),
                created_at: val_i64(pl, "created_at"),
                score: p.score,
            }
        })
        .collect())
}

pub async fn list_recent(
    client: &Qdrant,
    days: i64,
    filter_type: Option<&str>,
    limit: u64,
) -> Result<Vec<RecallResult>> {
    let cutoff = chrono::Utc::now().timestamp() - days * 86400;

    let mut conditions = vec![Condition::range(
        "created_at",
        Range { gte: Some(cutoff as f64), ..Default::default() },
    )];

    if let Some(t) = filter_type {
        conditions.push(Condition::matches("type", t.to_string()));
    }

    let filter = Filter::must(conditions);
    let builder = ScrollPointsBuilder::new(COLLECTION)
        .filter(filter)
        .limit(limit as u32)
        .with_payload(true);

    let results = client.scroll(builder).await?;

    Ok(results
        .result
        .into_iter()
        .map(|p| {
            let pl = &p.payload;
            RecallResult {
                id: p.id.map(point_id_to_string).unwrap_or_default(),
                content: val_str(pl, "content", "").to_string(),
                memory_type: val_str(pl, "type", "note").to_string(),
                topics: val_strs(pl, "topics"),
                people: val_strs(pl, "people"),
                created_at: val_i64(pl, "created_at"),
                score: 0.0,
            }
        })
        .collect())
}

pub async fn brain_stats(client: &Qdrant) -> Result<BrainStats> {
    use qdrant_client::qdrant::CountPointsBuilder;

    let count_result = client.count(CountPointsBuilder::new(COLLECTION)).await?;
    let total = count_result.result.map(|r| r.count).unwrap_or(0);

    let mut by_type: HashMap<String, u64> = HashMap::new();
    let mut offset: Option<PointId> = None;

    loop {
        let mut builder = ScrollPointsBuilder::new(COLLECTION)
            .limit(100)
            .with_payload(true);
        if let Some(off) = offset {
            builder = builder.offset(off);
        }

        let results = client.scroll(builder).await?;
        let has_more = results.next_page_offset.is_some();

        for p in results.result {
            let t = val_str(&p.payload, "type", "note").to_string();
            *by_type.entry(t).or_insert(0) += 1;
        }

        if !has_more { break; }
        offset = results.next_page_offset;
    }

    Ok(BrainStats { total, by_type })
}
