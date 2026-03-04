pub mod tools;
pub mod transport;

pub mod handlers {
    use axum::Json;
    use serde_json::{json, Value};

    pub async fn health() -> Json<Value> {
        Json(json!({ "status": "ok", "service": "open-brain-rs" }))
    }
}
