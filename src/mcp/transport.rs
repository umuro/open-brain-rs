use axum::{
    extract::{Query, State},
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use dashmap::DashMap;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::mcp::tools::{handle_tool_call, tools_list};
use crate::AppState;

pub type SessionMap = Arc<DashMap<String, mpsc::Sender<String>>>;

#[derive(Deserialize)]
pub struct SessionQuery {
    #[serde(rename = "sessionId")]
    pub session_id: Option<String>,
}

#[derive(Deserialize, Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

#[derive(Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

impl JsonRpcResponse {
    fn ok(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }
    fn err(id: Option<Value>, code: i64, message: &str) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(json!({ "code": code, "message": message })),
        }
    }
}

pub async fn sse_handler(
    State(state): State<AppState>,
    query: Query<SessionQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let session_id = query
        .session_id
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let (tx, rx) = mpsc::channel::<String>(64);
    state.sessions.insert(session_id.clone(), tx);

    // Send the endpoint event so clients know where to POST messages
    let endpoint_msg = format!(
        "{{\"type\":\"endpoint\",\"endpoint\":\"/messages?sessionId={}\"}}",
        session_id
    );

    let stream = async_stream::stream! {
        // Send endpoint info immediately
        yield Ok(Event::default().event("endpoint").data(endpoint_msg));

        let mut rx_stream = ReceiverStream::new(rx);
        use tokio_stream::StreamExt;
        while let Some(msg) = rx_stream.next().await {
            yield Ok(Event::default().event("message").data(msg));
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

pub async fn message_handler(
    State(state): State<AppState>,
    query: Query<SessionQuery>,
    Json(body): Json<JsonRpcRequest>,
) -> Json<Value> {
    let response = dispatch_rpc(body, &state).await;
    let payload = serde_json::to_string(&response).unwrap_or_default();

    // Also push to SSE stream if session exists
    if let Some(session_id) = &query.session_id {
        if let Some(tx) = state.sessions.get(session_id) {
            let _ = tx.try_send(payload.clone());
        }
    }

    Json(serde_json::to_value(&response).unwrap_or(json!({})))
}

async fn dispatch_rpc(req: JsonRpcRequest, state: &AppState) -> JsonRpcResponse {
    match req.method.as_str() {
        "initialize" => JsonRpcResponse::ok(
            req.id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "open-brain-rs",
                    "version": "0.1.0"
                }
            }),
        ),
        "tools/list" => JsonRpcResponse::ok(req.id, json!({ "tools": tools_list() })),
        "tools/call" => {
            let params = req.params.unwrap_or(json!({}));
            let name = params["name"].as_str().unwrap_or("").to_string();
            let args = params["arguments"].clone();

            match handle_tool_call(&name, args, state).await {
                Ok(result) => JsonRpcResponse::ok(
                    req.id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&result).unwrap_or_default()
                        }],
                        "isError": false
                    }),
                ),
                Err(e) => JsonRpcResponse::ok(
                    req.id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": format!("Error: {}", e)
                        }],
                        "isError": true
                    }),
                ),
            }
        }
        "notifications/initialized" => JsonRpcResponse::ok(req.id, json!({})),
        _ => JsonRpcResponse::err(req.id, -32601, "Method not found"),
    }
}
