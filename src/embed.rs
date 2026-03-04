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
        Self {
            client: Client::new(),
            api_key,
        }
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={}",
            self.api_key
        );

        let req = EmbedRequest {
            model: "models/gemini-embedding-001".into(),
            content: Content {
                parts: vec![Part {
                    text: text.chars().take(8000).collect(),
                }],
            },
        };

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
