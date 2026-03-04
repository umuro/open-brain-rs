use anyhow::Result;
use lru::LruCache;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Semaphore};

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
    cache: Arc<Mutex<LruCache<String, Vec<f32>>>>,
    semaphore: Arc<Semaphore>,
}

impl Embedder {
    pub fn new(api_key: String, cache_size: usize, concurrency: usize) -> Self {
        let client = Client::builder()
            .pool_max_idle_per_host(32)
            .tcp_keepalive(Duration::from_secs(60))
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build reqwest client");

        let cache = Arc::new(Mutex::new(LruCache::new(
            NonZeroUsize::new(cache_size).expect("cache_size must be > 0"),
        )));

        Self {
            client,
            api_key,
            cache,
            semaphore: Arc::new(Semaphore::new(concurrency)),
        }
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let cache_key: String = text.chars().take(200).collect();

        {
            let mut cache = self.cache.lock().await;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }

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
                tokio::time::sleep(Duration::from_millis(wait)).await;
                continue;
            }
            let body: EmbedResponse = res.error_for_status()?.json().await?;
            let vector = body.embedding.values;

            {
                let mut cache = self.cache.lock().await;
                cache.put(cache_key, vector.clone());
            }

            return Ok(vector);
        }
        anyhow::bail!("Gemini 429 after 5 retries")
    }

    pub async fn embed_batch(&self, texts: &[&str]) -> Vec<Result<Vec<f32>>> {
        let futs = texts.iter().map(|&text| async move {
            let _permit = self
                .semaphore
                .acquire()
                .await
                .map_err(|e| anyhow::anyhow!("semaphore closed: {}", e))?;
            self.embed(text).await
        });
        futures::future::join_all(futs).await
    }
}
