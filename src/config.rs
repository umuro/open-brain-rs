use anyhow::{Context, Result};

pub struct Config {
    pub qdrant_url: String,
    pub gemini_api_key: String,
    pub port: u16,
    pub embed_cache_size: usize,
    pub embed_concurrency: usize,
    pub hnsw_ef: u64,
    pub hnsw_m: u64,
    pub stats_cache_ttl_secs: u64,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let qdrant_url =
            std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());

        let gemini_api_key =
            std::env::var("GEMINI_API_KEY").context("GEMINI_API_KEY must be set")?;

        let port = std::env::var("PORT")
            .unwrap_or_else(|_| "3737".to_string())
            .parse::<u16>()
            .context("PORT must be a valid u16")?;

        let embed_cache_size = std::env::var("EMBED_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000_usize);

        let embed_concurrency = std::env::var("EMBED_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8_usize);

        let hnsw_ef = std::env::var("HNSW_EF")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256_u64);

        let hnsw_m = std::env::var("HNSW_M")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32_u64);

        let stats_cache_ttl_secs = std::env::var("STATS_CACHE_TTL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60_u64);

        Ok(Self {
            qdrant_url,
            gemini_api_key,
            port,
            embed_cache_size,
            embed_concurrency,
            hnsw_ef,
            hnsw_m,
            stats_cache_ttl_secs,
        })
    }
}
