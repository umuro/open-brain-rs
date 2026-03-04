use anyhow::{Context, Result};

pub struct Config {
    pub qdrant_url: String,
    pub gemini_api_key: String,
    pub port: u16,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let qdrant_url = std::env::var("QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6334".to_string());

        let gemini_api_key = std::env::var("GEMINI_API_KEY")
            .context("GEMINI_API_KEY must be set")?;

        let port = std::env::var("PORT")
            .unwrap_or_else(|_| "3737".to_string())
            .parse::<u16>()
            .context("PORT must be a valid u16")?;

        Ok(Self {
            qdrant_url,
            gemini_api_key,
            port,
        })
    }
}
