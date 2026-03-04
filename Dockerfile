# Multi-stage build for open-brain Rust server
# Stage 1: Build
FROM rust:latest AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Cache dependencies layer — dummy src to compile deps first (bust: 2026-03-03)
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src/bin && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > src/bin/migrate.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src

# Build actual source
COPY src ./src
RUN touch src/main.rs src/bin/migrate.rs && \
    cargo build --release --bin open-brain

# Stage 2: Runtime (minimal)
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/open-brain .

ENV RUST_LOG=info
ENV PORT=3737

EXPOSE 3737

CMD ["./open-brain"]
