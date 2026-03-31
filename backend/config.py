"""
GraphoLab Backend — Application Settings.

All values can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = "GraphoLab API"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── Security / JWT ────────────────────────────────────────────────────────
    secret_key: str = "CHANGE_ME_in_production_use_openssl_rand_hex_32"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # ── Database (PostgreSQL) ─────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://grapholab:grapholab@localhost:5432/grapholab"

    # ── MinIO (S3-compatible storage) ─────────────────────────────────────────
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "grapholab"
    minio_secret_key: str = "grapholab123"
    minio_bucket: str = "grapholab-docs"
    minio_secure: bool = False

    # ── AI model paths (mirrors Gradio demo defaults) ─────────────────────────
    signet_weights: Path = Path("data/signet.pth")
    writer_samples_dir: Path = Path("data/samples")
    rag_cache_dir: Path = Path("data/rag_cache")

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # ── CORS (comma-separated origins for the React frontend) ─────────────────
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]


settings = Settings()
