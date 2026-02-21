"""Application configuration. All env vars defined here with defaults."""

from pydantic_settings import BaseSettings
from typing import Optional


class NexusConfig(BaseSettings):
    # ── App ──
    app_name: str = "nexus"
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = "change-me-in-production"

    # ── Database ──
    database_url: str = "postgresql+asyncpg://nexus:nexus@localhost:5432/nexus"

    # ── Redis ──
    redis_url: str = "redis://localhost:6379/0"

    # ── LLM (litellm) ──
    default_llm_model: str = "anthropic/claude-sonnet-4-20250514"
    llm_api_key: Optional[str] = None          # set ANTHROPIC_API_KEY or OPENAI_API_KEY in env
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.1               # low temp for deterministic declarations

    # ── Embeddings ──
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    embedding_dimensions: int = 384

    # ── Vector Store ──
    chroma_persist_dir: str = "./data/chroma"

    # ── Anomaly Gates ──
    gate_intent_threshold: float = 0.75         # cosine similarity minimum
    gate_drift_sigma: float = 2.5               # standard deviations for drift detection
    gate_default_ttl: int = 120                 # seconds

    # ── Cost ──
    default_budget_usd: float = 50.0            # per tenant per month
    budget_alert_pct: float = 0.8               # alert at 80%

    # ── Auth ──
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    api_key_prefix: str = "nxs_"

    # ── Rate Limits ──
    rate_limit_requests_per_minute: int = 60
    rate_limit_chains_per_hour: int = 100

    # ── Server ──
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]  # Vite dev server

    model_config = {"env_prefix": "NEXUS_", "env_file": ".env", "extra": "ignore"}


config = NexusConfig()
