"""Application configuration + declarative YAML config loader for NEXUS.

All env vars defined here with NEXUS_ prefix.
YAML loaders: load_personas_yaml(), load_tools_yaml()
"""

import warnings
from pydantic_settings import BaseSettings
from typing import Optional

from nexus.config.loader import load_personas_yaml, load_tools_yaml
from nexus.config.schema import PersonaYAML, ToolYAML, PersonasConfig, ToolsConfig


class NexusConfig(BaseSettings):
    # ── App ──
    app_name: str = "nexus"
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = "change-me-in-production-nexus-insecure-default-key"

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

    # ── Workflows ──
    max_workflow_steps: int = 50
    max_concurrent_workflows: int = 10
    workflow_execution_timeout: int = 3600          # 1 hour max per workflow run

    # ── Triggers ──
    webhook_base_url: str = "http://localhost:8000"  # public base URL for webhook URLs
    cron_check_interval: int = 15                    # seconds between cron checks
    max_triggers_per_workflow: int = 5

    # ── Credentials ──
    credential_encryption_key: str = ""              # 32-byte Fernet key
    credential_max_per_tenant: int = 100

    # ── MCP ──
    mcp_connection_timeout: int = 10                 # seconds
    mcp_tool_timeout: int = 60                       # per-tool execution timeout
    mcp_max_servers: int = 20                        # per tenant

    # ── Code Sandbox ──
    sandbox_max_memory_mb: int = 256
    sandbox_max_execution_seconds: int = 30
    sandbox_allowed_imports: list[str] = [
        "json", "math", "re", "datetime", "collections",
        "itertools", "functools", "hashlib", "base64",
        "urllib.parse", "csv", "io",
    ]

    # ── Background Execution ──
    worker_concurrency: int = 4
    task_queue_url: str = "redis://localhost:6379/1"  # separate DB from cache

    model_config = {"env_prefix": "NEXUS_", "env_file": ".env", "extra": "ignore"}


config = NexusConfig()

if config.secret_key == "change-me-in-production-nexus-insecure-default-key":
    warnings.warn("NEXUS_SECRET_KEY is insecure default — set a strong value in .env", stacklevel=1)
elif len(config.secret_key.encode()) < 32:
    warnings.warn(
        f"NEXUS_SECRET_KEY is {len(config.secret_key.encode())} bytes — "
        "minimum 32 required for HS256 (RFC 7518 §3.2)", stacklevel=1
    )


__all__ = [
    "NexusConfig",
    "config",
    "load_personas_yaml",
    "load_tools_yaml",
    "PersonaYAML",
    "ToolYAML",
    "PersonasConfig",
    "ToolsConfig",
]
