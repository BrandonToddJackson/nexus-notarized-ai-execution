"""nexus config — Show resolved NEXUS configuration."""

import typer
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def config_show():
    """Show the resolved NEXUS configuration.

    Reads from environment variables and .env file.
    Sensitive values (secret_key, API keys) are masked.

    Example:
        nexus config
    """
    from nexus.config import NexusConfig
    cfg = NexusConfig()

    def mask(val: str) -> str:
        s = str(val)
        if len(s) <= 8:
            return "***"
        return s[:4] + "…" + "***"

    sensitive = {"secret_key", "llm_api_key"}

    table = Table(
        box=box.ROUNDED,
        header_style="bold dim",
        show_lines=False,
        title="[bold]NEXUS Configuration[/bold]",
    )
    table.add_column("Key", style="cyan", width=30)
    table.add_column("Value", width=55)
    table.add_column("Env Var", style="dim", width=35)

    sections = [
        ("App", [
            ("debug", "NEXUS_DEBUG"),
            ("log_level", "NEXUS_LOG_LEVEL"),
            ("secret_key", "NEXUS_SECRET_KEY"),
        ]),
        ("Database", [
            ("database_url", "NEXUS_DATABASE_URL"),
            ("redis_url", "NEXUS_REDIS_URL"),
        ]),
        ("LLM", [
            ("default_llm_model", "NEXUS_DEFAULT_LLM_MODEL"),
            ("llm_api_key", "ANTHROPIC_API_KEY / OPENAI_API_KEY"),
            ("llm_max_tokens", "NEXUS_LLM_MAX_TOKENS"),
            ("llm_temperature", "NEXUS_LLM_TEMPERATURE"),
        ]),
        ("Embeddings", [
            ("embedding_model", "NEXUS_EMBEDDING_MODEL"),
            ("embedding_dimensions", "NEXUS_EMBEDDING_DIMENSIONS"),
            ("chroma_persist_dir", "NEXUS_CHROMA_PERSIST_DIR"),
        ]),
        ("Anomaly Gates", [
            ("gate_intent_threshold", "NEXUS_GATE_INTENT_THRESHOLD"),
            ("gate_drift_sigma", "NEXUS_GATE_DRIFT_SIGMA"),
            ("gate_default_ttl", "NEXUS_GATE_DEFAULT_TTL"),
        ]),
        ("Budget & Limits", [
            ("default_budget_usd", "NEXUS_DEFAULT_BUDGET_USD"),
            ("rate_limit_requests_per_minute", "NEXUS_RATE_LIMIT_REQUESTS_PER_MINUTE"),
            ("rate_limit_chains_per_hour", "NEXUS_RATE_LIMIT_CHAINS_PER_HOUR"),
        ]),
        ("Server", [
            ("host", "NEXUS_HOST"),
            ("port", "NEXUS_PORT"),
            ("cors_origins", "NEXUS_CORS_ORIGINS"),
        ]),
    ]

    first = True
    for section_name, fields in sections:
        if not first:
            table.add_row("", "", "")
        first = False
        table.add_row(f"[bold dim]── {section_name} ──[/bold dim]", "", "")
        for attr, env_var in fields:
            val = getattr(cfg, attr, None)
            if val is None:
                display = "[dim](not set)[/dim]"
            elif attr in sensitive:
                display = mask(str(val))
            else:
                display = str(val)
            table.add_row(f"  {attr}", display, env_var)

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Source: environment variables + .env file (prefix: NEXUS_)[/dim]")
