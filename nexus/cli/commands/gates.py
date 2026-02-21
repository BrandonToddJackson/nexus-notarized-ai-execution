"""nexus gates — Show anomaly gate thresholds and pass/fail statistics."""

import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


async def _gates_stats(tenant_id: str) -> dict:
    """Query DB for gate-level pass/fail statistics across all seals."""
    try:
        from nexus.db.database import init_db, async_session
        from nexus.db.repository import Repository

        await init_db()

        async with async_session() as session:
            repo = Repository(session)
            seals = await repo.list_seals(tenant_id, limit=1000)

        gate_names = ["scope", "intent", "ttl", "drift"]
        stats: dict[str, dict] = {g: {"pass": 0, "fail": 0, "warn": 0} for g in gate_names}
        total_seals = len(seals)

        for seal_model in seals:
            anomaly = seal_model.anomaly_result or {}
            for gate in anomaly.get("gates", []):
                name = gate.get("gate_name", "")
                verdict = gate.get("verdict", "pass").lower()
                if name in stats:
                    stats[name][verdict] = stats[name].get(verdict, 0) + 1

        return {"stats": stats, "total": total_seals, "db_available": True}

    except Exception:
        return {"db_available": False}


def gates_status(
    tenant: str = typer.Option("demo", "--tenant", "-t", help="Tenant ID for statistics"),
    stats: bool = typer.Option(False, "--stats", "-s", help="Show pass/fail statistics from DB"),
):
    """Show anomaly gate thresholds and (optionally) historical pass/fail rates.

    NEXUS enforces 4 security gates before any agent action executes:

    \b
    Gate 1 — Scope:  Is this tool in the persona's allowed_tools list?
    Gate 2 — Intent: Cosine similarity ≥ threshold vs persona intent_patterns
    Gate 3 — TTL:    Has the persona been active longer than max_ttl_seconds?
    Gate 4 — Drift:  Is this action statistically anomalous vs history? (σ)

    If ANY gate fails, the action is BLOCKED and sealed as blocked in the ledger.

    Use --stats to pull historical pass/fail rates from the database.

    Example:
        nexus gates
        nexus gates --stats
        nexus gates --stats --tenant my-tenant
    """
    from nexus.config import NexusConfig
    cfg = NexusConfig()

    console.print()

    # Gate configuration table
    config_table = Table(
        box=box.ROUNDED,
        header_style="bold dim",
        show_lines=False,
        title="[bold]Gate Configuration[/bold]",
    )
    config_table.add_column("Gate", width=8)
    config_table.add_column("Name", width=10)
    config_table.add_column("Mechanism", width=35)
    config_table.add_column("Threshold", justify="right", width=12)
    config_table.add_column("Config Key", style="dim", width=30)

    gate_info = [
        (
            "Gate 1",
            "Scope",
            "Tool in persona.allowed_tools?",
            "[green]allowlist[/green]",
            "persona.allowed_tools",
        ),
        (
            "Gate 2",
            "Intent",
            "Cosine similarity ≥ threshold",
            f"[cyan]{cfg.gate_intent_threshold:.2f}[/cyan]",
            "NEXUS_GATE_INTENT_THRESHOLD",
        ),
        (
            "Gate 3",
            "TTL",
            "Persona active < max_ttl_seconds",
            f"[cyan]{cfg.gate_default_ttl}s[/cyan]",
            "NEXUS_GATE_DEFAULT_TTL",
        ),
        (
            "Gate 4",
            "Drift",
            "Action within σ of historical baseline",
            f"[cyan]{cfg.gate_drift_sigma}σ[/cyan]",
            "NEXUS_GATE_DRIFT_SIGMA",
        ),
    ]

    for gate, name, mechanism, threshold, config_key in gate_info:
        config_table.add_row(gate, f"[bold]{name}[/bold]", mechanism, threshold, config_key)

    console.print(config_table)

    # Additional config context
    console.print()
    console.print(Panel(
        f"[bold]LLM model:[/bold]        [dim]{cfg.default_llm_model}[/dim]\n"
        f"[bold]Embedding model:[/bold]  [dim]{cfg.embedding_model} ({cfg.embedding_dimensions}d)[/dim]\n"
        f"[bold]Budget limit:[/bold]     [dim]${cfg.default_budget_usd:.2f} / tenant / month[/dim]\n"
        f"[bold]Rate limit:[/bold]       [dim]{cfg.rate_limit_requests_per_minute} req/min, "
        f"{cfg.rate_limit_chains_per_hour} chains/hour[/dim]",
        title="[bold]Runtime Configuration[/bold]",
        border_style="dim",
    ))

    if not stats:
        console.print()
        console.print("[dim]Use [cyan]--stats[/cyan] to show historical pass/fail rates from the database.[/dim]")
        return

    # Pull stats from DB
    with console.status("[dim]Querying gate statistics...[/dim]"):
        result = asyncio.run(_gates_stats(tenant))

    if not result.get("db_available"):
        console.print()
        console.print("[yellow]Database unavailable — cannot show statistics.[/yellow]")
        console.print("[dim]Is the database running? Try: docker compose up postgres -d[/dim]")
        return

    db_stats = result["stats"]
    total_seals = result["total"]

    if total_seals == 0:
        console.print()
        console.print(f"[dim]No seals found for tenant '{tenant}'.[/dim]")
        return

    console.print()
    stats_table = Table(
        box=box.ROUNDED,
        header_style="bold dim",
        show_lines=False,
        title=f"[bold]Gate Statistics[/bold] [dim]({total_seals} seals, tenant: {tenant})[/dim]",
    )
    stats_table.add_column("Gate", width=10)
    stats_table.add_column("Pass", justify="right", width=8)
    stats_table.add_column("Fail", justify="right", width=8)
    stats_table.add_column("Pass Rate", justify="right", width=12)
    stats_table.add_column("Bar", width=30)

    for gate_name in ["scope", "intent", "ttl", "drift"]:
        s = db_stats.get(gate_name, {})
        passed = s.get("pass", 0)
        failed = s.get("fail", 0)
        checked = passed + failed
        rate = passed / checked if checked > 0 else 1.0
        bar_width = 20
        filled = round(rate * bar_width)
        bar_color = "green" if rate >= 0.95 else ("yellow" if rate >= 0.8 else "red")
        bar = f"[{bar_color}]{'█' * filled}[/{bar_color}][dim]{'░' * (bar_width - filled)}[/dim]"

        stats_table.add_row(
            f"[bold]{gate_name}[/bold]",
            f"[green]{passed}[/green]",
            f"[red]{failed}[/red]" if failed else "[dim]0[/dim]",
            f"[bold]{rate * 100:.1f}%[/bold]",
            bar,
        )

    console.print(stats_table)
