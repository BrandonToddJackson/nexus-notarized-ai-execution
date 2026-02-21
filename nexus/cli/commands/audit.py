"""nexus audit — Export or view the ledger for compliance and review."""

import asyncio
import json
import typer
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


async def _audit(tenant_id: str, limit: int, export_path: str | None, fmt: str) -> None:
    from nexus.db.database import init_db, async_session
    from nexus.db.repository import Repository

    await init_db()

    async with async_session() as session:
        repo = Repository(session)
        seal_models = await repo.list_seals(tenant_id, limit=limit, offset=0)

    if not seal_models:
        console.print(f"[yellow]No seals found for tenant:[/yellow] {tenant_id}")
        return

    # Stats
    total = len(seal_models)
    executed = sum(1 for s in seal_models if s.status == "executed")
    blocked = sum(1 for s in seal_models if s.status == "blocked")
    failed = sum(1 for s in seal_models if s.status == "failed")

    if fmt == "json" or export_path:
        records = []
        for m in seal_models:
            records.append({
                "id": str(m.id),
                "chain_id": str(m.chain_id),
                "step_index": m.step_index,
                "tenant_id": m.tenant_id,
                "persona_id": m.persona_id,
                "tool_name": m.tool_name,
                "status": m.status,
                "fingerprint": m.fingerprint,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "completed_at": m.completed_at.isoformat() if m.completed_at else None,
                "anomaly_verdict": (m.anomaly_result or {}).get("overall_verdict"),
                "error": m.error,
            })

        payload = {
            "exported_at": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
            "total": total,
            "seals": records,
        }
        json_str = json.dumps(payload, indent=2)

        if export_path:
            with open(export_path, "w") as f:
                f.write(json_str)
            console.print(f"[green]Exported {total} seal(s) to[/green] {export_path}")
        else:
            console.print_json(json_str)
        return

    # Rich table view
    console.print()
    console.print(Panel(
        f"[bold]Tenant:[/bold] {tenant_id}  "
        f"[dim]{total} seal(s)[/dim]\n"
        f"[green]{executed} executed[/green]  "
        + (f"[red]{blocked} blocked[/red]  " if blocked else "")
        + (f"[yellow]{failed} failed[/yellow]" if failed else ""),
        title="[bold blue]NEXUS Audit Ledger[/bold blue]",
        border_style="blue",
    ))

    table = Table(
        box=box.ROUNDED,
        header_style="bold dim",
        show_lines=False,
    )
    table.add_column("#", width=4, justify="right", style="dim")
    table.add_column("Seal ID", width=14)
    table.add_column("Chain", width=14)
    table.add_column("Persona", width=14)
    table.add_column("Tool", width=18)
    table.add_column("Status", width=10)
    table.add_column("Verdict", width=8)
    table.add_column("Timestamp", width=20)

    status_color = {"executed": "green", "blocked": "red", "pending": "yellow", "failed": "red"}
    verdict_icon = {"pass": "[green]✓[/green]", "fail": "[red]✗[/red]"}

    for i, m in enumerate(seal_models, 1):
        st = m.status or "pending"
        color = status_color.get(st, "white")
        verdict = (m.anomaly_result or {}).get("overall_verdict", "")
        icon = verdict_icon.get(verdict.lower(), f"[dim]{verdict}[/dim]")
        ts = m.created_at.strftime("%Y-%m-%d %H:%M:%S") if m.created_at else ""

        table.add_row(
            str(i),
            f"[dim]{str(m.id)[:12]}…[/dim]",
            f"[dim]{str(m.chain_id)[:12]}…[/dim]",
            f"[magenta]{m.persona_id or ''}[/magenta]",
            f"[cyan]{m.tool_name or ''}[/cyan]",
            f"[{color}]{st}[/{color}]",
            icon,
            f"[dim]{ts}[/dim]",
        )

    console.print(table)
    console.print()
    console.print(f"[dim]Showing {min(total, limit)} of {total} seal(s). "
                  f"Use --limit N to see more, --export file.json for compliance export.[/dim]")


def audit_ledger(
    tenant: str = typer.Option("demo", "--tenant", "-t", help="Tenant ID"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max seals to show"),
    export: str = typer.Option(None, "--export", "-e", help="Export to file path (JSON)"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table or json"),
):
    """View or export the audit ledger — the immutable record of all agent actions.

    Every action executed by NEXUS agents is permanently recorded in the ledger
    as a cryptographically sealed record. This command surfaces that trail for
    compliance review, debugging, or export.

    Requires: PostgreSQL running with ledger data.

    Examples:
        nexus audit
        nexus audit --limit 100
        nexus audit --format json
        nexus audit --export report.json
        nexus audit --tenant my-tenant
    """
    try:
        asyncio.run(_audit(tenant, limit, export, fmt))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("[dim]Is the database running? Try: docker compose up postgres -d[/dim]")
        raise typer.Exit(1)
