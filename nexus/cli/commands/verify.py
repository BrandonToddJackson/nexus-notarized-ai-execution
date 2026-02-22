"""nexus verify — Verify Merkle chain integrity from the terminal.

This is a NEXUS differentiator: no other agent framework lets you
cryptographically verify that an execution audit trail was not tampered with.
"""

import asyncio
from datetime import datetime, timezone
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


async def _verify(chain_id: str, tenant_id: str) -> None:
    from nexus.db.database import init_db, async_session
    from nexus.db.repository import Repository
    from nexus.core.notary import Notary

    await init_db()

    async with async_session() as session:
        repo = Repository(session)
        seal_models = await repo.get_chain_seals(chain_id)

    if not seal_models:
        console.print(f"[red]No chain found:[/red] {chain_id}")
        raise typer.Exit(1)

    # Convert DB models → Seal objects for verification
    from nexus.types import (
        Seal, IntentDeclaration, AnomalyResult, ActionStatus, GateVerdict, RiskLevel,
    )

    seals = []
    for m in sorted(seal_models, key=lambda s: s.step_index):
        m_intent: dict = dict(m.intent) if m.intent else {}
        m_anomaly: dict = dict(m.anomaly_result) if m.anomaly_result else {}
        seals.append(Seal(
            id=str(m.id),
            chain_id=str(m.chain_id),
            step_index=int(m.step_index),
            tenant_id=str(m.tenant_id),
            persona_id=str(m.persona_id),
            intent=IntentDeclaration(**m_intent) if m_intent else IntentDeclaration(
                task_description="", planned_action="", tool_name=str(m.tool_name or ""),
                tool_params={}, resource_targets=[], reasoning="",
            ),
            anomaly_result=AnomalyResult(**m_anomaly) if m_anomaly else AnomalyResult(
                gates=[], overall_verdict=GateVerdict.PASS, risk_level=RiskLevel.LOW, persona_id="", action_fingerprint="",
            ),
            tool_name=str(m.tool_name or ""),
            tool_params=dict(m.tool_params) if m.tool_params else {},
            tool_result=m.tool_result,
            status=ActionStatus(m.status) if m.status else ActionStatus.PENDING,
            cot_trace=list(m.cot_trace) if m.cot_trace else [],
            fingerprint=str(m.fingerprint or ""),
            parent_fingerprint=str(m.parent_fingerprint or ""),
            created_at=m.created_at if isinstance(m.created_at, datetime) else datetime.now(timezone.utc),
            completed_at=m.completed_at if isinstance(m.completed_at, datetime) else None,
            error=str(m.error) if m.error else None,
        ))

    from nexus.exceptions import SealIntegrityError
    notary = Notary()
    integrity_error: str | None = None
    with console.status("[dim]Verifying Merkle chain...[/dim]"):
        try:
            notary.verify_chain(seals)
            intact = True
        except SealIntegrityError as exc:
            intact = False
            integrity_error = str(exc)

    # Build the chain visualization table
    table = Table(box=box.ROUNDED, header_style="bold dim", show_lines=True)
    table.add_column("#", width=4, justify="right")
    table.add_column("Seal ID", width=16)
    table.add_column("Tool", width=18)
    table.add_column("Status", width=10)
    table.add_column("Fingerprint", width=20)
    table.add_column("Parent FP", width=20)

    for seal in seals:
        status = seal.status.value if hasattr(seal.status, "value") else str(seal.status)
        status_color = {"executed": "green", "blocked": "red", "pending": "yellow"}.get(status, "white")
        table.add_row(
            str(seal.step_index + 1),
            f"[dim]{seal.id[:14]}…[/dim]",
            f"[cyan]{seal.tool_name}[/cyan]",
            f"[{status_color}]{status}[/{status_color}]",
            f"[dim]{seal.fingerprint[:18]}…[/dim]",
            f"[dim]{seal.parent_fingerprint[:18] if seal.parent_fingerprint else '(genesis)'}[/dim]",
        )

    if intact:
        verdict_panel = Panel(
            "[bold green]✓ CHAIN INTACT[/bold green]\n\n"
            f"All {len(seals)} seal(s) verified. Merkle fingerprints are unbroken.\n"
            "[dim]No tampering detected.[/dim]",
            border_style="green",
            title="[bold]Integrity Verification[/bold]",
        )
    else:
        detail = f"\n[dim]{integrity_error}[/dim]" if integrity_error else ""
        verdict_panel = Panel(
            "[bold red]✗ CHAIN COMPROMISED[/bold red]\n\n"
            f"Fingerprint mismatch detected in {len(seals)} seal(s).\n"
            "[dim]One or more seals may have been tampered with.[/dim]"
            + detail,
            border_style="red",
            title="[bold]Integrity Verification[/bold]",
        )

    console.print()
    console.print(f"[bold]Chain:[/bold] [dim]{chain_id}[/dim]  "
                  f"[dim]{len(seals)} seal(s)[/dim]")
    console.print()
    console.print(table)
    console.print()
    console.print(verdict_panel)

    if not intact:
        raise typer.Exit(1)


def verify_chain(
    chain_id: str = typer.Argument(..., help="Chain ID to verify"),
    tenant: str = typer.Option("demo", "--tenant", "-t", help="Tenant ID"),
):
    """Verify Merkle chain integrity — cryptographic proof of tamper-free execution.

    Each seal in a NEXUS chain contains a fingerprint derived from:
        SHA256(previous_fingerprint + SHA256(seal_content))

    Any modification to a historical seal breaks all subsequent fingerprints.
    This command re-derives the full chain and confirms every link is intact.

    Requires: PostgreSQL running with sealed chain data.

    Example:
        nexus verify abc-123-chain-id
    """
    try:
        asyncio.run(_verify(chain_id, tenant))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("[dim]Is the database running? Try: docker compose up postgres -d[/dim]")
        raise typer.Exit(1)
