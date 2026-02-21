"""nexus inspect — Full seal details: intent, 4 gates, fingerprint, CoT."""

import asyncio
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import json

console = Console()


async def _inspect(seal_id: str, tenant_id: str, raw: bool) -> None:
    from nexus.db.database import init_db, async_session
    from nexus.db.models import SealModel
    from sqlalchemy import select

    await init_db()

    async with async_session() as session:
        result = await session.execute(
            select(SealModel).where(SealModel.id == seal_id)
        )
        seal_model = result.scalar_one_or_none()

    if seal_model is None:
        console.print(f"[red]Seal not found:[/red] {seal_id}")
        raise typer.Exit(1)

    if raw:
        data = {
            "id": str(seal_model.id),
            "chain_id": str(seal_model.chain_id),
            "step_index": seal_model.step_index,
            "tenant_id": seal_model.tenant_id,
            "persona_id": seal_model.persona_id,
            "tool_name": seal_model.tool_name,
            "tool_params": seal_model.tool_params,
            "tool_result": str(seal_model.tool_result) if seal_model.tool_result else None,
            "status": seal_model.status,
            "intent": seal_model.intent,
            "anomaly_result": seal_model.anomaly_result,
            "cot_trace": seal_model.cot_trace,
            "fingerprint": seal_model.fingerprint,
            "parent_fingerprint": seal_model.parent_fingerprint,
            "created_at": seal_model.created_at.isoformat() if seal_model.created_at else None,
            "completed_at": seal_model.completed_at.isoformat() if seal_model.completed_at else None,
            "error": seal_model.error,
        }
        console.print_json(json.dumps(data, indent=2))
        return

    status = seal_model.status or "pending"
    status_color = {"executed": "green", "blocked": "red", "pending": "yellow"}.get(status, "white")

    # Header
    console.print()
    console.print(Panel(
        f"[bold]ID:[/bold]      [dim]{seal_model.id}[/dim]\n"
        f"[bold]Chain:[/bold]   [dim]{seal_model.chain_id}[/dim]  "
        f"[dim](step {seal_model.step_index + 1})[/dim]\n"
        f"[bold]Tenant:[/bold]  [dim]{seal_model.tenant_id}[/dim]\n"
        f"[bold]Persona:[/bold] [magenta]{seal_model.persona_id}[/magenta]\n"
        f"[bold]Status:[/bold]  [{status_color}]{status.upper()}[/{status_color}]",
        title="[bold blue]Seal Inspection[/bold blue]",
        border_style="blue",
    ))

    # Intent declaration
    intent = seal_model.intent or {}
    if intent:
        console.print()
        console.print("[bold]Intent Declaration[/bold]")
        intent_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        intent_table.add_column("Field", style="dim", width=18)
        intent_table.add_column("Value")
        for field, label in [
            ("tool_name", "Tool"),
            ("planned_action", "Planned action"),
            ("reasoning", "Reasoning"),
        ]:
            val = intent.get(field, "")
            if val:
                intent_table.add_row(label, str(val)[:120])
        if intent.get("resource_targets"):
            intent_table.add_row("Resources", ", ".join(intent["resource_targets"]))
        if intent.get("confidence") is not None:
            intent_table.add_row("Confidence", f"{intent['confidence']:.2f}")
        console.print(intent_table)

    # Gate results
    anomaly = seal_model.anomaly_result or {}
    gates = anomaly.get("gates", [])
    if gates:
        console.print()
        console.print("[bold]Anomaly Gate Results[/bold]")
        gate_table = Table(box=box.ROUNDED, header_style="bold dim", show_lines=False)
        gate_table.add_column("Gate", width=10)
        gate_table.add_column("Verdict", width=10)
        gate_table.add_column("Score", justify="right", width=8)
        gate_table.add_column("Threshold", justify="right", width=10)
        gate_table.add_column("Details")

        icons = {"pass": "[green]✓ pass[/green]", "fail": "[red]✗ fail[/red]", "warn": "[yellow]⚠ warn[/yellow]"}
        for gate in gates:
            verdict = gate.get("verdict", "pass")
            gate_table.add_row(
                gate.get("gate_name", ""),
                icons.get(verdict.lower(), verdict),
                f"{gate.get('score', 0):.3f}",
                f"{gate.get('threshold', 0):.3f}",
                f"[dim]{gate.get('details', '')}[/dim]",
            )

        overall = anomaly.get("overall_verdict", "pass")
        overall_color = "green" if overall == "pass" else "red"
        console.print(gate_table)
        console.print(f"[bold]Overall verdict:[/bold] [{overall_color}]{overall.upper()}[/{overall_color}]  "
                      f"[dim]risk: {anomaly.get('risk_level', 'unknown')}[/dim]")

    # Chain-of-thought
    cot_trace = seal_model.cot_trace or []
    if cot_trace:
        console.print()
        console.print("[bold]Chain-of-Thought[/bold]")
        for i, entry in enumerate(cot_trace, 1):
            console.print(f"  [dim]{i}.[/dim] {entry}")

    # Merkle fingerprints
    console.print()
    console.print("[bold]Merkle Fingerprints[/bold]")
    fp_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    fp_table.add_column("Label", style="dim", width=18)
    fp_table.add_column("Value", style="dim")
    fp_table.add_row("Fingerprint", seal_model.fingerprint or "(none)")
    fp_table.add_row("Parent fingerprint", seal_model.parent_fingerprint or "(genesis)")
    console.print(fp_table)

    # Error
    if seal_model.error:
        console.print()
        console.print(Panel(
            f"[red]{seal_model.error}[/red]",
            title="[red bold]Error[/red bold]",
            border_style="red",
        ))


def inspect_seal(
    seal_id: str = typer.Argument(..., help="Seal ID to inspect"),
    tenant: str = typer.Option("demo", "--tenant", "-t", help="Tenant ID"),
    raw: bool = typer.Option(False, "--raw", "-r", help="Output raw JSON"),
):
    """Show full details for a single seal — intent, gates, CoT, and fingerprint.

    Every NEXUS action produces a seal containing the complete decision record:
    what the agent intended, how each gate evaluated it, the reasoning trace,
    and the cryptographic fingerprint that chains it to all other seals.

    Requires: PostgreSQL running with sealed data.

    Example:
        nexus inspect <seal-id>
        nexus inspect <seal-id> --raw
    """
    try:
        asyncio.run(_inspect(seal_id, tenant, raw))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("[dim]Is the database running? Try: docker compose up postgres -d[/dim]")
        raise typer.Exit(1)
