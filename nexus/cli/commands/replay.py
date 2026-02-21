"""nexus replay — Step-by-step chain replay with gate results and CoT.

Replays a historical chain execution showing each step's:
- Declared intent
- 4 gate results (scope / intent / TTL / drift)
- Action status (executed / blocked)
- Chain-of-thought reasoning trace
"""

import asyncio
import time
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import box

console = Console()

_GATE_ICONS = {
    "pass": "[bold green]✓[/bold green]",
    "fail": "[bold red]✗[/bold red]",
    "warn": "[bold yellow]⚠[/bold yellow]",
}


def _gate_row(gate) -> tuple:
    verdict = gate.verdict.value if hasattr(gate.verdict, "value") else str(gate.verdict)
    return (
        gate.gate_name,
        _GATE_ICONS.get(verdict.lower(), verdict),
        f"{gate.score:.3f} / {gate.threshold:.3f}",
        f"[dim]{gate.details}[/dim]",
    )


async def _replay(chain_id: str, delay: float) -> None:
    from nexus.db.database import init_db, async_session
    from nexus.db.repository import Repository

    await init_db()

    async with async_session() as session:
        repo = Repository(session)
        seal_models = await repo.get_chain_seals(chain_id)

    if not seal_models:
        console.print(f"[red]No chain found:[/red] {chain_id}")
        raise typer.Exit(1)

    seals = sorted(seal_models, key=lambda s: s.step_index)
    total = len(seals)

    console.print()
    console.print(Panel(
        f"[bold]Chain:[/bold] [dim]{chain_id}[/dim]\n"
        f"[bold]Steps:[/bold] {total}",
        title="[bold blue]NEXUS Chain Replay[/bold blue]",
        border_style="blue",
    ))

    for i, m in enumerate(seals):
        console.print()
        console.print(Rule(f"[bold]Step {i + 1} / {total}[/bold]", style="dim"))

        status: str = str(m.status) if m.status else "pending"
        status_color = {"executed": "green", "blocked": "red", "pending": "yellow"}.get(status, "white")

        # Intent declaration
        intent: dict = dict(m.intent) if m.intent else {}
        console.print(f"[bold]Tool:[/bold]    [cyan]{m.tool_name}[/cyan]")
        console.print(f"[bold]Persona:[/bold] [magenta]{m.persona_id}[/magenta]")
        if intent.get("planned_action"):
            console.print(f"[bold]Intent:[/bold]  [dim]{intent['planned_action']}[/dim]")
        if intent.get("reasoning"):
            console.print(f"[bold]Reason:[/bold]  [dim]{intent['reasoning'][:120]}[/dim]")

        console.print()

        # Gate results table
        anomaly: dict = dict(m.anomaly_result) if m.anomaly_result else {}
        gates = anomaly.get("gates", [])
        if gates:
            gate_table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim", padding=(0, 1))
            gate_table.add_column("Gate", width=8)
            gate_table.add_column("", width=4)
            gate_table.add_column("Score / Threshold", width=20)
            gate_table.add_column("Details")

            for gate_data in gates:
                verdict = gate_data.get("verdict", "pass")
                icon = _GATE_ICONS.get(verdict.lower(), verdict)
                score = gate_data.get("score", 0.0)
                threshold = gate_data.get("threshold", 0.0)
                gate_table.add_row(
                    gate_data.get("gate_name", ""),
                    icon,
                    f"{score:.3f} / {threshold:.3f}",
                    f"[dim]{gate_data.get('details', '')}[/dim]",
                )
            console.print(gate_table)

        # Status
        console.print(f"[bold]Status:[/bold]  [{status_color}]{status.upper()}[/{status_color}]")

        # CoT trace
        cot_trace: list = list(m.cot_trace) if m.cot_trace else []
        if cot_trace:
            console.print()
            console.print("[bold dim]Chain-of-Thought:[/bold dim]")
            for j, entry in enumerate(cot_trace, 1):
                console.print(f"  [dim]{j}.[/dim] {entry}")

        # Tool result preview
        if m.tool_result and status == "executed":
            result_str = str(m.tool_result)
            preview = result_str[:200] + ("…" if len(result_str) > 200 else "")
            console.print()
            console.print(f"[bold dim]Result preview:[/bold dim] [dim]{preview}[/dim]")

        if delay > 0 and i < total - 1:
            time.sleep(delay)

    console.print()
    console.print(Rule("[dim]End of replay[/dim]", style="dim"))


def replay_chain(
    chain_id: str = typer.Argument(..., help="Chain ID to replay"),
    delay: float = typer.Option(0.0, "--delay", "-d", help="Pause between steps (seconds)"),
):
    """Replay a chain execution step by step — intent, gates, CoT, and outcome.

    Shows the full decision trail for each step:
    - What the agent declared it intended to do
    - How all 4 anomaly gates evaluated the action
    - What the chain-of-thought reasoning contained
    - Whether the action was executed or blocked

    Requires: PostgreSQL running with sealed chain data.

    Example:
        nexus replay abc-123-chain-id
        nexus replay abc-123-chain-id --delay 1.0
    """
    try:
        asyncio.run(_replay(chain_id, delay))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("[dim]Is the database running? Try: docker compose up postgres -d[/dim]")
        raise typer.Exit(1)
