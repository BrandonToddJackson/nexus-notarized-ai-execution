"""nexus sales-growth — Run the outbound sales growth agent cycle."""

import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def run_sales_growth(
    campaign_ids: str = typer.Option(..., "--campaigns", help="Comma-separated Instantly campaign IDs"),
    sheets_id: str = typer.Option(..., "--sheets-id", help="Google Sheets spreadsheet ID"),
    from_number: str = typer.Option(..., "--from-number", help="Retell phone number (E.164, e.g. +15551234567)"),
    interval: int = typer.Option(6, "--interval", help="Cycle interval in hours (default 6)"),
    once: bool = typer.Option(False, "--once", help="Run a single cycle then exit"),
):
    """Run the outbound sales growth agent cycle.

    Polls warm Instantly leads, enriches phone numbers, sends LinkedIn DMs,
    triggers Retell voice calls, and logs results to Google Sheets.

    Every action passes through the NEXUS 4-gate accountability pipeline.

    Examples:
        nexus sales-growth --campaigns abc123 --sheets-id 1BxiM --from-number +15551234567 --once
        nexus sales-growth --campaigns abc123,def456 --sheets-id 1BxiM --from-number +15551234567 --interval 4
    """
    try:
        asyncio.run(_run(campaign_ids.split(","), sheets_id, from_number, interval, once))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        raise typer.Exit(1)


async def _run(
    campaign_ids: list[str],
    sheets_id: str,
    from_number: str,
    interval_hours: int,
    once: bool,
) -> None:
    from nexus.cli.commands.run import _build_in_memory_engine
    from nexus.core.orchestrators.sales_growth import SalesGrowthOrchestrator

    config = {
        "campaign_ids": campaign_ids,
        "sheets_id": sheets_id,
        "retell_from_number": from_number,
        "interval_hours": interval_hours,
        "tenant_id": "cli-user",
    }

    console.print(
        f"[bold blue]NEXUS Sales Growth Agent[/bold blue]\n"
        f"  Campaigns : [cyan]{', '.join(campaign_ids)}[/cyan]\n"
        f"  Sheets ID : [cyan]{sheets_id}[/cyan]\n"
        f"  From      : [cyan]{from_number}[/cyan]\n"
        f"  Mode      : {'[yellow]single cycle[/yellow]' if once else f'[green]daemon ({interval_hours}h interval)[/green]'}"
    )
    console.print()

    with console.status("[dim]Initializing NEXUS engine...[/dim]"):
        engine = _build_in_memory_engine()

    orchestrator = SalesGrowthOrchestrator(engine, config)

    if once:
        with console.status("[blue]Running sales cycle...[/blue]"):
            result = await orchestrator.run_cycle()
        _print_cycle_result(result)
    else:
        console.print(f"[dim]Running every {interval_hours}h. Press Ctrl+C to stop.[/dim]")
        await orchestrator.run_forever()


def _print_cycle_result(result: dict) -> None:
    """Render a sales cycle result as a rich summary table."""
    table = Table(
        title="Sales Cycle Complete",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Metric", style="dim", width=22)
    table.add_column("Value", width=30)

    table.add_row("Leads processed", str(result.get("leads_count", 0)))

    linkedin = result.get("linkedin_result", {})
    table.add_row(
        "LinkedIn DMs sent",
        f"[green]{linkedin.get('sent', 0)}[/green] / [red]{linkedin.get('failed', 0)} failed[/red]",
    )

    retell = result.get("retell_result", {})
    batch_id = retell.get("batch_call_id") or "—"
    table.add_row("Retell batch ID", f"[cyan]{batch_id}[/cyan]")
    table.add_row("Calls scheduled", str(retell.get("scheduled_count", 0)))
    table.add_row("Timestamp", result.get("timestamp", "—"))

    console.print(table)
