"""nexus seed — Seed database with default data."""

import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


async def _seed() -> None:
    from nexus.db.database import init_db, async_session
    from nexus.db.seed import run_seed
    from nexus.db.repository import Repository

    with console.status("[dim]Connecting to database...[/dim]"):
        await init_db()

    async with async_session() as session:
        repo = Repository(session)

        with console.status("[dim]Seeding...[/dim]"):
            await run_seed(session)

        # Verify what's in the DB now
        tenant = await repo.get_tenant("demo")
        personas = await repo.list_personas("demo")

    console.print()
    console.print("[bold green]Database seeded successfully![/bold green]")
    console.print()

    # Tenant summary
    if tenant:
        console.print(f"[bold]Tenant:[/bold]  {tenant.name} [dim](id: {tenant.id})[/dim]")
        console.print("[bold]API Key:[/bold] [cyan]nxs_demo_key_12345[/cyan]  [dim](demo)[/dim]")
        console.print()

    # Persona table
    if personas:
        table = Table(
            box=box.ROUNDED,
            header_style="bold dim",
            show_lines=False,
            title=f"[bold]{len(personas)} Personas[/bold]",
        )
        table.add_column("Name", style="cyan", width=16)
        table.add_column("Tools", width=50)
        table.add_column("TTL", justify="right", width=6)
        table.add_column("Risk", width=8)

        for p in personas:
            tools = ", ".join(p.allowed_tools or [])
            table.add_row(
                p.name,
                f"[dim]{tools}[/dim]",
                str(p.max_ttl_seconds),
                p.risk_tolerance,
            )
        console.print(table)

    console.print()
    console.print("[dim]Run [cyan]nexus run \"What is NEXUS?\"[/cyan] to test.[/dim]")


def seed_db():
    """Seed the database with demo tenant, personas, and tools.

    Creates:
    - Demo tenant with API key nxs_demo_key_12345
    - 5 default personas (researcher, analyst, creator, communicator, operator)

    Requires: PostgreSQL running and NEXUS_DATABASE_URL set in .env
    """
    try:
        asyncio.run(_seed())
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("[dim]Is the database running? Try: docker compose up postgres -d[/dim]")
        raise typer.Exit(1)
