"""nexus seed — Seed database with default data."""

import asyncio
import typer
from rich.console import Console

console = Console()


def seed_db():
    """Seed the database with demo tenant, personas, and tools."""
    # TODO: Implement — call seed_database() from nexus.db.seed
    console.print("[green]Seeding database...[/green]")
    console.print("[yellow]Not yet implemented — see Phase 5 in BUILD_SPEC[/yellow]")
