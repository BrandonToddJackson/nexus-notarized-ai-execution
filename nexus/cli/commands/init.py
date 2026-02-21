"""nexus init — Scaffold a new NEXUS project."""

import os
import typer
from rich.console import Console

console = Console()


def init_project(
    name: str = typer.Argument("my-nexus-project", help="Project directory name"),
):
    """Scaffold a new NEXUS project directory.

    Creates:
    - .env with placeholder API keys
    - personas.yaml with 5 default personas
    - tools.yaml with built-in tool config
    - main.py entry point
    - docker-compose.yml for local development
    """
    # TODO: Implement — create directory, write template files
    console.print(f"[green]Creating NEXUS project: {name}[/green]")
    console.print("[yellow]Not yet implemented — see Phase 10 in BUILD_SPEC[/yellow]")
