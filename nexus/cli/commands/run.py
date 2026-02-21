"""nexus run — Execute a task from the command line."""

import asyncio
import typer
from rich.console import Console

console = Console()


def run_task(
    task: str = typer.Argument(..., help="Task to execute"),
    persona: str = typer.Option(None, "--persona", "-p", help="Persona to use"),
):
    """Execute a task through the NEXUS pipeline and print the seal summary.

    Args:
        task: The task to execute (e.g., "What is NEXUS?")
        persona: Optional persona override
    """
    # TODO: Implement — create engine, run task, print seal summary
    console.print(f"[blue]Running task:[/blue] {task}")
    console.print("[yellow]Not yet implemented — see Phase 10 in BUILD_SPEC[/yellow]")
