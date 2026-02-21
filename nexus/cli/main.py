"""NEXUS CLI — Typer application."""

import typer
from rich.console import Console

from nexus.version import __version__

app = typer.Typer(
    name="nexus",
    help="NEXUS — Notarized AI Execution. The agent framework where AI actions are accountable.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main(version: bool = typer.Option(False, "--version", "-v", help="Show version")):
    """NEXUS CLI."""
    if version:
        console.print(f"NEXUS v{__version__}")
        raise typer.Exit()


# Import and register command modules
from nexus.cli.commands import init, run, dev, seed

app.command()(init.init_project)
app.command(name="run")(run.run_task)
app.command()(dev.dev_server)
app.command()(seed.seed_db)


if __name__ == "__main__":
    app()
