"""nexus dev — Start local development server."""

import typer
from rich.console import Console

console = Console()


def dev_server(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
):
    """Start the NEXUS API server in development mode with hot reload."""
    import uvicorn
    console.print(f"[green]Starting NEXUS dev server on {host}:{port}[/green]")
    uvicorn.run("nexus.api.main:app", host=host, port=port, reload=True)
