"""nexus tools — List all registered NEXUS tools."""

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def tools_list():
    """List all registered NEXUS tools and their metadata.

    Shows name, description, risk level, resource pattern,
    and timeout for every built-in tool.

    Example:
        nexus tools
    """
    from nexus.tools.registry import ToolRegistry
    from nexus.tools.plugin import get_registered_tools
    import nexus.tools.builtin  # noqa: F401 — triggers @tool registrations

    registry = ToolRegistry()
    for _name, (definition, impl) in get_registered_tools().items():
        registry.register(definition, impl)

    tool_defs = registry.list_tools()

    if not tool_defs:
        console.print("[yellow]No tools registered.[/yellow]")
        return

    risk_color = {"low": "green", "medium": "yellow", "high": "red"}

    table = Table(
        box=box.ROUNDED,
        header_style="bold dim",
        show_lines=False,
        title=f"[bold]{len(tool_defs)} Registered Tools[/bold]",
    )
    table.add_column("Name", style="cyan", width=20)
    table.add_column("Description", width=42)
    table.add_column("Risk", width=8)
    table.add_column("Resource Pattern", width=18, style="dim")
    table.add_column("Timeout", justify="right", width=9, style="dim")
    table.add_column("Approval?", width=10)

    for tool in sorted(tool_defs, key=lambda t: t.name):
        risk = tool.risk_level.value if hasattr(tool.risk_level, "value") else str(tool.risk_level)
        color = risk_color.get(risk.lower(), "white")
        requires_approval = getattr(tool, "requires_approval", False)
        table.add_row(
            tool.name,
            f"[dim]{tool.description}[/dim]",
            f"[{color}]{risk}[/{color}]",
            tool.resource_pattern or "*",
            f"{tool.timeout_seconds}s",
            "[yellow]yes[/yellow]" if requires_approval else "[dim]no[/dim]",
        )

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Add custom tools with [cyan]@tool[/cyan] decorator in nexus/tools/builtin/.[/dim]")
