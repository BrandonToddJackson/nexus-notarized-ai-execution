"""CLI commands for the NEXUS Plugin Marketplace.

Accessed via: ``nexus plugin <subcommand>``
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def _get_plugin_registry():
    """Create a headless PluginRegistry for CLI use.

    In the CLI context there is no running engine, so we pass
    ``tool_registry=None``.  Installed tools are registered when the API
    server starts via ``NexusEngine.load_plugins()``.
    """
    from nexus.config import config
    from nexus.marketplace import PluginRegistry
    return PluginRegistry(tool_registry=None, config=config)


# ─── install ─────────────────────────────────────────────────────────────────


def plugin_install(
    package: str = typer.Argument(
        ...,
        help="PyPI package name or short plugin name (e.g. 'weather' or 'nexus-plugin-weather'). "
             "May include ==version.",
    ),
    version: Optional[str] = typer.Option(
        None, "--version", "-v", help="Specific version to install."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Reinstall even if already installed."
    ),
):
    """Install a plugin from the NEXUS marketplace (PyPI).

    Examples:

        nexus plugin install weather

        nexus plugin install nexus-plugin-github==2.0.0

        nexus plugin install weather --version 1.2.0
    """
    async def _run():
        from nexus.marketplace.plugin_sdk import PluginError
        registry = _get_plugin_registry()
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(f"Installing {package}...", total=None)
                manifest = await registry.install(package, version=version, force=force)
        except PluginError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)

        console.print(
            f"[green]✓[/green] Installed [bold]{manifest.name}[/bold] v{manifest.version}"
        )
        if manifest.tools:
            table = Table("Tool", "Risk", "Description", show_header=True)
            for tool in manifest.tools:
                table.add_row(tool.name, tool.risk_level, tool.description[:60])
            console.print(table)

    asyncio.run(_run())


# ─── uninstall ────────────────────────────────────────────────────────────────


def plugin_uninstall(
    plugin_name: str = typer.Argument(
        ...,
        help="Plugin name (manifest.name, e.g. 'weather'). Not the pip package name.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt."
    ),
):
    """Uninstall a plugin and remove its tools.

    Example:

        nexus plugin uninstall weather
    """
    if not yes:
        typer.confirm(
            f"Uninstall plugin '{plugin_name}' and remove its tools?", abort=True
        )

    async def _run():
        from nexus.marketplace.plugin_sdk import PluginError
        registry = _get_plugin_registry()
        try:
            await registry.uninstall(plugin_name)
        except PluginError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)
        console.print(f"[green]✓[/green] Uninstalled [bold]{plugin_name}[/bold]")

    asyncio.run(_run())


# ─── list ─────────────────────────────────────────────────────────────────────


def plugin_list():
    """List all installed plugins."""

    async def _run():
        registry = _get_plugin_registry()
        await registry.load_state()
        plugins = registry.list_installed()

        if not plugins:
            console.print("[dim]No plugins installed.[/dim]")
            return

        table = Table("Name", "Version", "Tools", "Personas", "Installed At")
        for m in plugins:
            table.add_row(
                m.name,
                m.version,
                str(len(m.tools)),
                str(len(m.personas)),
                (m.installed_at or "")[:19],
            )
        console.print(table)

    asyncio.run(_run())


# ─── search ───────────────────────────────────────────────────────────────────


def plugin_search(
    query: str = typer.Argument(
        ..., help="Search term (e.g. 'slack', 'github', 'database')."
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results (1–100)."),
):
    """Search the NEXUS plugin marketplace (PyPI).

    Examples:

        nexus plugin search slack

        nexus plugin search github --limit 5
    """
    async def _run():
        registry = _get_plugin_registry()
        await registry.load_state()

        with Progress(
            SpinnerColumn(),
            TextColumn("Searching PyPI..."),
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            results = await registry.search(query, limit=limit)

        if not results:
            console.print(f"[dim]No plugins found for '{query}'.[/dim]")
            return

        table = Table("Package", "Version", "Description", "Installed")
        for r in results:
            installed_badge = "[green]✓[/green]" if r["installed"] else ""
            table.add_row(
                r["name"],
                r["version"],
                r["description"][:60],
                installed_badge,
            )
        console.print(table)

    asyncio.run(_run())


# ─── upgrade ──────────────────────────────────────────────────────────────────


def plugin_upgrade(
    plugin_name: str = typer.Argument(..., help="Plugin name to upgrade."),
    version: Optional[str] = typer.Option(
        None, "--version", "-v", help="Specific target version."
    ),
):
    """Upgrade an installed plugin to the latest (or a specific) version.

    Examples:

        nexus plugin upgrade weather

        nexus plugin upgrade weather --version 2.0.0
    """
    async def _run():
        from nexus.marketplace.plugin_sdk import PluginError
        registry = _get_plugin_registry()
        await registry.load_state()
        try:
            old_version = registry.get(plugin_name).version
        except PluginError:
            console.print(f"[red]Error:[/red] Plugin '{plugin_name}' is not installed.")
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn(f"Upgrading {plugin_name}..."),
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            try:
                manifest = await registry.upgrade(plugin_name, target_version=version)
            except PluginError as exc:
                console.print(f"[red]Error:[/red] {exc}")
                raise typer.Exit(1)

        console.print(
            f"[green]✓[/green] Upgraded [bold]{plugin_name}[/bold] "
            f"from v{old_version} → v{manifest.version}"
        )

    asyncio.run(_run())


# ─── new ──────────────────────────────────────────────────────────────────────


def plugin_new(
    name: str = typer.Argument(
        ...,
        help="Plugin name (lowercase, hyphens). Example: my-tool",
    ),
    output_dir: Path = typer.Option(
        Path("."), "--output", "-o", help="Directory to create the package in."
    ),
):
    """Scaffold a new plugin package from the template.

    Creates a ready-to-publish ``nexus-plugin-<name>/`` directory with
    ``__init__.py``, ``tools.py``, ``personas.yaml``, ``pyproject.toml``,
    and ``README.md``.

    Example:

        nexus plugin new my-weather-tool
    """
    from nexus.marketplace.scaffolder import scaffold_plugin
    try:
        package_path = scaffold_plugin(name, output_dir)
        console.print(
            f"[green]✓[/green] Plugin scaffold created at "
            f"[bold]{package_path}[/bold]"
        )
        console.print(
            "\nNext steps:\n"
            f"  1. Edit [bold]{package_path.name}/{package_path.name.replace('-','_')}/tools.py[/bold]\n"
            f"  2. Update [bold]plugin_manifest[/bold] in __init__.py\n"
            f"  3. [bold]pip install -e .[/bold] to test locally\n"
            f"  4. [bold]nexus plugin install nexus-plugin-{name}[/bold] after publishing to PyPI"
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


# ─── verify ───────────────────────────────────────────────────────────────────


def plugin_verify(
    plugin_name: Optional[str] = typer.Argument(
        None, help="Plugin to verify. If omitted, verifies all installed plugins."
    ),
):
    """Recompute and compare SHA-256 checksums of installed plugins.

    Detects tampering introduced after initial install (supply-chain update
    attacks, e.g. the 2024 Ultralytics incident pattern).

    Example:

        nexus plugin verify weather

        nexus plugin verify
    """
    async def _run():
        registry = _get_plugin_registry()
        await registry.load_state()

        if plugin_name:
            from nexus.marketplace.plugin_sdk import PluginNotFoundError
            try:
                targets = [registry.get(plugin_name)]
            except PluginNotFoundError:
                console.print(f"[red]Error:[/red] Plugin '{plugin_name}' is not installed.")
                raise typer.Exit(1)
        else:
            targets = registry.list_installed()

        if not targets:
            console.print("[dim]No plugins installed.[/dim]")
            return

        all_ok = True
        for manifest in targets:
            mod_name = f"nexus_plugin_{manifest.name.replace('-', '_')}"
            current_checksum = await registry._compute_package_checksum(mod_name)

            if not manifest.checksum_sha256:
                console.print(
                    f"[yellow]?[/yellow] {manifest.name}: no stored checksum "
                    f"(installed before Phase 27)"
                )
                continue

            if current_checksum == manifest.checksum_sha256:
                console.print(
                    f"[green]✓[/green] {manifest.name} v{manifest.version}: checksum OK"
                )
            else:
                all_ok = False
                console.print(
                    f"[red bold]✗ TAMPERED[/red bold] {manifest.name}: "
                    f"stored={manifest.checksum_sha256[:16]}... "
                    f"current={current_checksum[:16]}... — UNINSTALL IMMEDIATELY"
                )

        if not all_ok:
            raise typer.Exit(1)

    asyncio.run(_run())
