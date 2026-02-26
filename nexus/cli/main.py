"""NEXUS CLI — Typer application."""

import typer
from rich.console import Console

from nexus.version import __version__

app = typer.Typer(
    name="nexus",
    help="NEXUS — Notarized AI Execution. The agent framework where AI actions are accountable.",
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", is_eager=True, help="Show version"),
):
    """NEXUS CLI."""
    if version:
        console.print(f"NEXUS v{__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# ── Core commands ──────────────────────────────────────────────────────────────
from nexus.cli.commands import init, run, dev, seed  # noqa: E402

app.command(name="init")(init.init_project)
app.command(name="run")(run.run_task)
app.command(name="dev")(dev.dev_server)
app.command(name="seed")(seed.seed_db)

# ── NEXUS differentiators ──────────────────────────────────────────────────────
from nexus.cli.commands import verify, replay, inspect, audit, gates  # noqa: E402

app.command(name="verify", help="Verify Merkle chain integrity")(verify.verify_chain)
app.command(name="replay", help="Step-by-step chain replay with gate results + CoT")(replay.replay_chain)
app.command(name="inspect", help="Full seal details: intent, gates, fingerprint, CoT")(inspect.inspect_seal)
app.command(name="audit", help="View or export the audit ledger")(audit.audit_ledger)
app.command(name="gates", help="Show gate thresholds and pass/fail statistics")(gates.gates_status)

# ── Parity commands ────────────────────────────────────────────────────────────
from nexus.cli.commands import config, tools  # noqa: E402

app.command(name="config", help="Show resolved configuration")(config.config_show)
app.command(name="tools", help="List all registered tools")(tools.tools_list)

# ── Plugin Marketplace (Phase 27) ──────────────────────────────────────────────
from nexus.cli.commands import plugin as plugin_cmd  # noqa: E402

plugin_app = typer.Typer(name="plugin", help="Plugin marketplace commands.")
plugin_app.command("install", help="Install a plugin from PyPI")(plugin_cmd.plugin_install)
plugin_app.command("uninstall", help="Uninstall a plugin")(plugin_cmd.plugin_uninstall)
plugin_app.command("list", help="List installed plugins")(plugin_cmd.plugin_list)
plugin_app.command("search", help="Search the NEXUS plugin marketplace")(plugin_cmd.plugin_search)
plugin_app.command("upgrade", help="Upgrade an installed plugin")(plugin_cmd.plugin_upgrade)
plugin_app.command("new", help="Scaffold a new plugin package")(plugin_cmd.plugin_new)
plugin_app.command("verify", help="Verify plugin checksums (tamper detection)")(plugin_cmd.plugin_verify)
app.add_typer(plugin_app)


if __name__ == "__main__":
    app()
