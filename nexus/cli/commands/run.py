"""nexus run — Execute a task from the command line."""

import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

_GATE_ICONS = {
    "pass": "[bold green]✓ pass[/bold green]",
    "fail": "[bold red]✗ fail[/bold red]",
    "warn": "[bold yellow]⚠ warn[/bold yellow]",
}
_STATUS_COLOR = {
    "executed": "green",
    "blocked": "red",
    "pending": "yellow",
    "failed": "red",
}


def _build_in_memory_engine():
    """Build NexusEngine with in-memory components — no DB or Redis required."""
    from nexus.config import NexusConfig
    from nexus.types import PersonaContract, RiskLevel
    from nexus.core.personas import PersonaManager
    from nexus.core.anomaly import AnomalyEngine
    from nexus.core.notary import Notary
    from nexus.core.ledger import Ledger
    from nexus.core.chain import ChainManager
    from nexus.core.verifier import IntentVerifier
    from nexus.core.output_validator import OutputValidator
    from nexus.core.cot_logger import CoTLogger
    from nexus.knowledge.embeddings import EmbeddingService
    from nexus.knowledge.store import KnowledgeStore
    from nexus.knowledge.context import ContextBuilder
    from nexus.tools.registry import ToolRegistry
    from nexus.tools.plugin import get_registered_tools
    import nexus.tools.builtin  # noqa: F401 — triggers @tool registrations
    from nexus.tools.selector import ToolSelector
    from nexus.tools.sandbox import Sandbox
    from nexus.tools.executor import ToolExecutor
    from nexus.reasoning.think_act import ThinkActGate
    from nexus.reasoning.continue_complete import ContinueCompleteGate
    from nexus.reasoning.escalate import EscalateGate
    from nexus.core.engine import NexusEngine
    from nexus.db.seed import DEFAULT_PERSONAS

    cfg = NexusConfig()

    # Personas: prefer personas.yaml in cwd (nexus init projects), fall back to defaults
    try:
        from nexus.config.loader import load_personas_yaml
        persona_contracts = load_personas_yaml()
    except FileNotFoundError:
        # No personas.yaml in cwd — build from embedded seed defaults
        persona_contracts = []
        for p in DEFAULT_PERSONAS:
            try:
                risk = RiskLevel(p["risk_tolerance"])
            except ValueError:
                risk = RiskLevel.MEDIUM
            persona_contracts.append(PersonaContract(
                name=p["name"],
                description=p["description"],
                allowed_tools=p["allowed_tools"],
                resource_scopes=p["resource_scopes"],
                intent_patterns=p["intent_patterns"],
                max_ttl_seconds=p["max_ttl_seconds"],
                risk_tolerance=risk,
            ))
    persona_manager = PersonaManager(persona_contracts)

    with console.status("[dim]Loading embedding model...[/dim]"):
        embedding_service = EmbeddingService(model_name=cfg.embedding_model)

    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="nexus_cli_")
    knowledge_store = KnowledgeStore(
        persist_dir=tmp_dir,
        embedding_fn=embedding_service.embed,
    )

    # fingerprint_store=None disables Gate 4 drift check (no Redis in CLI mode)
    anomaly_engine = AnomalyEngine(
        config=cfg,
        embedding_service=embedding_service,
        fingerprint_store=None,
    )

    notary = Notary()
    ledger = Ledger()
    chain_manager = ChainManager()
    verifier = IntentVerifier()
    output_validator = OutputValidator()
    cot_logger = CoTLogger()
    context_builder = ContextBuilder(knowledge_store=knowledge_store)

    tool_registry = ToolRegistry()
    for _name, (definition, impl) in get_registered_tools().items():
        tool_registry.register(definition, impl)

    tool_selector = ToolSelector(registry=tool_registry)
    sandbox = Sandbox()
    tool_executor = ToolExecutor(registry=tool_registry, sandbox=sandbox, verifier=verifier)

    return NexusEngine(
        persona_manager=persona_manager,
        anomaly_engine=anomaly_engine,
        notary=notary,
        ledger=ledger,
        chain_manager=chain_manager,
        context_builder=context_builder,
        tool_registry=tool_registry,
        tool_selector=tool_selector,
        tool_executor=tool_executor,
        output_validator=output_validator,
        cot_logger=cot_logger,
        think_act_gate=ThinkActGate(),
        continue_complete_gate=ContinueCompleteGate(),
        escalate_gate=EscalateGate(),
        config=cfg,
    )


def _print_seal(seal, index: int) -> None:
    """Render a single seal as a rich panel with gate table + CoT."""
    status = seal.status.value if hasattr(seal.status, "value") else str(seal.status)
    color = _STATUS_COLOR.get(status, "white")

    gate_table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim", padding=(0, 1))
    gate_table.add_column("Gate", style="dim", width=8)
    gate_table.add_column("Result", width=14)
    gate_table.add_column("Score", width=7, justify="right")
    gate_table.add_column("Details")

    for gate in seal.anomaly_result.gates:
        verdict = gate.verdict.value if hasattr(gate.verdict, "value") else str(gate.verdict)
        icon = _GATE_ICONS.get(verdict.lower(), verdict)
        gate_table.add_row(
            gate.gate_name,
            icon,
            f"{gate.score:.3f}",
            f"[dim]{gate.details}[/dim]",
        )

    tool_str = f"[cyan]{seal.tool_name}[/cyan]"
    if seal.tool_params:
        param_preview = ", ".join(
            f"[dim]{k}[/dim]={repr(v)[:40]}" for k, v in list(seal.tool_params.items())[:2]
        )
        tool_str += f"({param_preview})"

    title = (
        f"[bold]Step {index + 1}[/bold]  "
        f"{tool_str}  "
        f"[{color}]{status.upper()}[/{color}]"
    )
    subtitle = f"[dim]seal:{seal.id[:12]}…  fp:{seal.fingerprint[:12]}…[/dim]"

    console.print(Panel(gate_table, title=title, subtitle=subtitle, border_style=color))

    if seal.cot_trace:
        console.print("[bold dim]  Chain-of-Thought:[/bold dim]")
        for i, entry in enumerate(seal.cot_trace, 1):
            console.print(f"    [dim]{i}.[/dim] {entry}")
        console.print()


async def _execute(task: str, persona: str | None) -> None:
    engine = _build_in_memory_engine()

    with console.status(f"[blue]Executing:[/blue] {task}"):
        chain = await engine.run(task, "cli-user", persona)

    seals = await engine.ledger.get_chain(chain.id)
    status = chain.status.value if hasattr(chain.status, "value") else str(chain.status)
    color = _STATUS_COLOR.get(status, "white")

    executed = sum(1 for s in seals if (s.status.value if hasattr(s.status, "value") else str(s.status)) == "executed")
    blocked = sum(1 for s in seals if (s.status.value if hasattr(s.status, "value") else str(s.status)) == "blocked")

    summary = (
        f"[bold]Task:[/bold] {chain.task}\n"
        f"[bold]Chain:[/bold] [dim]{chain.id}[/dim]\n"
        f"[bold]Status:[/bold] [{color}]{status.upper()}[/{color}]  "
        f"[dim]{len(seals)} seal(s) — "
        f"[green]{executed} executed[/green]"
        + (f", [red]{blocked} blocked[/red]" if blocked else "")
        + "[/dim]"
    )
    console.print()
    console.print(Panel(summary, title="[bold blue]NEXUS Execution Summary[/bold blue]", border_style="blue"))

    for i, seal in enumerate(seals):
        _print_seal(seal, i)

    if not seals:
        console.print("[dim]No seals recorded.[/dim]")


def run_task(
    task: str = typer.Argument(..., help="Task to execute"),
    persona: str = typer.Option(None, "--persona", "-p", help="Persona name to use"),
):
    """Execute a task through the NEXUS pipeline and print the seal summary.

    Runs fully in-memory — no database or Redis required.
    Gate 4 (drift) is skipped in CLI mode (requires Redis).

    Example:
        nexus run "What is NEXUS?"
        nexus run "Search for recent AI papers" --persona researcher
    """
    try:
        asyncio.run(_execute(task, persona))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        raise typer.Exit(1)
