"""NEXUS Quickstart — Run a task end-to-end with no API keys or Docker.

All components run in-memory:
- No database (PostgreSQL not required)
- No Redis (Gate 4 drift check skipped)
- No API key (LLM decomposition skipped; single-step fallback)
- Embeddings load from HuggingFace on first run (~100 MB, cached thereafter)

Run:
    python examples/quickstart/main.py
"""

import asyncio
import os
import tempfile


def _build_engine(intent_threshold: float = 0.3):
    """Assemble NexusEngine from in-memory components.

    Args:
        intent_threshold: Gate 2 cosine similarity threshold.
                          Default 0.3 for quickstart demo (production uses 0.75).
    """
    # Override intent threshold via env so NexusConfig picks it up
    os.environ.setdefault("NEXUS_GATE_INTENT_THRESHOLD", str(intent_threshold))

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

    # ── Personas (from seed defaults — no DB needed) ──────────────────────────
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

    # ── Embeddings + Knowledge Store (ephemeral tmpdir) ───────────────────────
    print("Loading embedding model (cached after first run)...")
    embedding_service = EmbeddingService(model_name=cfg.embedding_model)
    tmp_dir = tempfile.mkdtemp(prefix="nexus_quickstart_")
    knowledge_store = KnowledgeStore(
        persist_dir=tmp_dir,
        embedding_fn=embedding_service.embed,
    )

    # ── Core security layer ───────────────────────────────────────────────────
    # fingerprint_store=None → Gate 4 drift skipped (no Redis in quickstart)
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

    # ── Tools (built-in @tool decorators auto-registered) ─────────────────────
    tool_registry = ToolRegistry()
    for _name, (definition, impl) in get_registered_tools().items():
        tool_registry.register(definition, impl)

    tool_selector = ToolSelector(registry=tool_registry)
    sandbox = Sandbox()
    tool_executor = ToolExecutor(registry=tool_registry, sandbox=sandbox, verifier=verifier)

    # ── Engine (no llm_client → single-step fallback, no API key needed) ──────
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
        llm_client=None,   # No API key required
    )


def _gate_icon(verdict: str) -> str:
    icons = {"pass": "✓ PASS", "fail": "✗ FAIL", "warn": "⚠ WARN", "skip": "— SKIP"}
    return icons.get(verdict.lower(), verdict.upper())


def _print_seal(seal, label: str = "") -> None:
    status = seal.status.value if hasattr(seal.status, "value") else str(seal.status)
    print(f"  Seal{label}: {seal.id[:16]}...")
    print(f"    Tool       : {seal.tool_name}")
    print(f"    Status     : {status.upper()}")
    print(f"    Fingerprint: {seal.fingerprint[:24]}...")
    print()

    print("    Gate Results:")
    for gate in seal.anomaly_result.gates:
        verdict = gate.verdict.value if hasattr(gate.verdict, "value") else str(gate.verdict)
        icon = _gate_icon(verdict)
        print(f"      [{icon:8s}] {gate.gate_name:10s}  score={gate.score:.3f}  {gate.details}")
    print()

    if seal.cot_trace:
        print("    Chain-of-Thought:")
        for j, entry in enumerate(seal.cot_trace, 1):
            print(f"      {j}. {entry}")
        print()

    if seal.tool_result:
        output_preview = str(seal.tool_result)[:120].replace("\n", " ")
        print(f"    Output     : {output_preview}")
        print()


async def demo_successful_execution(engine) -> None:
    """Demo 1: Researcher searches for information — all gates should pass."""
    from nexus.exceptions import AnomalyDetected, EscalationRequired

    task = "What is NEXUS?"
    tenant = "quickstart-demo"

    print(f"  Task   : {task!r}")
    print(f"  Tenant : {tenant!r}")
    print("  Persona: researcher")
    print()

    try:
        chain = await engine.run(task, tenant, persona_name="researcher")
    except AnomalyDetected as exc:
        print(f"  [BLOCKED] {exc}")
        return
    except EscalationRequired as exc:
        print(f"  [ESCALATED] {exc}")
        return

    seals = await engine.ledger.get_chain(chain.id)
    chain_status = chain.status.value if hasattr(chain.status, "value") else str(chain.status)

    print(f"  Chain : {chain.id}")
    print(f"  Status: {chain_status.upper()}  ({len(seals)} seal(s))")
    print()

    for i, seal in enumerate(seals):
        _print_seal(seal, label=f" {i + 1}")

    # Verify Merkle chain integrity
    from nexus.core.notary import Notary
    notary = Notary()
    try:
        notary.verify_chain(seals)
        print("  Merkle chain: VALID")
    except Exception as exc:
        print(f"  Merkle chain: COMPROMISED — {exc}")
    print()


async def demo_blocked_action(engine) -> None:
    """Demo 2: Misconfigured persona tries to use web_search — Gate 1 blocks it.

    The 'kb_only' persona only allows access to kb:* resources, but web_search
    generates a web:* resource target. Gate 1 correctly BLOCKS this action.
    """
    from nexus.exceptions import AnomalyDetected
    from nexus.types import PersonaContract, RiskLevel

    # Add a locked-down persona that can only access the knowledge base
    kb_only = PersonaContract(
        name="kb_only",
        description="Locked-down persona: knowledge base access only",
        allowed_tools=["web_search"],      # web_search is allowed by tool name...
        resource_scopes=["kb:*"],          # ...but only kb:* resources are in scope
        intent_patterns=["search for information", "look up"],
        max_ttl_seconds=60,
        risk_tolerance=RiskLevel.LOW,
    )
    engine.persona_manager._contracts["kb_only"] = kb_only

    task = "search for NEXUS documentation"
    tenant = "quickstart-demo"

    print(f"  Task   : {task!r}")
    print("  Persona: kb_only  (allowed_tools: [web_search], resource_scopes: [kb:*])")
    print("  Expected: BLOCKED — web_search targets web:* which is NOT in kb:*")
    print()

    try:
        await engine.run(task, tenant, persona_name="kb_only")
        print("  [UNEXPECTED] Action was not blocked — check anomaly gates!")
    except AnomalyDetected as exc:
        print("  [CORRECTLY BLOCKED] Gate 1 (Scope) blocked the action:")
        print(f"    {exc}")
    finally:
        # Clean up the demo persona
        engine.persona_manager._contracts.pop("kb_only", None)
    print()


async def main():
    print()
    print("=" * 64)
    print("  NEXUS Quickstart")
    print("  AI Agent Framework — Notarized Execution Demo")
    print("=" * 64)
    print()
    print("  Mode  : In-memory (no Docker, no API key required)")
    print("  Gates : Scope ✓  Intent ✓  TTL ✓  Drift — (no Redis)")
    print("  LLM   : Disabled (single-step fallback active)")
    print()

    # ── Build engine with demo-friendly intent threshold ─────────────────────
    # Production uses 0.75; we use 0.3 here so the demo passes without
    # an API key that would generate semantically-aligned planned actions.
    engine = _build_engine(intent_threshold=0.3)

    # ── Demo 1: Successful execution ──────────────────────────────────────────
    print("-" * 64)
    print("  Demo 1: Successful Task Execution")
    print("-" * 64)
    await demo_successful_execution(engine)

    # ── Demo 2: Blocked action ────────────────────────────────────────────────
    print("-" * 64)
    print("  Demo 2: Blocked Action (Gate 1 — Scope Check)")
    print("-" * 64)
    await demo_blocked_action(engine)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 64)
    print("  Quickstart complete.")
    print()
    print("  Next steps:")
    print("    docker compose up --build   # full stack (DB + Redis + API)")
    print("    curl -X POST http://localhost:8000/v1/execute \\")
    print('      -H "Authorization: nxs_demo_key_12345" \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -d \'{"task": "What is NEXUS?"}\'')
    print()
    print("    nexus run \"What is NEXUS?\"  # CLI mode")
    print("    pytest tests/ -v             # 483 tests")
    print("=" * 64)
    print()


if __name__ == "__main__":
    asyncio.run(main())
