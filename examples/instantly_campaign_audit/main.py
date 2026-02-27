"""NEXUS Example: Instantly.ai Campaign Audit.

Runs a full health report across all Instantly.ai campaigns — lead counts,
cross-campaign duplicate detection, sender warmup status, daily send capacity,
and actionable recommendations — all gated through the NEXUS 4-gate pipeline
and sealed in the immutable ledger.

Requires:
    INSTANTLY_API_KEY=<your-v2-bearer-token>  # app.instantly.ai → Settings → API Keys

Run:
    INSTANTLY_API_KEY=your_key python examples/instantly_campaign_audit/main.py
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("NEXUS_GATE_INTENT_THRESHOLD", "0.3")


def _check_api_key() -> None:
    key = os.environ.get("INSTANTLY_API_KEY", "")
    if not key:
        print("  ERROR: INSTANTLY_API_KEY is not set.")
        print("  Get your v2 Bearer token at: app.instantly.ai → Settings → API Keys")
        print("  Run: INSTANTLY_API_KEY=your_key python examples/instantly_campaign_audit/main.py")
        sys.exit(1)
    # Inject into NEXUS config namespace
    os.environ.setdefault("NEXUS_INSTANTLY_API_KEY", key)


def _build_engine():
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
    import nexus.tools.builtin  # noqa: F401 — registers all built-in tools including Instantly
    from nexus.tools.selector import ToolSelector
    from nexus.tools.sandbox import Sandbox
    from nexus.tools.executor import ToolExecutor
    from nexus.reasoning.think_act import ThinkActGate
    from nexus.reasoning.continue_complete import ContinueCompleteGate
    from nexus.reasoning.escalate import EscalateGate
    from nexus.core.engine import NexusEngine
    from nexus.db.seed import DEFAULT_PERSONAS

    cfg = NexusConfig()

    # Load all seeded personas — includes campaign_optimizer
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
    embedding_service = EmbeddingService(model_name=cfg.embedding_model)
    tmp_dir = tempfile.mkdtemp(prefix="nexus_instantly_audit_")
    knowledge_store = KnowledgeStore(persist_dir=tmp_dir, embedding_fn=embedding_service.embed)

    anomaly_engine = AnomalyEngine(cfg, embedding_service=embedding_service, fingerprint_store=None)
    notary = Notary()
    ledger = Ledger()
    chain_manager = ChainManager()

    tool_registry = ToolRegistry()
    for _name, (defn, impl) in get_registered_tools().items():
        tool_registry.register(defn, impl)

    return NexusEngine(
        persona_manager=persona_manager,
        anomaly_engine=anomaly_engine,
        notary=notary,
        ledger=ledger,
        chain_manager=chain_manager,
        context_builder=ContextBuilder(knowledge_store=knowledge_store),
        tool_registry=tool_registry,
        tool_selector=ToolSelector(registry=tool_registry),
        tool_executor=ToolExecutor(
            registry=tool_registry,
            sandbox=Sandbox(),
            verifier=IntentVerifier(),
            config=cfg,
        ),
        output_validator=OutputValidator(),
        cot_logger=CoTLogger(),
        think_act_gate=ThinkActGate(),
        continue_complete_gate=ContinueCompleteGate(),
        escalate_gate=EscalateGate(),
        config=cfg,
        llm_client=None,
    )


def _print_gate(gate) -> None:
    verdict = gate.verdict.value if hasattr(gate.verdict, "value") else str(gate.verdict)
    icons = {"pass": "✓", "fail": "✗", "skip": "—", "warn": "⚠"}
    icon = icons.get(verdict.lower(), "?")
    print(f"      [{icon}] {gate.gate_name:10s}  score={gate.score:.3f}  {gate.details}")


def _print_audit(result: dict) -> None:
    print()
    print("  ── Campaigns ──────────────────────────────────")
    for c in result.get("campaigns", []):
        status = c.get("status", "?").upper()
        leads = c.get("lead_count", 0)
        print(f"    {status:10s}  leads={leads:4d}  {c.get('name', '')}")

    print()
    senders = result.get("senders", {})
    print(f"  ── Senders ─────────────────────────────────────")
    print(f"    Total    : {senders.get('total', 0)}")
    print(f"    Warmed   : {senders.get('warmed', 0)}")
    print(f"    Capacity : {senders.get('daily_capacity', 0)} emails/day")

    dupes = result.get("duplicate_leads", 0)
    if dupes:
        print()
        print(f"  ── Duplicates ({dupes} leads appear in multiple campaigns) ──")
        for d in result.get("top_duplicates", [])[:5]:
            print(f"    {d['email']}  (in {d['count']} campaigns)")

    print()
    print("  ── Recommendations ─────────────────────────────")
    for r in result.get("recommendations", []):
        print(f"    • {r}")


async def main():
    _check_api_key()

    print()
    print("=" * 60)
    print("  NEXUS — Instantly.ai Campaign Audit")
    print("=" * 60)
    print()
    print("  Persona  : campaign_optimizer")
    print("  Tool     : instantly_audit")
    print("  Gates    : Scope · Intent · TTL · Drift (4-gate pipeline)")
    print()

    print("  Loading embedding model...")
    engine = _build_engine()

    task = "Run a full campaign audit"
    print(f"  Task: {task!r}")
    print()

    from nexus.exceptions import AnomalyDetected

    try:
        chain = await engine.run(task, tenant_id="instantly-audit-demo", persona_name="campaign_optimizer")
    except AnomalyDetected as exc:
        print(f"  [BLOCKED] {exc}")
        return

    seals = await engine.ledger.get_chain(chain.id)
    print(f"  Chain  : {chain.id}")
    print(f"  Status : {chain.status.value.upper()}  ({len(seals)} seal(s))")
    print()

    for seal in seals:
        status = seal.status.value if hasattr(seal.status, "value") else str(seal.status)
        print(f"  Seal   : {seal.id[:16]}...")
        print(f"  Tool   : {seal.tool_name}")
        print(f"  Status : {status.upper()}")
        print()
        print("  Gate Results:")
        for gate in seal.anomaly_result.gates:
            _print_gate(gate)

        if seal.tool_result and isinstance(seal.tool_result, dict):
            _print_audit(seal.tool_result)
        elif seal.error:
            print(f"\n  Error  : {seal.error}")

    print()
    print("=" * 60)
    print("  Audit complete. Every call was gated and sealed.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(main())
