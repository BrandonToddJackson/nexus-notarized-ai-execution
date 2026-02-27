"""NEXUS Example: Instantly.ai Cold Email Campaign Builder.

Demonstrates the full cold email campaign creation lifecycle through NEXUS:

  Step 1 — Audit: check sender warmup + existing campaigns (campaign_optimizer)
  Step 2 — Create: build a new campaign with email sequence + warmed senders (campaign_outreach)
  Step 3 — Add leads: enroll contacts with automatic dedup (campaign_outreach)
  Step 4 — Activate: launch the campaign (campaign_outreach)

Every step passes through the 4-gate anomaly pipeline and is sealed in the
immutable ledger. High-risk actions (create, activate) require explicit persona
scope — Gate 1 blocks any unauthorized call.

Requires:
    INSTANTLY_API_KEY=<your-v2-bearer-token>  # app.instantly.ai → Settings → API Keys

Run:
    INSTANTLY_API_KEY=your_key python examples/instantly_cold_email_campaign/main.py

Note:
    Set DRY_RUN=1 to run Steps 1-2 only (audit + create) without adding leads
    or activating, so you can review the campaign in Instantly before launching.

    DRY_RUN=1 INSTANTLY_API_KEY=your_key python examples/instantly_cold_email_campaign/main.py
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("NEXUS_GATE_INTENT_THRESHOLD", "0.3")

# ── Campaign configuration — edit before running ──────────────────────────────

CAMPAIGN_NAME = "NEXUS Demo — AI Automation Outreach"

# Full 3-step sequence — configure each step with subject, body, and delay
# delay_days is the number of days after the previous step (0 for the first step)
EMAIL_SEQUENCE = [
    {
        "subject": "Quick question about your automation stack",
        "body": (
            "Hi {{first_name}},\n\n"
            "I noticed {{company_name}} has been scaling its operations — congrats on the growth.\n\n"
            "I'm working on NEXUS, an open-source AI agent framework built around accountability: "
            "every action is gated, notarized, and sealed before execution. "
            "No black-box agents, no blind trust.\n\n"
            "Would a 15-minute call make sense to see if it fits your stack?\n\n"
            "Best,\nBrandon"
        ),
        "delay_days": 0,
    },
    {
        "subject": "Re: Quick question about your automation stack",
        "body": (
            "Hi {{first_name}},\n\n"
            "Just bumping this up — still curious if accountable AI automation is on your radar.\n\n"
            "Happy to share a demo or the GitHub repo.\n\n"
            "Best,\nBrandon"
        ),
        "delay_days": 3,
    },
    {
        "subject": "Last touch — NEXUS for {{company_name}}",
        "body": (
            "Hi {{first_name}},\n\n"
            "I'll keep this short — if AI automation accountability ever becomes a priority, "
            "NEXUS is open source and ready to run.\n\n"
            "github.com/BrandonToddJackson/nexus-notarized-ai-execution\n\n"
            "Best,\nBrandon"
        ),
        "delay_days": 7,
    },
]

# Demo leads — replace with your actual lead list
DEMO_LEADS = [
    {"email": "demo-lead-1@example.com", "first_name": "Alex", "last_name": "Chen", "company_name": "Acme Corp"},
    {"email": "demo-lead-2@example.com", "first_name": "Sam", "last_name": "Rivera", "company_name": "BuildFast"},
]

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

# ─────────────────────────────────────────────────────────────────────────────


def _check_api_key() -> None:
    key = os.environ.get("INSTANTLY_API_KEY", "")
    if not key:
        print("  ERROR: INSTANTLY_API_KEY is not set.")
        print("  Get your v2 Bearer token at: app.instantly.ai → Settings → API Keys")
        print("  Run: INSTANTLY_API_KEY=your_key python examples/instantly_cold_email_campaign/main.py")
        sys.exit(1)
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
    tmp_dir = tempfile.mkdtemp(prefix="nexus_cold_email_")
    knowledge_store = KnowledgeStore(persist_dir=tmp_dir, embedding_fn=embedding_service.embed)

    anomaly_engine = AnomalyEngine(cfg, embedding_service=embedding_service, fingerprint_store=None)

    tool_registry = ToolRegistry()
    for _name, (defn, impl) in get_registered_tools().items():
        tool_registry.register(defn, impl)

    return NexusEngine(
        persona_manager=persona_manager,
        anomaly_engine=anomaly_engine,
        notary=Notary(),
        ledger=Ledger(),
        chain_manager=ChainManager(),
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


def _print_gates(seal) -> None:
    for gate in seal.anomaly_result.gates:
        verdict = gate.verdict.value if hasattr(gate.verdict, "value") else str(gate.verdict)
        icons = {"pass": "✓", "fail": "✗", "skip": "—", "warn": "⚠"}
        icon = icons.get(verdict.lower(), "?")
        print(f"      [{icon}] {gate.gate_name:10s}  score={gate.score:.3f}")


async def _run_step(engine, label: str, task: str, persona: str, tenant: str) -> dict | None:
    from nexus.exceptions import AnomalyDetected

    print(f"  ── {label} {'─' * max(0, 50 - len(label))}")
    print(f"  Task   : {task[:80]!r}")
    print(f"  Persona: {persona}")

    try:
        chain = await engine.run(task, tenant_id=tenant, persona_name=persona)
    except AnomalyDetected as exc:
        print(f"  [BLOCKED] {exc}")
        return None

    seals = await engine.ledger.get_chain(chain.id)
    status = chain.status.value.upper()
    print(f"  Status : {status}  ({len(seals)} seal(s))")
    print()

    result = None
    for seal in seals:
        seal_status = seal.status.value if hasattr(seal.status, "value") else str(seal.status)
        print(f"  Seal   : {seal.id[:16]}...  tool={seal.tool_name}  status={seal_status.upper()}")
        _print_gates(seal)
        if seal.tool_result:
            result = seal.tool_result
        if seal.error:
            print(f"  Error  : {seal.error}")

    return result


async def main():
    _check_api_key()

    print()
    print("=" * 60)
    print("  NEXUS — Cold Email Campaign Builder")
    print("=" * 60)
    if DRY_RUN:
        print("  Mode: DRY RUN (audit + create only, no leads/activate)")
    print()
    print(f"  Campaign : {CAMPAIGN_NAME}")
    print(f"  Leads    : {len(DEMO_LEADS)} demo contacts")
    print()

    print("  Loading embedding model...")
    engine = _build_engine()
    tenant = "cold-email-demo"

    print()

    # ── Step 1: Audit ─────────────────────────────────────────────────────────
    audit_result = await _run_step(
        engine,
        label="Step 1: Campaign Audit",
        task="Run a full campaign audit",
        persona="campaign_optimizer",
        tenant=tenant,
    )

    if audit_result:
        senders = audit_result.get("senders", {})
        print(f"  Senders  : {senders.get('warmed', 0)}/{senders.get('total', 0)} warmed")
        print(f"  Capacity : {senders.get('daily_capacity', 0)} emails/day")
        n_camps = audit_result.get("total_campaigns", 0)
        print(f"  Campaigns: {n_camps} existing")

    print()

    # ── Step 2: Create Campaign ───────────────────────────────────────────────
    import json as _json
    create_task = (
        f"Create a campaign called {CAMPAIGN_NAME!r} "
        f"with sequence {_json.dumps(EMAIL_SEQUENCE)}"
    )

    create_result = await _run_step(
        engine,
        label="Step 2: Create Campaign",
        task=create_task,
        persona="campaign_outreach",
        tenant=tenant,
    )

    campaign_id = None
    if create_result and isinstance(create_result, dict):
        campaign_id = create_result.get("campaign_id", "")
        msg = create_result.get("message", "")
        status = create_result.get("status", "")
        print(f"  Campaign ID : {campaign_id}")
        print(f"  Status      : {status}")
        print(f"  Senders     : {create_result.get('senders', '?')} warmed")
        print(f"  Sequence    : {create_result.get('sequence_steps', '?')} step(s)")
        if msg:
            print(f"  Note        : {msg}")

    print()

    if DRY_RUN:
        print("  DRY RUN: skipping lead add and activation.")
        print(f"  Review your campaign in Instantly: app.instantly.ai")
        print()
    else:
        if not campaign_id:
            print("  Skipping Steps 3-4: no campaign ID returned from Step 2.")
        else:
            # ── Step 3: Add Leads ─────────────────────────────────────────────
            add_task = (
                f"Add leads to campaign {campaign_id!r} with leads "
                + _json.dumps(DEMO_LEADS)
            )
            add_result = await _run_step(
                engine,
                label="Step 3: Add Leads",
                task=add_task,
                persona="campaign_outreach",
                tenant=tenant,
            )

            if add_result and isinstance(add_result, dict):
                print(f"  Added    : {add_result.get('added', 0)}")
                print(f"  Skipped  : {add_result.get('skipped_duplicates', 0)} duplicates")

            print()

            # ── Step 4: Activate ──────────────────────────────────────────────
            activate_result = await _run_step(
                engine,
                label="Step 4: Activate Campaign",
                task=f"Activate campaign {campaign_id!r}",
                persona="campaign_outreach",
                tenant=tenant,
            )

            if activate_result and isinstance(activate_result, dict):
                print(f"  {activate_result.get('message', 'Campaign activated.')}")

            print()

    # ── Ledger summary ────────────────────────────────────────────────────────
    all_chains = []
    print("  ── Ledger Summary ──────────────────────────────")
    ledger_entries = engine.ledger._seals if hasattr(engine.ledger, "_seals") else {}
    total_seals = len(ledger_entries)
    print(f"  Total seals written : {total_seals}")
    print(f"  Every seal is Merkle-chained and tamper-evident.")
    print()
    print("=" * 60)
    print("  Campaign flow complete.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(main())
