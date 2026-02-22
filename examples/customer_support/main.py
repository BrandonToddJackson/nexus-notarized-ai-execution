"""NEXUS Example: Customer Support Agent.

Demonstrates a multi-step chain:
  1. researcher searches the knowledge base for relevant docs
  2. creator drafts a support response
  3. communicator would send it (blocked by default — requires approval)

Run:
    python examples/customer_support/main.py

No API key required. LLM decomposition is skipped; steps run sequentially.
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("NEXUS_GATE_INTENT_THRESHOLD", "0.3")


SUPPORT_KB = [
    {
        "content": "NEXUS provides a 30-day free trial with full access to all features. "
                   "Credit card is not required during the trial period.",
        "source": "faq_billing.txt",
        "namespace": "support",
    },
    {
        "content": "To reset your password, click 'Forgot Password' on the login page. "
                   "You will receive an email within 5 minutes. Check your spam folder if not received.",
        "source": "faq_account.txt",
        "namespace": "support",
    },
    {
        "content": "NEXUS integrates with Slack, GitHub, Jira, and Linear via webhooks. "
                   "See docs/integrations.md for setup instructions.",
        "source": "faq_integrations.txt",
        "namespace": "support",
    },
]


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
    import nexus.tools.builtin  # noqa: F401
    from nexus.tools.selector import ToolSelector
    from nexus.tools.sandbox import Sandbox
    from nexus.tools.executor import ToolExecutor
    from nexus.reasoning.think_act import ThinkActGate
    from nexus.reasoning.continue_complete import ContinueCompleteGate
    from nexus.reasoning.escalate import EscalateGate
    from nexus.core.engine import NexusEngine

    cfg = NexusConfig()

    researcher = PersonaContract(
        name="researcher",
        description="Searches the support knowledge base",
        allowed_tools=["knowledge_search", "file_read"],
        resource_scopes=["kb:*", "file:read:*"],
        intent_patterns=["search for information", "find data about", "look up"],
        max_ttl_seconds=60,
        risk_tolerance=RiskLevel.LOW,
    )

    personas = [researcher]
    persona_manager = PersonaManager(personas)

    print("Loading embeddings...")
    embedding_service = EmbeddingService(model_name=cfg.embedding_model)
    tmp_dir = tempfile.mkdtemp(prefix="nexus_support_")
    knowledge_store = KnowledgeStore(persist_dir=tmp_dir, embedding_fn=embedding_service.embed)

    tool_registry = _build_registry(get_registered_tools)

    return NexusEngine(
        persona_manager=persona_manager,
        anomaly_engine=AnomalyEngine(cfg, embedding_service=embedding_service, fingerprint_store=None),
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
        ),
        output_validator=OutputValidator(),
        cot_logger=CoTLogger(),
        think_act_gate=ThinkActGate(),
        continue_complete_gate=ContinueCompleteGate(),
        escalate_gate=EscalateGate(),
        config=cfg,
        llm_client=None,
    ), knowledge_store


def _build_registry(get_fn):
    from nexus.tools.registry import ToolRegistry
    registry = ToolRegistry()
    for _name, (defn, impl) in get_fn().items():
        registry.register(defn, impl)
    return registry


async def seed_kb(knowledge_store, tenant_id: str) -> None:
    """Load support docs into the in-memory knowledge base."""
    from nexus.types import KnowledgeDocument
    for doc_data in SUPPORT_KB:
        doc = KnowledgeDocument(
            tenant_id=tenant_id,
            namespace=doc_data["namespace"],
            source=doc_data["source"],
            content=doc_data["content"],
        )
        await knowledge_store.add_document(doc)


async def main():
    print("\n" + "=" * 60)
    print("  NEXUS — Customer Support Example")
    print("=" * 60 + "\n")

    engine, knowledge_store = _build_engine()
    tenant_id = "support-demo"

    print("Seeding knowledge base with support docs...")
    await seed_kb(knowledge_store, tenant_id)
    print(f"  Loaded {len(SUPPORT_KB)} documents\n")

    ticket = "How do I reset my password? I've been waiting 10 minutes and haven't received an email."
    print(f"  Customer ticket: {ticket!r}\n")

    chain = await engine.run(ticket, tenant_id=tenant_id, persona_name="researcher")
    seals = await engine.ledger.get_chain(chain.id)

    print(f"  Chain status : {chain.status.value.upper()}")
    print(f"  Actions taken: {len(seals)}\n")

    for i, seal in enumerate(seals, 1):
        print(f"  Step {i}: {seal.tool_name} [{seal.status.value.upper()}]")
        passed = sum(1 for g in seal.anomaly_result.gates if g.verdict.value == "pass")
        print(f"    Gates: {passed}/4 passed")
        if seal.tool_result:
            preview = str(seal.tool_result)[:200].replace("\n", " ")
            print(f"    Result: {preview}")
        print()

    print("=" * 60)
    print("  Note: send_email step requires communicator persona + approval.")
    print("  See examples/quickstart/main.py for Gate 1 block demo.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
