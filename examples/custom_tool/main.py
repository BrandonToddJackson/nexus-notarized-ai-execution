"""NEXUS Example: Custom Tool Registration.

Shows how to define and register a custom tool using the @tool decorator,
then run a task that uses it — all in-memory, no API key required.

Run:
    python examples/custom_tool/main.py
"""

import asyncio
import os
import sys

# Ensure the repo root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("NEXUS_GATE_INTENT_THRESHOLD", "0.3")

from nexus.tools.plugin import tool
from nexus.types import RiskLevel


# ── 1. Define your custom tool ───────────────────────────────────────────────

@tool(
    name="weather_lookup",
    description="Look up current weather for a city",
    risk_level=RiskLevel.LOW,
    resource_pattern="web:weather:*",
    timeout_seconds=10,
)
async def weather_lookup(city: str) -> str:
    """Stub: returns mock weather data. Replace with a real API call."""
    return f"Weather in {city}: 72°F, partly cloudy. Wind: 8 mph SW."


# ── 2. Build in-memory engine (same helper as quickstart) ────────────────────

def _build_engine():
    from nexus.config import NexusConfig
    from nexus.types import PersonaContract, RiskLevel as RL
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
    import tempfile

    cfg = NexusConfig()

    # Persona that allows the custom weather tool
    persona = PersonaContract(
        name="weather_agent",
        description="Looks up weather information",
        allowed_tools=["weather_lookup", "knowledge_search"],
        resource_scopes=["web:weather:*", "kb:*"],
        intent_patterns=["look up weather", "check weather", "find weather"],
        max_ttl_seconds=60,
        risk_tolerance=RL.LOW,
    )

    persona_manager = PersonaManager([persona])
    embedding_service = EmbeddingService(model_name=cfg.embedding_model)
    tmp_dir = tempfile.mkdtemp(prefix="nexus_custom_tool_")
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
        ),
        output_validator=OutputValidator(),
        cot_logger=CoTLogger(),
        think_act_gate=ThinkActGate(),
        continue_complete_gate=ContinueCompleteGate(),
        escalate_gate=EscalateGate(),
        config=cfg,
        llm_client=None,
    )


async def main():
    print("\n" + "=" * 56)
    print("  NEXUS — Custom Tool Example")
    print("=" * 56 + "\n")
    print("  Registered tool: weather_lookup")
    print("  Persona        : weather_agent")
    print("  No API key required\n")

    print("Loading embeddings...")
    engine = _build_engine()

    task = "What is the weather in San Francisco?"
    print(f"  Task: {task!r}\n")

    chain = await engine.run(task, tenant_id="custom-tool-demo", persona_name="weather_agent")
    seals = await engine.ledger.get_chain(chain.id)

    print(f"  Status : {chain.status.value.upper()}")
    print(f"  Seals  : {len(seals)}")

    for seal in seals:
        print(f"\n  Seal  : {seal.id[:16]}...")
        print(f"  Tool  : {seal.tool_name}")
        print(f"  Status: {seal.status.value.upper()}")
        for gate in seal.anomaly_result.gates:
            verdict = gate.verdict.value
            icon = "✓" if verdict == "pass" else ("✗" if verdict == "fail" else "—")
            print(f"    [{icon}] {gate.gate_name:10s}  {gate.details}")
        if seal.tool_result:
            print(f"  Result: {seal.tool_result}")

    print("\n" + "=" * 56)


if __name__ == "__main__":
    asyncio.run(main())
