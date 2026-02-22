"""NEXUS Example: Local LLM via Ollama (zero API keys).

Runs NEXUS with a locally-hosted Ollama model instead of a cloud provider.
Uses litellm's Ollama provider — no API key required.

Prerequisites:
    brew install ollama
    ollama pull llama3   # or mistral, phi3, etc.
    ollama serve         # starts on http://localhost:11434

Run:
    python examples/local_llm/main.py
    python examples/local_llm/main.py --model ollama/mistral

Environment (optional, override via CLI arg):
    NEXUS_DEFAULT_LLM_MODEL=ollama/llama3
"""

import asyncio
import os
import sys
import argparse
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("NEXUS_GATE_INTENT_THRESHOLD", "0.3")


def _build_engine(model: str):
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
    from nexus.db.seed import DEFAULT_PERSONAS

    # Configure litellm to use Ollama
    os.environ["NEXUS_DEFAULT_LLM_MODEL"] = model

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
    print(f"Loading embedding model ({cfg.embedding_model})...")
    embedding_service = EmbeddingService(model_name=cfg.embedding_model)
    tmp_dir = tempfile.mkdtemp(prefix="nexus_local_llm_")
    knowledge_store = KnowledgeStore(persist_dir=tmp_dir, embedding_fn=embedding_service.embed)

    anomaly_engine = AnomalyEngine(cfg, embedding_service=embedding_service, fingerprint_store=None)
    notary = Notary()
    ledger = Ledger()
    chain_manager = ChainManager()

    tool_registry = ToolRegistry()
    for _name, (defn, impl) in get_registered_tools().items():
        tool_registry.register(defn, impl)

    # LLM client: litellm routes "ollama/llama3" to local Ollama automatically.
    # complete() must return {"content": str, "usage": dict} to match LLMClient interface.
    try:
        import litellm
        class OllamaClient:
            async def complete(self, messages, **kw):
                resp = await litellm.acompletion(model=model, messages=messages, **kw)
                msg = resp.choices[0].message
                usage = resp.usage or {}
                return {
                    "content": msg.content or "",
                    "tool_calls": [],
                    "usage": {
                        "input_tokens": getattr(usage, "prompt_tokens", 0),
                        "output_tokens": getattr(usage, "completion_tokens", 0),
                    },
                }
        llm_client = OllamaClient()
        print(f"LLM: {model} via Ollama (litellm)\n")
    except Exception as e:
        print(f"[warn] Could not create Ollama client ({e}). Falling back to single-step mode.\n")
        llm_client = None

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
        llm_client=llm_client,
    )


async def main():
    parser = argparse.ArgumentParser(description="NEXUS Local LLM Example")
    parser.add_argument("--model", default="ollama/llama3", help="litellm model string")
    parser.add_argument("--task", default="What are the key benefits of NEXUS?")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  NEXUS — Local LLM Example (Ollama)")
    print("=" * 60)
    print(f"  Model : {args.model}")
    print(f"  Task  : {args.task!r}")
    print("  Cost  : $0.00 (no cloud API calls)\n")

    engine = _build_engine(args.model)
    chain = await engine.run(args.task, tenant_id="local-llm-demo", persona_name="researcher")
    seals = await engine.ledger.get_chain(chain.id)

    print(f"\n  Status : {chain.status.value.upper()}")
    print(f"  Seals  : {len(seals)}")

    for i, seal in enumerate(seals, 1):
        print(f"\n  Seal {i}: {seal.tool_name} → {seal.status.value.upper()}")
        for gate in seal.anomaly_result.gates:
            icon = "✓" if gate.verdict.value == "pass" else ("✗" if gate.verdict.value == "fail" else "—")
            print(f"    [{icon}] {gate.gate_name}")
        if seal.tool_result:
            print(f"  Result: {str(seal.tool_result)[:200]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
