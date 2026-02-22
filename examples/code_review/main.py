"""NEXUS Example: Code Review Agent.

A multi-step chain that:
  1. Reads source files from the local filesystem
  2. Analyzes patterns and computes metrics
  3. Writes a review report

Demonstrates: multi-step chains, file tool usage, analyst persona.

Run:
    python examples/code_review/main.py [--target-dir /path/to/code]

No API key required.
"""

import asyncio
import os
import sys
import tempfile
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("NEXUS_GATE_INTENT_THRESHOLD", "0.3")


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

    # analyst persona: can read files and compute stats — appropriate for code review
    analyst = PersonaContract(
        name="analyst",
        description="Analyzes data and computes statistics",
        allowed_tools=["file_read", "compute_stats", "knowledge_search"],
        resource_scopes=["file:*", "data:*", "kb:*"],
        intent_patterns=["analyze data", "compute statistics", "calculate", "summarize findings"],
        max_ttl_seconds=120,
        risk_tolerance=RiskLevel.MEDIUM,
    )

    print("Loading embeddings...")
    embedding_service = EmbeddingService(model_name=cfg.embedding_model)
    tmp_dir = tempfile.mkdtemp(prefix="nexus_code_review_")
    knowledge_store = KnowledgeStore(persist_dir=tmp_dir, embedding_fn=embedding_service.embed)

    tool_registry = ToolRegistry()
    for _name, (defn, impl) in get_registered_tools().items():
        tool_registry.register(defn, impl)

    return NexusEngine(
        persona_manager=PersonaManager([analyst]),
        anomaly_engine=AnomalyEngine(cfg, embedding_service=embedding_service, fingerprint_store=None),
        notary=Notary(),
        ledger=Ledger(),
        chain_manager=ChainManager(),
        context_builder=ContextBuilder(knowledge_store=knowledge_store),
        tool_registry=tool_registry,
        tool_selector=ToolSelector(registry=tool_registry),
        tool_executor=ToolExecutor(
            registry=tool_registry, sandbox=Sandbox(), verifier=IntentVerifier()
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
    parser = argparse.ArgumentParser(description="NEXUS Code Review Example")
    parser.add_argument(
        "--target-dir",
        default=str(Path(__file__).parent.parent.parent / "nexus" / "core"),
        help="Directory to review",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  NEXUS — Code Review Example")
    print("=" * 60)
    print(f"  Target : {args.target_dir}")
    print(f"  Persona: analyst (file_read + compute_stats)\n")

    engine = _build_engine()

    task = f"Analyze the Python files in {args.target_dir} and summarize their structure"
    print(f"  Task: {task!r}\n")

    chain = await engine.run(task, tenant_id="code-review-demo", persona_name="analyst")
    seals = await engine.ledger.get_chain(chain.id)

    print(f"  Chain status : {chain.status.value.upper()}")
    print(f"  Actions taken: {len(seals)}\n")

    for i, seal in enumerate(seals, 1):
        gates_pass = sum(1 for g in seal.anomaly_result.gates if g.verdict.value == "pass")
        gates_fail = sum(1 for g in seal.anomaly_result.gates if g.verdict.value == "fail")
        print(f"  Step {i}: {seal.tool_name}")
        print(f"    Status : {seal.status.value.upper()}")
        print(f"    Gates  : {gates_pass} pass, {gates_fail} fail")
        if seal.tool_result:
            preview = str(seal.tool_result)[:300].replace("\n", " ")
            print(f"    Result : {preview}")
        print()

    # Verify Merkle chain
    from nexus.core.notary import Notary
    notary = Notary()
    try:
        notary.verify_chain(seals)
        print(f"  Merkle chain: VALID ({len(seals)} seals)")
    except Exception as e:
        print(f"  Merkle chain: ERROR — {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
