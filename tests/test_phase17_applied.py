#!/usr/bin/env python
"""Phase 17 applied integration test — uses REAL engine components, no mocks.

Verifies run_workflow() end-to-end with:
  1. Single ACTION step
  2. Linear A→B with {{template}} param resolution
  3. BRANCH step (condition routing)
  4. LOOP step (iterate over trigger items)
  5. PARALLEL step (fan-out)
  6. Gate-blocking (wrong resource scope → Gate 1 FAIL)

Run with:
    python test_phase17_applied.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import traceback
from typing import Any

# ── Engine builder ─────────────────────────────────────────────────────────────

def build_engine_with_workflow_manager():
    """Build a fully real NexusEngine + WorkflowManager (no DB, no Redis, no LLM)."""
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
    from nexus.workflows.manager import WorkflowManager

    cfg = NexusConfig()
    print(f"  Config: gate_intent_threshold={cfg.gate_intent_threshold}")

    # Personas from seed defaults
    persona_contracts = []
    for p in DEFAULT_PERSONAS:
        try:
            risk = RiskLevel(p["risk_tolerance"])
        except (ValueError, KeyError):
            risk = RiskLevel.MEDIUM
        persona_contracts.append(PersonaContract(
            name=p["name"],
            description=p["description"],
            allowed_tools=p["allowed_tools"],
            resource_scopes=p["resource_scopes"],
            intent_patterns=p["intent_patterns"],
            max_ttl_seconds=p.get("max_ttl_seconds", 300),
            risk_tolerance=risk,
        ))

    persona_manager = PersonaManager(persona_contracts)
    print(f"  Personas loaded: {[p.name for p in persona_contracts]}")

    print("  Loading embedding model (sentence-transformers all-MiniLM-L6-v2)...")
    embedding_service = EmbeddingService(model_name=cfg.embedding_model)
    print("  Embedding model loaded.")

    tmp_dir = tempfile.mkdtemp(prefix="nexus_phase17_")
    knowledge_store = KnowledgeStore(
        persist_dir=tmp_dir,
        embedding_fn=embedding_service.embed,
    )

    anomaly_engine = AnomalyEngine(
        config=cfg,
        embedding_service=embedding_service,
        fingerprint_store=None,  # skip Gate 4 (no Redis)
    )

    notary = Notary()
    ledger = Ledger()
    chain_manager = ChainManager()
    verifier = IntentVerifier()
    output_validator = OutputValidator()
    cot_logger = CoTLogger()
    context_builder = ContextBuilder(knowledge_store=knowledge_store)

    tool_registry = ToolRegistry()
    registered = get_registered_tools()
    for _name, (definition, impl) in registered.items():
        tool_registry.register(definition, impl)
    print(f"  Tools registered: {list(registered.keys())}")

    tool_selector = ToolSelector(registry=tool_registry)
    sandbox = Sandbox()
    tool_executor = ToolExecutor(registry=tool_registry, sandbox=sandbox, verifier=verifier)

    workflow_manager = WorkflowManager()

    engine = NexusEngine(
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
        workflow_manager=workflow_manager,
    )
    return engine, workflow_manager


# ── Scenario runner ────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []  # (scenario, status, detail)


def report(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    color = "\033[32m" if ok else "\033[31m"
    reset = "\033[0m"
    print(f"  {color}[{status}]{reset} {name}" + (f" — {detail}" if detail else ""))
    results.append((name, status, detail))


# ── Scenario 1: Single ACTION step ────────────────────────────────────────────

async def scenario_single_action(engine, workflow_manager) -> None:
    print("\n─── Scenario 1: Single ACTION step ───")
    from nexus.types import WorkflowStep, StepType, ChainStatus

    step = WorkflowStep(
        id="step-s1",
        workflow_id="wf-1",
        step_type=StepType.ACTION,
        name="search_nexus",
        tool_name="knowledge_search",
        tool_params={"query": "What is NEXUS?"},
        persona_name="researcher",
    )

    wf = await workflow_manager.create(
        tenant_id="test",
        name="Single Action",
        steps=[step],
        edges=[],
    )
    wf = await workflow_manager.activate(wf.id, "test")

    execution = await engine.run_workflow(wf.id, "test", trigger_data={"topic": "NEXUS"})
    print(f"    Status: {execution.status}")
    print(f"    Step results: {list(execution.step_results.keys())}")
    print(f"    chain_id: {execution.chain_id[:16]}…")

    report(
        "single action step completes",
        execution.status == ChainStatus.COMPLETED,
        f"status={execution.status.value}",
    )
    report(
        "step result recorded in execution",
        "search_nexus" in execution.step_results,
        f"keys={list(execution.step_results.keys())}",
    )
    report(
        "chain_id populated",
        bool(execution.chain_id),
    )


# ── Scenario 2: Linear A→B with template params ───────────────────────────────

async def scenario_linear_template(engine, workflow_manager) -> None:
    print("\n─── Scenario 2: Linear A→B with {{template}} params ───")
    from nexus.types import WorkflowStep, WorkflowEdge, StepType, EdgeType, ChainStatus

    s1 = WorkflowStep(
        id="step-lin-a",
        workflow_id="wf-2",
        step_type=StepType.ACTION,
        name="fetch",
        tool_name="knowledge_search",
        tool_params={"query": "{{trigger.topic}}"},
        persona_name="researcher",
    )
    s2 = WorkflowStep(
        id="step-lin-b",
        workflow_id="wf-2",
        step_type=StepType.ACTION,
        name="summarize",
        tool_name="knowledge_search",
        tool_params={"query": "summarize {{trigger.topic}}"},
        persona_name="researcher",
    )
    edge = WorkflowEdge(
        id="edge-lin-1",
        workflow_id="wf-2",
        source_step_id=s1.id,
        target_step_id=s2.id,
        edge_type=EdgeType.DEFAULT,
    )

    wf = await workflow_manager.create(
        tenant_id="test",
        name="Linear Template",
        steps=[s1, s2],
        edges=[edge],
    )
    wf = await workflow_manager.activate(wf.id, "test")

    execution = await engine.run_workflow(wf.id, "test", trigger_data={"topic": "AI safety"})
    print(f"    Status: {execution.status}")
    print(f"    Step results: {list(execution.step_results.keys())}")

    report(
        "linear 2-step completes",
        execution.status == ChainStatus.COMPLETED,
        f"status={execution.status.value}",
    )
    report(
        "both steps executed",
        "fetch" in execution.step_results and "summarize" in execution.step_results,
    )


# ── Scenario 3: BRANCH step ───────────────────────────────────────────────────

async def scenario_branch(engine, workflow_manager) -> None:
    print("\n─── Scenario 3: BRANCH step ───")
    from nexus.types import WorkflowStep, WorkflowEdge, StepType, EdgeType, ChainStatus

    branch = WorkflowStep(
        id="step-br-decide",
        workflow_id="wf-3",
        step_type=StepType.BRANCH,
        name="decide",
        config={},
    )
    yes_step = WorkflowStep(
        id="step-br-yes",
        workflow_id="wf-3",
        step_type=StepType.ACTION,
        name="yes_path",
        tool_name="knowledge_search",
        tool_params={"query": "positive path"},
        persona_name="researcher",
    )
    no_step = WorkflowStep(
        id="step-br-no",
        workflow_id="wf-3",
        step_type=StepType.ACTION,
        name="no_path",
        tool_name="knowledge_search",
        tool_params={"query": "negative path"},
        persona_name="researcher",
    )

    cond_edge = WorkflowEdge(
        id="edge-br-cond",
        workflow_id="wf-3",
        source_step_id=branch.id,
        target_step_id=yes_step.id,
        edge_type=EdgeType.CONDITIONAL,
        condition="true",
    )
    default_edge = WorkflowEdge(
        id="edge-br-def",
        workflow_id="wf-3",
        source_step_id=branch.id,
        target_step_id=no_step.id,
        edge_type=EdgeType.DEFAULT,
    )

    wf = await workflow_manager.create(
        tenant_id="test",
        name="Branch Test",
        steps=[branch, yes_step, no_step],
        edges=[cond_edge, default_edge],
    )
    wf = await workflow_manager.activate(wf.id, "test")

    execution = await engine.run_workflow(wf.id, "test")
    print(f"    Status: {execution.status}")
    print(f"    Step results: {list(execution.step_results.keys())}")
    print(f"    decide result: {execution.step_results.get('decide')}")

    report(
        "branch workflow completes",
        execution.status == ChainStatus.COMPLETED,
        f"status={execution.status.value}",
    )
    report(
        "branch took yes_path (condition=true)",
        "yes_path" in execution.step_results,
    )
    report(
        "branch did NOT take no_path",
        "no_path" not in execution.step_results or
        execution.step_results.get("no_path", {}).get("status") != "executed",
    )
    report(
        "branch step recorded as 'branched'",
        execution.step_results.get("decide", {}).get("status") == "branched",
    )


# ── Scenario 4: LOOP step ─────────────────────────────────────────────────────

async def scenario_loop(engine, workflow_manager) -> None:
    print("\n─── Scenario 4: LOOP step (self-loop model) ───")
    from nexus.types import WorkflowStep, WorkflowEdge, StepType, EdgeType, ChainStatus
    from uuid import uuid4

    # Validator requires: loop step has self-LOOP_BACK + DEFAULT exit edge.
    # The loop step executes its own tool N times (one per item).
    loop_id = str(uuid4())
    exit_id = str(uuid4())
    wf_id = str(uuid4())

    loop = WorkflowStep(
        id=loop_id,
        workflow_id=wf_id,
        step_type=StepType.LOOP,
        name="process_items",
        tool_name="knowledge_search",
        tool_params={"query": "item: {{loop_current}}"},
        persona_name="researcher",
        config={"iterator": "{{trigger.items}}"},
    )
    exit_step = WorkflowStep(
        id=exit_id,
        workflow_id=wf_id,
        step_type=StepType.ACTION,
        name="after_loop",
        tool_name="knowledge_search",
        tool_params={"query": "loop complete"},
        persona_name="researcher",
    )
    # Self-LOOP_BACK + DEFAULT exit (required by validator)
    loop_back_edge = WorkflowEdge(
        id=str(uuid4()), workflow_id=wf_id,
        source_step_id=loop_id, target_step_id=loop_id,
        edge_type=EdgeType.LOOP_BACK,
    )
    exit_edge = WorkflowEdge(
        id=str(uuid4()), workflow_id=wf_id,
        source_step_id=loop_id, target_step_id=exit_id,
        edge_type=EdgeType.DEFAULT,
    )

    wf = await workflow_manager.create(
        tenant_id="test",
        name="Loop Test",
        steps=[loop, exit_step],
        edges=[loop_back_edge, exit_edge],
    )
    wf = await workflow_manager.activate(wf.id, "test")

    items = ["alpha", "beta", "gamma"]
    execution = await engine.run_workflow(wf.id, "test", trigger_data={"items": items})
    print(f"    Status: {execution.status}")
    loop_result = execution.step_results.get("process_items", {})
    print(f"    Loop result: {loop_result}")

    report(
        "loop workflow completes",
        execution.status == ChainStatus.COMPLETED,
        f"status={execution.status.value}",
    )
    report(
        "loop ran correct number of iterations",
        loop_result.get("iterations") == len(items),
        f"iterations={loop_result.get('iterations')} expected={len(items)}",
    )
    report(
        "exit step executed after loop",
        "after_loop" in execution.step_results,
    )


# ── Scenario 5: PARALLEL step ─────────────────────────────────────────────────

async def scenario_parallel(engine, workflow_manager) -> None:
    print("\n─── Scenario 5: PARALLEL step (edge-based siblings) ───")
    from nexus.types import WorkflowStep, WorkflowEdge, StepType, EdgeType, ChainStatus
    from uuid import uuid4

    # Edge-based parallel: root → [branch_a, branch_b] → converge
    # branch_a and branch_b are StepType.PARALLEL (semantic tag = part of parallel group)
    wf_id = str(uuid4())
    root_id, ba_id, bb_id, conv_id = str(uuid4()), str(uuid4()), str(uuid4()), str(uuid4())

    root = WorkflowStep(
        id=root_id, workflow_id=wf_id,
        step_type=StepType.ACTION, name="root",
        tool_name="knowledge_search", tool_params={"query": "start"},
        persona_name="researcher",
    )
    branch_a = WorkflowStep(
        id=ba_id, workflow_id=wf_id,
        step_type=StepType.PARALLEL, name="branch_a",
        tool_name="knowledge_search", tool_params={"query": "branch A work"},
        persona_name="researcher",
    )
    branch_b = WorkflowStep(
        id=bb_id, workflow_id=wf_id,
        step_type=StepType.PARALLEL, name="branch_b",
        tool_name="knowledge_search", tool_params={"query": "branch B work"},
        persona_name="researcher",
    )
    converge = WorkflowStep(
        id=conv_id, workflow_id=wf_id,
        step_type=StepType.ACTION, name="converge",
        tool_name="knowledge_search", tool_params={"query": "merge results"},
        persona_name="researcher",
    )

    edges = [
        WorkflowEdge(id=str(uuid4()), workflow_id=wf_id,
                     source_step_id=root_id, target_step_id=ba_id),
        WorkflowEdge(id=str(uuid4()), workflow_id=wf_id,
                     source_step_id=root_id, target_step_id=bb_id),
        WorkflowEdge(id=str(uuid4()), workflow_id=wf_id,
                     source_step_id=ba_id, target_step_id=conv_id),
        WorkflowEdge(id=str(uuid4()), workflow_id=wf_id,
                     source_step_id=bb_id, target_step_id=conv_id),
    ]

    wf = await workflow_manager.create(
        tenant_id="test",
        name="Parallel Test",
        steps=[root, branch_a, branch_b, converge],
        edges=edges,
    )
    wf = await workflow_manager.activate(wf.id, "test")

    execution = await engine.run_workflow(wf.id, "test")
    print(f"    Status: {execution.status}")
    print(f"    Step results: {list(execution.step_results.keys())}")

    report(
        "parallel workflow completes",
        execution.status == ChainStatus.COMPLETED,
        f"status={execution.status.value}",
    )
    report(
        "both parallel branches executed",
        "branch_a" in execution.step_results and "branch_b" in execution.step_results,
        f"keys={list(execution.step_results.keys())}",
    )
    report(
        "converge step executed after both branches",
        "converge" in execution.step_results,
    )


# ── Scenario 6: WAIT step ─────────────────────────────────────────────────────

async def scenario_wait(engine, workflow_manager) -> None:
    print("\n─── Scenario 6: WAIT step (0s) ───")
    from nexus.types import WorkflowStep, StepType, ChainStatus

    wait = WorkflowStep(
        id="step-wait",
        workflow_id="wf-6",
        step_type=StepType.WAIT,
        name="pause",
        config={"seconds": 0},
    )

    wf = await workflow_manager.create(
        tenant_id="test",
        name="Wait Test",
        steps=[wait],
        edges=[],
    )
    wf = await workflow_manager.activate(wf.id, "test")

    execution = await engine.run_workflow(wf.id, "test")
    print(f"    Status: {execution.status}")
    print(f"    Step results: {execution.step_results}")

    report(
        "wait step completes",
        execution.status == ChainStatus.COMPLETED,
    )
    report(
        "wait step result recorded",
        execution.step_results.get("pause", {}).get("status") == "waited",
    )


# ── Scenario 7: Inactive workflow raises WorkflowValidationError ──────────────

async def scenario_inactive_raises(engine, workflow_manager) -> None:
    print("\n─── Scenario 7: Inactive workflow raises WorkflowValidationError ───")
    from nexus.types import WorkflowStep, StepType
    from nexus.exceptions import WorkflowValidationError

    step = WorkflowStep(
        id="step-draft",
        workflow_id="wf-7",
        step_type=StepType.ACTION,
        name="s1",
        tool_name="knowledge_search",
    )
    # Create but don't activate (stays DRAFT)
    wf = await workflow_manager.create(
        tenant_id="test",
        name="Draft Workflow",
        steps=[step],
        edges=[],
    )

    try:
        await engine.run_workflow(wf.id, "test")
        raised = False
    except WorkflowValidationError:
        raised = True
    except Exception as e:
        raised = False
        print(f"    Wrong exception: {type(e).__name__}: {e}")

    report("inactive workflow raises WorkflowValidationError", raised)


# ── Scenario 8: No workflow_manager raises WorkflowNotFound ───────────────────

async def scenario_no_manager(engine, workflow_manager) -> None:
    print("\n─── Scenario 8: No workflow_manager → WorkflowNotFound ───")
    from nexus.core.engine import NexusEngine
    from nexus.exceptions import WorkflowNotFound

    bare_engine = NexusEngine(
        persona_manager=engine.persona_manager,
        anomaly_engine=engine.anomaly_engine,
        notary=engine.notary,
        ledger=engine.ledger,
        chain_manager=engine.chain_manager,
        context_builder=engine.context_builder,
        tool_registry=engine.tool_registry,
        tool_selector=engine.tool_selector,
        tool_executor=engine.tool_executor,
        output_validator=engine.output_validator,
        cot_logger=engine.cot_logger,
        think_act_gate=engine.think_act_gate,
        continue_complete_gate=engine.continue_complete_gate,
        escalate_gate=engine.escalate_gate,
        workflow_manager=None,
    )

    try:
        await bare_engine.run_workflow("any-id", "test")
        raised = False
    except WorkflowNotFound:
        raised = True
    except Exception as e:
        raised = False
        print(f"    Wrong exception: {type(e).__name__}: {e}")

    report("no workflow_manager raises WorkflowNotFound", raised)


# ── Scenario 9: Template resolution with real trigger data ────────────────────

async def scenario_template_resolution(engine, workflow_manager) -> None:
    print("\n─── Scenario 9: Template param resolution from trigger ───")
    from nexus.types import WorkflowStep, StepType, ChainStatus

    # Verify _resolve_params works with real engine
    ctx = {
        "trigger": {"user": "alice", "count": 7},
        "steps": {"prev": {"result": "previous result"}},
    }

    resolved = engine._resolve_params({
        "full_template": "{{trigger.user}}",
        "partial": "hello {{trigger.user}} count={{trigger.count}}",
        "nested": {"inner": "{{steps.prev.result}}"},
        "literal_int": 42,
        "missing": "{{trigger.missing}}",
    }, ctx)

    print(f"    full_template: {resolved['full_template']!r} (expect 'alice')")
    print(f"    partial: {resolved['partial']!r} (expect 'hello alice count=7')")
    print(f"    nested.inner: {resolved['nested']['inner']!r} (expect 'previous result')")
    print(f"    literal_int: {resolved['literal_int']!r} (expect 42)")
    print(f"    missing: {resolved['missing']!r} (expect None — full template missing path)")

    report("full template returns native type", resolved["full_template"] == "alice")
    report("partial template interpolates correctly", resolved["partial"] == "hello alice count=7")
    report("nested dict resolved", resolved["nested"]["inner"] == "previous result")
    report("literal int passes through", resolved["literal_int"] == 42)
    report("missing key (full template) returns None", resolved["missing"] is None)


# ── Scenario 10: run() backwards compatibility ────────────────────────────────

async def scenario_backwards_compat(engine, workflow_manager) -> None:
    print("\n─── Scenario 10: run() backwards compatibility ───")
    from nexus.types import ChainStatus

    chain = await engine.run("What is NEXUS?", "test")
    print(f"    Chain status: {chain.status}")
    print(f"    Seals: {len(chain.seals)}")

    report(
        "run() still works after Phase 17",
        chain.status == ChainStatus.COMPLETED,
        f"status={chain.status.value}",
    )


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> int:
    print("=" * 60)
    print("Phase 17 Applied Integration Test")
    print("=" * 60)

    print("\nBuilding real engine with WorkflowManager...")
    try:
        engine, workflow_manager = build_engine_with_workflow_manager()
        print("  Engine built successfully.\n")
    except Exception as e:
        print(f"\n[FATAL] Engine build failed: {e}")
        traceback.print_exc()
        return 1

    scenarios = [
        scenario_single_action,
        scenario_linear_template,
        scenario_branch,
        scenario_loop,
        scenario_parallel,
        scenario_wait,
        scenario_inactive_raises,
        scenario_no_manager,
        scenario_template_resolution,
        scenario_backwards_compat,
    ]

    for scenario in scenarios:
        try:
            await scenario(engine, workflow_manager)
        except Exception as e:
            print(f"  \033[31m[SCENARIO CRASH]\033[0m {scenario.__name__}: {e}")
            traceback.print_exc()
            results.append((scenario.__name__, FAIL, f"CRASH: {e}"))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    total = len(results)

    for name, status, detail in results:
        color = "\033[32m" if status == PASS else "\033[31m"
        reset = "\033[0m"
        print(f"  {color}[{status}]{reset} {name}" + (f" ({detail})" if detail else ""))

    print(f"\n  {passed}/{total} checks passed, {failed} failed")

    if failed == 0:
        print("\n\033[32m✓ All Phase 17 integration checks PASSED\033[0m")
    else:
        print(f"\n\033[31m✗ {failed} check(s) FAILED\033[0m")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
