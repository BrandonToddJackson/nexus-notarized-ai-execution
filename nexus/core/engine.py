"""Core NEXUS engine. THE most important file. Wires EVERYTHING together.

Orchestrates: decompose → retrieve → plan → [gate → execute → validate] per step.

See NEXUS_BUILD_SPEC.md Section 8 (Phase 4) and Section 19 (Integration Wiring)
for the complete execution flow.
"""

import asyncio
import contextvars
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from nexus.types import (
    ChainPlan, Seal, PersonaContract,
    ActionStatus, ChainStatus, GateVerdict, ReasoningDecision,
    WorkflowDefinition, WorkflowExecution, WorkflowStatus,
    StepType, EdgeType, TriggerType,
)
from nexus.core.trust import maybe_degrade
from nexus.exceptions import (
    AnomalyDetected, ChainAborted, EscalationRequired,
    WorkflowNotFound, WorkflowValidationError,
)
from nexus.config import NexusConfig
from nexus.core.personas import PersonaManager
from nexus.core.anomaly import AnomalyEngine
from nexus.core.notary import Notary
from nexus.core.ledger import Ledger
from nexus.core.chain import ChainManager
from nexus.core.output_validator import OutputValidator
from nexus.core.cot_logger import CoTLogger
from nexus.knowledge.context import ContextBuilder
from nexus.tools.registry import ToolRegistry
from nexus.tools.selector import ToolSelector
from nexus.tools.executor import ToolExecutor
from nexus.credentials.vault import sanitize_tool_params
from nexus.reasoning.think_act import ThinkActGate
from nexus.reasoning.continue_complete import ContinueCompleteGate
from nexus.reasoning.escalate import EscalateGate

logger = logging.getLogger(__name__)

# Per-request callbacks override instance callbacks — async-safe via ContextVar
_request_callbacks: contextvars.ContextVar = contextvars.ContextVar("_nexus_callbacks", default=None)


class NexusEngine:
    """Single entry point. Orchestrates the full NEXUS execution pipeline.

    Constructor dependencies (all injected):
        - persona_manager: PersonaManager
        - anomaly_engine: AnomalyEngine
        - notary: Notary
        - ledger: Ledger
        - chain_manager: ChainManager
        - context_builder: ContextBuilder
        - tool_registry: ToolRegistry
        - tool_selector: ToolSelector
        - tool_executor: ToolExecutor
        - output_validator: OutputValidator
        - cot_logger: CoTLogger
        - think_act_gate: ThinkActGate
        - continue_complete_gate: ContinueCompleteGate
        - escalate_gate: EscalateGate
        - llm_client: LLMClient (from Phase 6)
        - config: NexusConfig
    """

    def __init__(
        self,
        persona_manager: PersonaManager,
        anomaly_engine: AnomalyEngine,
        notary: Notary,
        ledger: Ledger,
        chain_manager: ChainManager,
        context_builder: ContextBuilder,
        tool_registry: ToolRegistry,
        tool_selector: ToolSelector,
        tool_executor: ToolExecutor,
        output_validator: OutputValidator,
        cot_logger: CoTLogger,
        think_act_gate: ThinkActGate,
        continue_complete_gate: ContinueCompleteGate,
        escalate_gate: EscalateGate,
        llm_client=None,  # Phase 6
        cost_tracker=None,  # Phase 6
        config: NexusConfig = None,
        callbacks: list = None,
        workflow_manager=None,  # Phase 17
        event_bus=None,  # Phase 22
    ):
        self.persona_manager = persona_manager
        self.anomaly_engine = anomaly_engine
        self.notary = notary
        self.ledger = ledger
        self.chain_manager = chain_manager
        self.context_builder = context_builder
        self.tool_registry = tool_registry
        self.tool_selector = tool_selector
        self.tool_executor = tool_executor
        self.output_validator = output_validator
        self.cot_logger = cot_logger
        self.think_act_gate = think_act_gate
        self.continue_complete_gate = continue_complete_gate
        self.escalate_gate = escalate_gate
        self.llm_client = llm_client
        self.cost_tracker = cost_tracker
        self.config = config or NexusConfig()
        self.callbacks = callbacks or []
        self.workflow_manager = workflow_manager  # Phase 17
        self._event_bus = event_bus  # Phase 22

    async def run(self, task: str, tenant_id: str, persona_name: str = None, callbacks: list = None) -> ChainPlan:
        """FULL EXECUTION LOOP.

        1. DECOMPOSE: Ask LLM to break task into chain steps
           → chain = chain_manager.create_chain(tenant_id, task, steps)

        2. For each step in chain:
           a. RETRIEVE: context_builder.build()
           b. THINK/ACT GATE: think_act_gate.decide() — loop if THINK (max 3)
           c. PLAN: tool_selector.select()
           d. ACTIVATE PERSONA: persona_manager.activate()
           e. GATE CHECK: anomaly_engine.check()
           f. CREATE SEAL: notary.create_seal() — PENDING
           g. If FAIL: finalize BLOCKED, raise AnomalyDetected
           h. VERIFY + EXECUTE: tool_executor.execute()
           i. VALIDATE OUTPUT: output_validator.validate()
           j. FINALIZE SEAL: notary.finalize_seal()
           k. APPEND: ledger.append()
           l. COST: cost_tracker.record() if available
           m. REVOKE PERSONA: persona_manager.revoke()
           n. CONTINUE/COMPLETE: continue_complete_gate.decide()

        3. Return completed chain with all seal IDs

        Args:
            task: The user's task (e.g., "Analyze customer churn from Q3 data")
            tenant_id: Tenant context
            persona_name: Optional persona override. If None, auto-selected per step.

        Returns:
            Completed ChainPlan with all seal IDs
        """
        _tok = _request_callbacks.set(callbacks) if callbacks is not None else None
        try:
            return await self._run(task, tenant_id, persona_name)
        finally:
            if _tok is not None:
                _request_callbacks.reset(_tok)

    async def _run(self, task: str, tenant_id: str, persona_name: str = None) -> ChainPlan:
        """Internal run — called by run() after context var is set."""
        logger.info(f"[Engine] run() task={task!r} tenant={tenant_id}")
        await self._fire_callbacks("task_started", {"task": task, "tenant_id": tenant_id})

        # 1. DECOMPOSE task into steps
        try:
            steps = await self._decompose_task(task, tenant_id)
        except Exception as exc:
            raise ChainAborted(
                f"Task decomposition failed: {exc}",
                completed_steps=0,
                total_steps=0,
            ) from exc

        if not steps:
            raise ChainAborted("Task produced empty decomposition", completed_steps=0, total_steps=0)

        # Create immutable chain plan
        chain = self.chain_manager.create_chain(tenant_id, task, steps)
        logger.info(f"[Engine] Chain {chain.id} created with {len(steps)} step(s)")
        await self._fire_callbacks("chain_created", {
            "chain_id": chain.id,
            "steps": len(steps),
            "task": task,
        })

        # 2. Execute each step
        session_history: list[dict] = []
        retry_counts: dict[str, int] = {}
        step_index = 0

        while step_index < len(chain.steps):
            step = chain.steps[step_index]
            retry_key = f"{chain.id}:{step_index}"
            retry_count = retry_counts.get(retry_key, 0)

            # Determine persona for this step
            step_persona_name = persona_name or step.get("persona") or "researcher"
            step_action = step.get("action", task)

            logger.info(
                f"[Engine] Step {step_index}/{len(chain.steps)-1}: "
                f"action={step_action!r} persona={step_persona_name!r}"
            )
            await self._fire_callbacks("step_started", {
                "chain_id": chain.id,
                "step_index": step_index,
                "step": step,
                "persona": step_persona_name,
            })

            # Track whether persona was activated so we can safely revoke in finally
            activation_done = False
            seal: Optional[Seal] = None
            result: Any = None

            try:
                # Resolve persona contract (without official activation yet)
                persona_contract = self._resolve_persona(step_persona_name)

                # ── a. RETRIEVE: Build context (think/act loop, max 3) ─────────
                context = None
                loop_count = 0
                while True:
                    context = await self.context_builder.build(
                        tenant_id=tenant_id,
                        task=step_action,
                        persona=persona_contract,
                        session_history=session_history,
                    )
                    decision = self.think_act_gate.decide(context, loop_count)
                    if decision == ReasoningDecision.ACT:
                        break
                    loop_count += 1
                    logger.debug(f"[Engine] THINK loop {loop_count}, confidence={context.confidence:.2f}")

                cot_key = f"{chain.id}:{step_index}"
                self.cot_logger.log(
                    cot_key,
                    f"Context built: confidence={context.confidence:.2f}, think_loops={loop_count}",
                )

                # ── c. PLAN: Select tool ──────────────────────────────────────
                intent = await self.tool_selector.select(
                    task=step_action,
                    persona=persona_contract,
                    context=context,
                )

                # Honour the step's explicit tool hint when selector returns empty
                if not intent.tool_name and step.get("tool"):
                    step_params = step.get("params", {})
                    intent = intent.model_copy(update={
                        "tool_name": step["tool"],
                        "tool_params": step_params,
                        "resource_targets": list(step_params.values()),
                    })

                self.cot_logger.log(
                    cot_key,
                    f"Tool selected: {intent.tool_name!r}, reasoning={intent.reasoning!r}",
                )

                # ── d. ACTIVATE PERSONA (officially starts TTL clock) ────────
                activated_persona = self.persona_manager.activate(step_persona_name, tenant_id)
                activation_done = True
                activation_time = self.persona_manager.get_activation_time(step_persona_name) or datetime.now(timezone.utc)

                self.cot_logger.log(cot_key, f"Persona '{step_persona_name}' activated")

                # ── e. ANOMALY CHECK (4 gates) ────────────────────────────────
                anomaly_result = await self.anomaly_engine.check(
                    persona=activated_persona,
                    intent=intent,
                    activation_time=activation_time,
                    tenant_id=tenant_id,
                )

                await self._fire_callbacks("anomaly_checked", {
                    "chain_id": chain.id,
                    "step_index": step_index,
                    "verdict": anomaly_result.overall_verdict,
                    "gates": [g.model_dump() for g in anomaly_result.gates],
                })

                for gate in anomaly_result.gates:
                    self.cot_logger.log(
                        cot_key,
                        f"Gate {gate.gate_name}: verdict={gate.verdict} score={gate.score:.3f}",
                    )

                # ── f. CREATE SEAL (PENDING) ───────────────────────────────────
                # Sanitize tool_params before sealing so secrets never enter the ledger.
                sealed_intent = intent.model_copy(
                    update={"tool_params": sanitize_tool_params(intent.tool_params)}
                )
                seal = self.notary.create_seal(
                    chain_id=chain.id,
                    step_index=step_index,
                    tenant_id=tenant_id,
                    persona_id=step_persona_name,
                    intent=sealed_intent,
                    anomaly_result=anomaly_result,
                )

                # ── g. BLOCK if any gate failed ────────────────────────────────
                if anomaly_result.overall_verdict == GateVerdict.FAIL:
                    failed_gates = [
                        g.details for g in anomaly_result.gates
                        if g.verdict == GateVerdict.FAIL
                    ]
                    block_reason = f"Anomaly gates failed: {failed_gates}"
                    seal = self.notary.finalize_seal(
                        seal, None, ActionStatus.BLOCKED, error=block_reason
                    )
                    cot_trace = self.cot_logger.get_trace(cot_key)
                    seal = seal.model_copy(update={"cot_trace": cot_trace})
                    await self.ledger.append(seal)
                    self.cot_logger.clear(cot_key)

                    # Degrade trust tier on gate failure (persona's track record worsens)
                    try:
                        degraded = maybe_degrade(activated_persona)
                        self.persona_manager._contracts[step_persona_name] = degraded
                    except Exception:
                        pass  # trust update is best-effort; never block execution path

                    self.persona_manager.revoke(step_persona_name)
                    activation_done = False

                    await self._fire_callbacks("step_blocked", {
                        "chain_id": chain.id,
                        "step_index": step_index,
                        "seal_id": seal.id,
                        "reason": block_reason,
                    })

                    raise AnomalyDetected(
                        f"Action blocked at step {step_index}: {block_reason}",
                        gate_results=anomaly_result.gates,
                        chain_id=str(chain.id),
                    )

                # ── h+i. VERIFY + EXECUTE (executor handles both) ─────────────
                self.cot_logger.log(cot_key, f"Executing tool '{intent.tool_name}'")
                result, error_str = await self.tool_executor.execute(
                    intent, tenant_id=tenant_id, persona_name=step_persona_name
                )

                # ── j. VALIDATE OUTPUT ────────────────────────────────────────
                is_valid, validation_reason = await self.output_validator.validate(intent, result)
                if not is_valid:
                    logger.warning(
                        f"[Engine] Step {step_index} output validation: {validation_reason}"
                    )
                self.cot_logger.log(
                    cot_key,
                    f"Output valid={is_valid}: {validation_reason}",
                )

                # ── k. FINALIZE SEAL ──────────────────────────────────────────
                status = ActionStatus.EXECUTED if not error_str else ActionStatus.FAILED
                cot_trace = self.cot_logger.get_trace(cot_key)
                seal = self.notary.finalize_seal(seal, result, status, error=error_str)
                seal = seal.model_copy(update={"cot_trace": cot_trace})
                self.cot_logger.clear(cot_key)

                # ── l. APPEND TO LEDGER ────────────────────────────────────────
                await self.ledger.append(seal)

                # ── l2. STORE FINGERPRINT (Gate 4 drift baseline) ─────────────
                if (
                    self.anomaly_engine.fingerprint_store is not None
                    and not isinstance(self.anomaly_engine.fingerprint_store, dict)
                    and status == ActionStatus.EXECUTED
                ):
                    try:
                        await self.anomaly_engine.fingerprint_store.store(
                            tenant_id, step_persona_name, anomaly_result.action_fingerprint
                        )
                    except Exception as fp_exc:
                        logger.warning(f"[Engine] Fingerprint storage failed: {fp_exc}")

                # ── m. RECORD COST (if available) ──────────────────────────────
                if self.cost_tracker is not None:
                    try:
                        # LLM usage is tracked separately; provide zero-usage record
                        # for non-LLM steps so the chain is fully accounted for
                        await self.cost_tracker.record(
                            tenant_id=tenant_id,
                            chain_id=chain.id,
                            seal_id=seal.id,
                            model=self.config.default_llm_model,
                            usage={"input_tokens": 0, "output_tokens": 0},
                        )
                    except Exception as cost_exc:
                        logger.warning(f"[Engine] Cost tracking failed: {cost_exc}")

                # ── n. REVOKE PERSONA ─────────────────────────────────────────
                self.persona_manager.revoke(step_persona_name)
                activation_done = False

                # Advance chain (records seal_id, updates status)
                chain = self.chain_manager.advance(chain, seal.id)

                await self._fire_callbacks("step_completed", {
                    "chain_id": chain.id,
                    "step_index": step_index,
                    "seal_id": seal.id,
                    "status": status.value,
                    "tool": intent.tool_name,
                })

                # Accumulate session history for context continuity
                session_history.append({
                    "step_index": step_index,
                    "action": step_action,
                    "tool": intent.tool_name,
                    "result": result,
                    "status": status.value,
                })

            except AnomalyDetected:
                # Already sealed, logged, and persona revoked inside the block above.
                # Fail the chain and re-raise — caller decides how to surface this.
                if activation_done:
                    self.persona_manager.revoke(step_persona_name)
                chain = self.chain_manager.fail(chain, f"AnomalyDetected at step {step_index}")
                raise

            except (ChainAborted, EscalationRequired):
                if activation_done:
                    self.persona_manager.revoke(step_persona_name)
                raise

            except Exception as exc:
                logger.error(
                    f"[Engine] Unhandled error at step {step_index}: {exc}",
                    exc_info=True,
                )
                if activation_done:
                    self.persona_manager.revoke(step_persona_name)
                    activation_done = False

                # Ask EscalateGate whether to retry or escalate
                gate_decision = self.escalate_gate.decide(exc, retry_count, chain)

                if gate_decision == ReasoningDecision.RETRY:
                    retry_counts[retry_key] = retry_count + 1
                    logger.info(
                        f"[Engine] Retrying step {step_index} "
                        f"(attempt {retry_counts[retry_key]})"
                    )
                    # Don't advance step_index — re-run this step
                    continue

                # ESCALATE: seal the failure, fail the chain, raise
                if seal is not None:
                    try:
                        cot_key = f"{chain.id}:{step_index}"
                        cot_trace = self.cot_logger.get_trace(cot_key)
                        seal = self.notary.finalize_seal(
                            seal, None, ActionStatus.FAILED, error=str(exc)
                        )
                        seal = seal.model_copy(update={"cot_trace": cot_trace})
                        self.cot_logger.clear(cot_key)
                        await self.ledger.append(seal)
                    except Exception as seal_exc:
                        logger.warning(f"[Engine] Failed to seal error: {seal_exc}")

                escalation_ctx = self.escalate_gate.build_escalation_context(chain, exc)
                chain = self.chain_manager.escalate(chain, str(exc))
                await self._fire_callbacks("chain_escalated", {
                    "chain_id": chain.id,
                    "context": escalation_ctx,
                })
                raise EscalationRequired(
                    f"Chain escalated at step {step_index}: {exc}",
                    context=escalation_ctx,
                ) from exc

            # ── o. CONTINUE / COMPLETE GATE ────────────────────────────────────
            cc_decision = self.continue_complete_gate.decide(chain, result, seal)

            if cc_decision == ReasoningDecision.COMPLETE:
                logger.info(f"[Engine] Chain {chain.id} complete after step {step_index}")
                break

            elif cc_decision == ReasoningDecision.CONTINUE:
                step_index += 1

            elif cc_decision == ReasoningDecision.RETRY:
                # The seal already recorded a FAILED status; treat as continue to
                # next step rather than looping infinitely on a bad step.
                logger.info(f"[Engine] Step {step_index} failed seal — moving to next step")
                step_index += 1

            elif cc_decision == ReasoningDecision.ESCALATE:
                escalation_ctx = self.escalate_gate.build_escalation_context(
                    chain, Exception(f"ContinueCompleteGate returned ESCALATE at step {step_index}")
                )
                chain = self.chain_manager.escalate(chain, "Escalated by ContinueCompleteGate")
                await self._fire_callbacks("chain_escalated", {
                    "chain_id": chain.id,
                    "context": escalation_ctx,
                })
                raise EscalationRequired(
                    f"Chain escalated after step {step_index}: continue/complete gate",
                    context=escalation_ctx,
                )

            else:
                # Unknown decision — advance normally
                step_index += 1

        # Mark chain COMPLETED if not already in a terminal state
        if chain.status not in (ChainStatus.COMPLETED, ChainStatus.FAILED, ChainStatus.ESCALATED):
            chain = chain.model_copy(update={
                "status": ChainStatus.COMPLETED,
                "completed_at": datetime.now(timezone.utc),
            })

        logger.info(f"[Engine] Chain {chain.id} finished: status={chain.status}")
        await self._fire_callbacks("chain_completed", {
            "chain_id": chain.id,
            "status": chain.status.value,
            "seal_count": len(chain.seals),
        })

        return chain

    async def _decompose_task(self, task: str, tenant_id: str) -> list[dict]:
        """Ask LLM to break task into steps.

        Uses DECOMPOSE_TASK prompt from nexus.llm.prompts.
        Parse LLM response as JSON array of step objects.

        If no LLM client, return a single-step plan using the first available tool.

        Args:
            task: Original user task
            tenant_id: Tenant context

        Returns:
            List of step dicts: [{"action": "...", "tool": "...", "params": {...}, "persona": "..."}, ...]
        """
        if self.llm_client is not None:
            from nexus.llm.prompts import DECOMPOSE_TASK

            tools = self.tool_registry.list_tools()
            personas = self.persona_manager.list_personas()

            tool_list = "\n".join(
                f"- {t.name}: {t.description}" for t in tools
            ) or "No tools available"
            persona_list = "\n".join(
                f"- {p.name}: {p.description}" for p in personas
            ) or "No personas available"

            system_prompt = DECOMPOSE_TASK.format(
                tool_list=tool_list,
                persona_list=persona_list,
            )

            try:
                response = await self.llm_client.complete(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps({"task": task})},
                    ]
                )
                raw = response.get("content", "").strip()

                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("```", 2)[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                steps = json.loads(raw)
                if isinstance(steps, list) and steps:
                    logger.debug(f"[Engine] LLM decomposed into {len(steps)} step(s)")
                    return steps

            except Exception as exc:
                logger.warning(
                    f"[Engine] LLM decomposition failed ({exc}), falling back to single step"
                )

        # Fallback: single-step plan using first available tool
        tools = self.tool_registry.list_tools()
        first_tool = tools[0].name if tools else "knowledge_search"

        personas = self.persona_manager.list_personas()
        default_persona = personas[0].name if personas else "researcher"

        return [{
            "action": task,
            "tool": first_tool,
            "params": {"query": task},
            "persona": default_persona,
        }]

    # ── Private helpers ──────────────────────────────────────────────────────

    def _resolve_persona(self, persona_name: str) -> PersonaContract:
        """Return a persona contract by name, falling back to the first available.

        Does NOT officially activate the persona (no TTL clock started).
        Used for context building before the security activation in step d.
        """
        contract = self.persona_manager.get_persona(persona_name)
        if contract is not None:
            return contract

        # Fallback: use any loaded persona
        all_personas = self.persona_manager.list_personas()
        if all_personas:
            logger.warning(
                f"[Engine] Persona '{persona_name}' not found, using '{all_personas[0].name}'"
            )
            return all_personas[0]

        # Last resort: minimal synthetic persona (should not happen in production)
        from nexus.types import PersonaContract, RiskLevel
        logger.error("[Engine] No personas loaded — using synthetic fallback")
        return PersonaContract(
            name=persona_name,
            description="Synthetic fallback persona",
            allowed_tools=[],
            resource_scopes=["*"],
            intent_patterns=["do anything"],
            risk_tolerance=RiskLevel.LOW,
        )

    async def _fire_callbacks(self, event: str, data: dict) -> None:
        """Invoke all registered callbacks for a lifecycle event."""
        cbs = _request_callbacks.get()
        if cbs is None:
            cbs = self.callbacks
        if not cbs:
            return
        for cb in cbs:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event, data)
                else:
                    cb(event, data)
            except Exception as cb_exc:
                logger.warning(f"[Engine] Callback error on '{event}': {cb_exc}")

    # ── Phase 17: DAG Workflow Execution ─────────────────────────────────────

    # Compiled once for template resolution
    _TEMPLATE_RE = re.compile(r"\{\{([^}]+)\}\}")

    async def run_workflow(
        self,
        workflow_id: str,
        tenant_id: str,
        trigger_data: Optional[dict] = None,
        persona_override: Optional[str] = None,
    ) -> WorkflowExecution:
        """Execute a WorkflowDefinition as a DAG.

        Loads the workflow, verifies it is ACTIVE, creates a WorkflowExecution
        and a backing ChainPlan (for Merkle-sealed audit trail), then drives
        execution via _execute_dag_layer starting from the DAG entry points.

        Args:
            workflow_id:      ID of the WorkflowDefinition to run.
            tenant_id:        Tenant context.
            trigger_data:     Key/value payload from the trigger (webhook body,
                              cron metadata, etc.).  Available as
                              ``{{trigger.<key>}}`` in step param templates.
            persona_override: Force every step to use this persona instead of
                              the per-step persona_name.

        Returns:
            Completed (or failed) WorkflowExecution with step_results populated.

        Raises:
            WorkflowNotFound:      if workflow_id does not exist for tenant.
            WorkflowValidationError: if the workflow is not in ACTIVE status.
        """
        from nexus.workflows.dag import get_entry_points  # lazy — avoids circular at module load

        trigger_data = trigger_data or {}

        if self.workflow_manager is None:
            raise WorkflowNotFound(
                "NexusEngine has no workflow_manager configured.",
                workflow_id=workflow_id,
            )

        workflow = await self.workflow_manager.get(workflow_id, tenant_id)

        if workflow.status != WorkflowStatus.ACTIVE:
            raise WorkflowValidationError(
                f"Workflow '{workflow_id}' is not ACTIVE (status={workflow.status.value}). "
                "Call workflow_manager.activate() before running.",
                violations=[f"status={workflow.status.value}"],
            )

        # Create a WorkflowExecution record
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_version=workflow.version,
            tenant_id=tenant_id,
            trigger_type=TriggerType.MANUAL,
            trigger_data=trigger_data,
            status=ChainStatus.EXECUTING,
            started_at=datetime.now(timezone.utc),
        )

        # One chain per workflow execution — all step seals are attached to it
        steps_meta = [
            {"action": s.name, "persona": persona_override or s.persona_name}
            for s in workflow.steps
        ]
        chain = self.chain_manager.create_chain(
            tenant_id, f"workflow:{workflow_id}", steps_meta
        )
        execution = execution.model_copy(update={"chain_id": chain.id})

        logger.info(
            f"[Engine] run_workflow wf={workflow_id} chain={chain.id} "
            f"steps={len(workflow.steps)}"
        )
        await self._fire_callbacks("workflow_started", {
            "workflow_id": workflow_id,
            "chain_id": chain.id,
            "tenant_id": tenant_id,
        })

        # Mutable execution context shared across all steps
        context: dict[str, Any] = {
            "steps": {},            # step_name → {"result": ..., "status": ...}
            "trigger": trigger_data,
            "loop_current": None,
            "loop_index": 0,
            "_done": set(),         # set of step_ids already executed
        }

        # Global step-index counter (mutable list for pass-by-reference)
        step_index_counter = [0]

        try:
            # Exclude LOOP_BACK edges when computing entry points — loop steps have
            # a self-referential LOOP_BACK edge that would otherwise hide them from
            # the entry-point check (same logic used by WorkflowValidator).
            dag_edges = [e for e in workflow.edges if e.edge_type != EdgeType.LOOP_BACK]
            entry_ids = get_entry_points(workflow.steps, dag_edges)
            if not entry_ids:
                raise WorkflowValidationError(
                    "Workflow has no entry points.", violations=["no entry points"]
                )

            await self._execute_dag_layer(
                entry_ids, workflow, chain, context,
                step_index_counter, tenant_id, persona_override,
            )

            execution = execution.model_copy(update={
                "status": ChainStatus.COMPLETED,
                "completed_at": datetime.now(timezone.utc),
                "step_results": dict(context["steps"]),
            })
            await self._fire_callbacks("workflow_completed", {
                "workflow_id": workflow_id,
                "chain_id": chain.id,
                "step_count": step_index_counter[0],
            })
            if self._event_bus is not None:
                try:
                    await self._event_bus.emit("workflow.completed", {
                        "workflow_id": workflow_id,
                        "execution_id": execution.id,
                        "tenant_id": execution.tenant_id,
                        "result": execution.step_results,
                    })
                except Exception:
                    logger.exception("EventBus.emit(workflow.completed) failed — non-fatal")

        except (WorkflowNotFound, WorkflowValidationError):
            raise

        except Exception as exc:
            logger.error(f"[Engine] run_workflow failed: {exc}", exc_info=True)
            execution = execution.model_copy(update={
                "status": ChainStatus.FAILED,
                "completed_at": datetime.now(timezone.utc),
                "error": str(exc),
                "step_results": dict(context["steps"]),
            })
            await self._fire_callbacks("workflow_failed", {
                "workflow_id": workflow_id,
                "chain_id": chain.id,
                "error": str(exc),
            })
            if self._event_bus is not None:
                try:
                    await self._event_bus.emit("workflow.failed", {
                        "workflow_id": workflow_id,
                        "execution_id": execution.id,
                        "tenant_id": execution.tenant_id,
                        "error": str(exc),
                    })
                except Exception:
                    logger.exception("EventBus.emit(workflow.failed) failed — non-fatal")
            raise

        return execution

    async def _execute_dag_layer(
        self,
        step_ids: list[str],
        workflow: WorkflowDefinition,
        chain: ChainPlan,
        context: dict,
        step_index_counter: list[int],
        tenant_id: str,
        persona_override: Optional[str],
    ) -> None:
        """Execute one layer of the DAG then recurse into the next ready layer.

        A "layer" is a set of step_ids that are currently ready (all non-LOOP_BACK
        predecessors have completed).  Steps are executed sequentially within a
        layer *unless* their type is PARALLEL, which fans out with asyncio.gather.

        After all steps in the layer finish, the method collects their children
        (evaluating CONDITIONAL edges), deduplicates, checks parent readiness,
        and recurses.
        """
        from nexus.workflows.dag import get_children, get_parents, evaluate_condition

        step_map = {s.id: s for s in workflow.steps}
        completed_ids: list[str] = []

        for step_id in step_ids:
            if step_id in context["_done"]:
                continue  # guard against duplicate entries in the layer list

            step = step_map.get(step_id)
            if step is None:
                logger.warning(f"[Engine] DAG layer: step '{step_id}' not in workflow")
                continue

            step_index = step_index_counter[0]
            step_index_counter[0] += 1

            logger.info(
                f"[Engine] DAG step {step_index} id={step_id} "
                f"type={step.step_type.value} name={step.name!r}"
            )

            try:
                if step.step_type == StepType.ACTION:
                    await self._execute_action_step(
                        step, workflow, chain, context, step_index, tenant_id, persona_override
                    )

                elif step.step_type == StepType.BRANCH:
                    await self._execute_branch_step(
                        step, workflow, chain, context, step_index_counter, tenant_id, persona_override
                    )

                elif step.step_type == StepType.LOOP:
                    await self._execute_loop_step(
                        step, workflow, chain, context, step_index_counter, tenant_id, persona_override
                    )

                elif step.step_type == StepType.PARALLEL:
                    await self._execute_parallel_step(
                        step, workflow, chain, context, step_index_counter, tenant_id, persona_override,
                        step_index=step_index,
                    )

                elif step.step_type == StepType.SUB_WORKFLOW:
                    await self._execute_sub_workflow_step(
                        step, context, tenant_id, persona_override
                    )

                elif step.step_type == StepType.WAIT:
                    delay = float(step.config.get("seconds", 1))
                    await asyncio.sleep(delay)
                    context["steps"][step.name] = {"status": "waited", "seconds": delay}

                elif step.step_type == StepType.HUMAN_APPROVAL:
                    await self._execute_approval_step(
                        step, chain, context, step_index, tenant_id
                    )

                else:
                    logger.warning(
                        f"[Engine] Unknown step type '{step.step_type}' — skipping"
                    )
                    context["steps"][step.name] = {"status": "skipped"}

            except (AnomalyDetected, EscalationRequired):
                # Propagate gate/escalation failures — run_workflow marks as FAILED.
                raise

            except Exception as exc:
                logger.error(
                    f"[Engine] Step '{step.name}' failed: {exc}", exc_info=True
                )
                context["steps"][step.name] = {"status": "failed", "error": str(exc)}
                raise

            context["_done"].add(step_id)
            completed_ids.append(step_id)

        # ── Determine next ready layer ────────────────────────────────────
        candidate_ids: list[str] = []
        seen: set[str] = set()

        for step_id in completed_ids:
            for child_id, edge in get_children(step_id, workflow.edges):
                if edge.edge_type == EdgeType.LOOP_BACK:
                    continue  # loop-back edges are consumed by _execute_loop_step
                if child_id in seen or child_id in context["_done"]:
                    continue

                # Evaluate conditional edges
                if edge.condition and edge.condition.lower() not in ("", "default"):
                    try:
                        passes = evaluate_condition(edge.condition, context)
                    except Exception as cond_exc:
                        logger.warning(
                            f"[Engine] Condition eval failed on edge→{child_id}: {cond_exc}"
                        )
                        passes = False
                    if not passes:
                        continue

                seen.add(child_id)
                candidate_ids.append(child_id)

        # Only include candidates whose non-LOOP_BACK parents are all done
        ready_ids: list[str] = []
        for child_id in candidate_ids:
            parents = get_parents(child_id, workflow.edges)
            blocking_parents = [
                pid for pid, e in parents
                if e.edge_type != EdgeType.LOOP_BACK and pid not in context["_done"]
            ]
            if not blocking_parents:
                ready_ids.append(child_id)

        if ready_ids:
            await self._execute_dag_layer(
                ready_ids, workflow, chain, context,
                step_index_counter, tenant_id, persona_override,
            )

    async def _execute_action_step(
        self,
        step,
        workflow: WorkflowDefinition,
        chain: ChainPlan,
        context: dict,
        step_index: int,
        tenant_id: str,
        persona_override: Optional[str],
    ) -> Any:
        """Run one ACTION step through the full NEXUS gate pipeline.

        Mirrors the per-step logic in _run() but sources the tool name, params,
        and persona from the WorkflowStep and resolves ``{{...}}`` templates from
        the execution context before activation.
        """
        step_persona_name = persona_override or step.persona_name or "researcher"
        step_action = step.description or step.name
        resolved_params = self._resolve_params(step.tool_params, context)

        activation_done = False
        seal = None
        result: Any = None
        cot_key = f"{chain.id}:wf:{step.id}"

        try:
            # Resolve persona contract (no TTL yet)
            persona_contract = self._resolve_persona(step_persona_name)

            # Build retrieval context (think/act loop, max 3)
            kb_context = None
            loop_count = 0
            while True:
                kb_context = await self.context_builder.build(
                    tenant_id=tenant_id,
                    task=step_action,
                    persona=persona_contract,
                    session_history=[],
                )
                decision = self.think_act_gate.decide(kb_context, loop_count)
                if decision == ReasoningDecision.ACT:
                    break
                loop_count += 1

            # Select tool (for intent declaration; overridden by step.tool_name below)
            intent = await self.tool_selector.select(
                task=step_action,
                persona=persona_contract,
                context=kb_context,
            )

            # Override with step-specified tool if provided.
            # resource_targets is intentionally NOT overridden here — the selector
            # already computed correctly-prefixed targets (e.g. "kb:search_slug")
            # that match the persona's resource_scopes.  Overriding with raw param
            # values (e.g. "What is NEXUS?") would fail Gate 1 scope check.
            if step.tool_name:
                intent = intent.model_copy(update={
                    "tool_name": step.tool_name,
                    "tool_params": resolved_params,
                })
            elif resolved_params:
                intent = intent.model_copy(update={"tool_params": resolved_params})

            self.cot_logger.log(
                cot_key,
                f"[wf] Tool selected: {intent.tool_name!r} params={resolved_params}",
            )

            # Activate persona (starts TTL clock)
            activated_persona = self.persona_manager.activate(step_persona_name, tenant_id)
            activation_done = True
            activation_time = (
                self.persona_manager.get_activation_time(step_persona_name)
                or datetime.now(timezone.utc)
            )

            # 4 anomaly gates
            anomaly_result = await self.anomaly_engine.check(
                persona=activated_persona,
                intent=intent,
                activation_time=activation_time,
                tenant_id=tenant_id,
            )

            # Create PENDING seal — sanitize tool_params so secrets never enter the ledger
            sealed_intent = intent.model_copy(
                update={"tool_params": sanitize_tool_params(intent.tool_params)}
            )
            seal = self.notary.create_seal(
                chain_id=chain.id,
                step_index=step_index,
                tenant_id=tenant_id,
                persona_id=step_persona_name,
                intent=sealed_intent,
                anomaly_result=anomaly_result,
            )

            # Block if any gate failed
            if anomaly_result.overall_verdict == GateVerdict.FAIL:
                failed_gates = [
                    g.details for g in anomaly_result.gates
                    if g.verdict == GateVerdict.FAIL
                ]
                block_reason = f"Anomaly gates failed: {failed_gates}"
                cot_trace = self.cot_logger.get_trace(cot_key)
                seal = self.notary.finalize_seal(seal, None, ActionStatus.BLOCKED, error=block_reason)
                seal = seal.model_copy(update={"cot_trace": cot_trace})
                self.cot_logger.clear(cot_key)
                await self.ledger.append(seal)

                try:
                    degraded = maybe_degrade(activated_persona)
                    self.persona_manager._contracts[step_persona_name] = degraded
                except Exception:
                    pass

                self.persona_manager.revoke(step_persona_name)
                activation_done = False

                context["steps"][step.name] = {
                    "status": "blocked",
                    "seal_id": seal.id,
                    "reason": block_reason,
                }
                raise AnomalyDetected(
                    f"Workflow step '{step.name}' blocked: {block_reason}",
                    gate_results=anomaly_result.gates,
                    chain_id=str(chain.id),
                )

            # Execute tool
            self.cot_logger.log(cot_key, f"[wf] Executing tool '{intent.tool_name}'")
            result, error_str = await self.tool_executor.execute(
                intent, tenant_id=tenant_id, persona_name=step_persona_name
            )

            # Validate output
            is_valid, validation_reason = await self.output_validator.validate(intent, result)
            self.cot_logger.log(
                cot_key, f"[wf] Output valid={is_valid}: {validation_reason}"
            )

            # Finalize seal
            status = ActionStatus.EXECUTED if not error_str else ActionStatus.FAILED
            cot_trace = self.cot_logger.get_trace(cot_key)
            seal = self.notary.finalize_seal(seal, result, status, error=error_str)
            seal = seal.model_copy(update={"cot_trace": cot_trace})
            self.cot_logger.clear(cot_key)
            await self.ledger.append(seal)

            # Revoke persona
            self.persona_manager.revoke(step_persona_name)
            activation_done = False

            # Store result in execution context
            context["steps"][step.name] = {
                "result": result,
                "status": status.value,
                "seal_id": seal.id,
            }
            return result

        finally:
            if activation_done:
                try:
                    self.persona_manager.revoke(step_persona_name)
                except Exception:
                    pass

    async def _execute_branch_step(
        self,
        step,
        workflow: WorkflowDefinition,
        chain: ChainPlan,
        context: dict,
        step_index_counter: list[int],
        tenant_id: str,
        persona_override: Optional[str],
    ) -> None:
        """Evaluate conditional edges and execute the first matching branch.

        Edge evaluation order: CONDITIONAL edges first (in definition order),
        then the DEFAULT edge as fallback.  Stores the chosen target step name
        in context so downstream templates can reference it.
        """
        from nexus.workflows.dag import get_children, evaluate_condition

        children = get_children(step.id, workflow.edges)

        # Separate conditional from default edges
        conditional_edges = [
            (cid, e) for cid, e in children
            if e.edge_type == EdgeType.CONDITIONAL
        ]
        default_edges = [
            (cid, e) for cid, e in children
            if e.edge_type == EdgeType.DEFAULT
        ]

        chosen_id: Optional[str] = None

        for child_id, edge in conditional_edges:
            if not edge.condition or edge.condition.lower() in ("", "default"):
                continue
            try:
                if evaluate_condition(edge.condition, context):
                    chosen_id = child_id
                    break
            except Exception as exc:
                logger.warning(
                    f"[Engine] Branch '{step.name}' condition error: {exc}"
                )

        if chosen_id is None and default_edges:
            chosen_id = default_edges[0][0]

        if chosen_id is None:
            logger.warning(
                f"[Engine] Branch '{step.name}' has no matching edge — skipping"
            )
            context["steps"][step.name] = {"status": "no_branch_matched"}
            # Mark all branch targets as done so the outer layer doesn't execute them
            for child_id, _ in children:
                context["_done"].add(child_id)
            return

        # Mark unchosen branches as consumed so the outer layer's next-layer
        # collection does not execute them.
        for child_id, _ in children:
            if child_id != chosen_id:
                context["_done"].add(child_id)

        context["steps"][step.name] = {"status": "branched", "chosen_step_id": chosen_id}
        await self._execute_dag_layer(
            [chosen_id], workflow, chain, context,
            step_index_counter, tenant_id, persona_override,
        )

    async def _execute_loop_step(
        self,
        step,
        workflow: WorkflowDefinition,
        chain: ChainPlan,
        context: dict,
        step_index_counter: list[int],
        tenant_id: str,
        persona_override: Optional[str],
    ) -> None:
        """Iterate over a collection, executing the loop step's own action each time.

        Graph contract (enforced by WorkflowValidator):
          - The loop step has a self-referential LOOP_BACK edge (source == target ==
            this step).  This is a structural marker only — the engine does NOT
            re-enter the step via DAG traversal; it re-executes the step's action
            directly in this method.
          - The loop step has exactly one DEFAULT outgoing edge to the exit step.
            That edge is followed automatically by ``_execute_dag_layer`` after
            this method returns.

        The iterator is resolved from ``step.config["iterator"]`` — a ``{{...}}``
        template or literal list.  ``context["loop_current"]`` and
        ``context["loop_index"]`` are set on each iteration so downstream param
        templates can reference them.

        If the step has no ``tool_name`` (pure control step), iteration still
        runs but no tool is executed.
        """
        # Resolve iterator
        iterator_expr = step.config.get("iterator", "[]")
        items = self._resolve_value(iterator_expr, context)
        if not isinstance(items, (list, tuple)):
            items = [items] if items is not None else []

        results: list[Any] = []
        saved_loop_current = context.get("loop_current")
        saved_loop_index = context.get("loop_index", 0)

        for i, item in enumerate(items):
            context["loop_current"] = item
            context["loop_index"] = i

            if step.tool_name:
                # Self-loop model: execute this step's own tool for each iteration.
                # Each iteration gets its own seal (step_index increments).
                iter_index = step_index_counter[0]
                step_index_counter[0] += 1
                try:
                    result = await self._execute_action_step(
                        step, workflow, chain, context,
                        iter_index, tenant_id, persona_override,
                    )
                    results.append(result)
                except AnomalyDetected:
                    results.append(None)
                    raise  # bubble up — loop is aborted on gate failure

        # Restore outer loop context (supports nested loops)
        context["loop_current"] = saved_loop_current
        context["loop_index"] = saved_loop_index

        context["steps"][step.name] = {
            "status": "completed",
            "iterations": len(items),
            "results": results,
        }

    async def _execute_parallel_step(
        self,
        step,
        workflow: WorkflowDefinition,
        chain: ChainPlan,
        context: dict,
        step_index_counter: list[int],
        tenant_id: str,
        persona_override: Optional[str],
        step_index: int = 0,
    ) -> None:
        """Execute a PARALLEL-typed step.

        Two execution models are supported:

        **Edge-based (graph-native, preferred)**
          When ``step.config`` has no ``"branches"`` key the step is treated as a
          *branch participant* — an action step whose siblings (same parent) are
          the other branches.  The step simply executes its own tool via
          ``_execute_action_step``.  Sibling steps are dispatched concurrently by
          ``_execute_dag_layer`` (multiple step_ids in one layer call).

        **Config-based (fan-out controller)**
          When ``step.config["branches"]`` lists step IDs, this step acts as a
          fan-out controller — it dispatches all listed branches concurrently via
          ``asyncio.gather`` and collects their results.
          ``step.config.get("fail_fast", True)`` controls error behaviour.
        """
        branch_ids: list[str] = list(step.config.get("branches", []))

        if not branch_ids:
            # Edge-based model: PARALLEL step is itself an action node.
            if step.tool_name:
                await self._execute_action_step(
                    step, workflow, chain, context,
                    step_index, tenant_id, persona_override,
                )
            else:
                context["steps"][step.name] = {"status": "no_action"}
            return

        # Config-based fan-out model
        fail_fast: bool = step.config.get("fail_fast", True)

        async def _run_branch(bid: str) -> None:
            await self._execute_dag_layer(
                [bid], workflow, chain, context,
                step_index_counter, tenant_id, persona_override,
            )

        if fail_fast:
            await asyncio.gather(*[_run_branch(bid) for bid in branch_ids])
        else:
            gather_results = await asyncio.gather(
                *[_run_branch(bid) for bid in branch_ids],
                return_exceptions=True,
            )
            errors = [r for r in gather_results if isinstance(r, Exception)]
            if errors:
                logger.warning(
                    f"[Engine] Parallel step '{step.name}' had {len(errors)} "
                    f"branch failure(s): {errors}"
                )

        branch_results = {}
        for bid in branch_ids:
            branch_step = next((s for s in workflow.steps if s.id == bid), None)
            if branch_step and branch_step.name in context["steps"]:
                branch_results[branch_step.name] = context["steps"][branch_step.name]

        context["steps"][step.name] = {
            "status": "completed",
            "branches": branch_results,
        }

    async def _execute_sub_workflow_step(
        self,
        step,
        context: dict,
        tenant_id: str,
        persona_override: Optional[str],
    ) -> None:
        """Execute a nested workflow and store its execution in context.

        The sub-workflow ID is taken from ``step.config["sub_workflow_id"]``.
        Trigger data is passed through from the parent context's trigger dict.
        """
        sub_workflow_id: Optional[str] = step.config.get("sub_workflow_id")
        if not sub_workflow_id:
            logger.warning(
                f"[Engine] Sub-workflow step '{step.name}' missing sub_workflow_id"
            )
            context["steps"][step.name] = {
                "status": "failed",
                "error": "missing sub_workflow_id in step config",
            }
            return

        sub_trigger = dict(context.get("trigger", {}))

        try:
            sub_execution = await self.run_workflow(
                workflow_id=sub_workflow_id,
                tenant_id=tenant_id,
                trigger_data=sub_trigger,
                persona_override=persona_override,
            )
            context["steps"][step.name] = {
                "status": sub_execution.status.value,
                "execution_id": sub_execution.id,
                "step_results": sub_execution.step_results,
            }
        except Exception as exc:
            context["steps"][step.name] = {
                "status": "failed",
                "error": str(exc),
            }
            raise

    async def _execute_approval_step(
        self,
        step,
        chain: ChainPlan,
        context: dict,
        step_index: int,
        tenant_id: str,
    ) -> None:
        """Block until a human approval is received or the step times out.

        Creates an approval request record and polls for resolution.  On timeout
        the step raises EscalationRequired.  In v1 the approval record is stored
        in ``context["_approvals"]`` so API routes / webhooks can resolve it.

        ``step.config`` keys:
          - ``instructions``: human-readable instructions for the approver
          - ``approvers``:    list of user IDs who may approve
          - ``timeout_seconds``: seconds before the step escalates (default 3600)
        """
        instructions: str = step.config.get("instructions", "Manual approval required.")
        approvers: list[str] = step.config.get("approvers", [])
        timeout_seconds: float = float(
            step.config.get("timeout_seconds", step.timeout_seconds or 3600)
        )

        approval_id = f"approval:{chain.id}:{step_index}"
        approval_record: dict[str, Any] = {
            "id": approval_id,
            "step_id": step.id,
            "step_name": step.name,
            "chain_id": chain.id,
            "tenant_id": tenant_id,
            "instructions": instructions,
            "approvers": approvers,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Register approval in context so API routes can resolve it
        approvals: dict = context.setdefault("_approvals", {})
        approvals[approval_id] = approval_record

        logger.info(
            f"[Engine] Approval step '{step.name}' waiting "
            f"(timeout={timeout_seconds}s, approvers={approvers})"
        )

        # Poll for resolution — resolution happens by an external caller updating
        # context["_approvals"][approval_id]["status"] to "approved" or "denied".
        poll_interval = 1.0
        elapsed = 0.0
        while elapsed < timeout_seconds:
            status = approval_record.get("status", "pending")
            if status == "approved":
                context["steps"][step.name] = {
                    "status": "approved",
                    "approval_id": approval_id,
                }
                return
            if status == "denied":
                context["steps"][step.name] = {
                    "status": "denied",
                    "approval_id": approval_id,
                }
                raise EscalationRequired(
                    f"Approval step '{step.name}' was denied.",
                    context={"approval_id": approval_id, "chain_id": chain.id},
                )
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout — escalate
        context["steps"][step.name] = {
            "status": "timeout",
            "approval_id": approval_id,
        }
        raise EscalationRequired(
            f"Approval step '{step.name}' timed out after {timeout_seconds}s.",
            context={"approval_id": approval_id, "chain_id": chain.id},
        )

    # ── DAG traversal helpers ──────────────────────────────────────────────────

    def _collect_reachable_step_ids(
        self, start_id: str, workflow: WorkflowDefinition
    ) -> set[str]:
        """BFS from start_id, returning all reachable step IDs (excluding LOOP_BACK).

        Used by _execute_loop_step to identify which step IDs must be removed from
        context["_done"] before each loop iteration so they can re-execute.
        """
        from nexus.workflows.dag import get_children

        valid_ids = {s.id for s in workflow.steps}
        visited: set[str] = set()
        queue = [start_id]
        while queue:
            sid = queue.pop(0)
            if sid in visited or sid not in valid_ids:
                continue
            visited.add(sid)
            for child_id, edge in get_children(sid, workflow.edges):
                if edge.edge_type != EdgeType.LOOP_BACK:
                    queue.append(child_id)
        return visited

    # ── Template resolution helpers ────────────────────────────────────────────

    def _resolve_params(self, params: dict, context: dict) -> dict:
        """Recursively resolve ``{{...}}`` templates in a params dict.

        Handles nested dicts and lists.  Scalar non-string values are returned
        unchanged.
        """
        return {k: self._resolve_value(v, context) for k, v in params.items()}

    def _resolve_value(self, value: Any, context: dict) -> Any:
        """Resolve a single value that may contain ``{{...}}`` templates.

        If the *entire* string is a single ``{{path}}`` template the resolved
        Python value is returned (preserving type).  Otherwise, all templates
        in the string are substituted as their string representations.
        """
        if isinstance(value, str):
            # Full-value template: return actual Python value
            full = re.fullmatch(r"\{\{([^}]+)\}\}", value.strip())
            if full:
                return self._lookup_path(full.group(1).strip(), context)
            # Partial templates: string interpolation
            return self._TEMPLATE_RE.sub(
                lambda m: str(self._lookup_path(m.group(1).strip(), context) or ""),
                value,
            )
        if isinstance(value, dict):
            return self._resolve_params(value, context)
        if isinstance(value, list):
            return [self._resolve_value(v, context) for v in value]
        return value

    def _lookup_path(self, path: str, context: dict) -> Any:
        """Navigate a dot-separated path through the context dict.

        Supports both dict key access and attribute access for mixed objects.
        Returns None for any missing segment.

        Examples::

            _lookup_path("trigger.user_id", context)
            _lookup_path("steps.fetch_data.result", context)
            _lookup_path("loop_current", context)
        """
        parts = path.split(".")
        val: Any = context
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = getattr(val, part, None)
            if val is None:
                return None
        return val
