"""Core NEXUS engine. THE most important file. Wires EVERYTHING together.

Orchestrates: decompose → retrieve → plan → [gate → execute → validate] per step.

See NEXUS_BUILD_SPEC.md Section 8 (Phase 4) and Section 19 (Integration Wiring)
for the complete execution flow.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional

from nexus.types import (
    ChainPlan, Seal, IntentDeclaration, PersonaContract,
    ActionStatus, ChainStatus, GateVerdict, ReasoningDecision,
)
from nexus.exceptions import (
    NexusError, AnomalyDetected, ChainAborted, EscalationRequired,
)
from nexus.config import NexusConfig
from nexus.core.personas import PersonaManager
from nexus.core.anomaly import AnomalyEngine
from nexus.core.notary import Notary
from nexus.core.ledger import Ledger
from nexus.core.chain import ChainManager
from nexus.core.verifier import IntentVerifier
from nexus.core.output_validator import OutputValidator
from nexus.core.cot_logger import CoTLogger
from nexus.knowledge.context import ContextBuilder
from nexus.tools.registry import ToolRegistry
from nexus.tools.selector import ToolSelector
from nexus.tools.executor import ToolExecutor
from nexus.reasoning.think_act import ThinkActGate
from nexus.reasoning.continue_complete import ContinueCompleteGate
from nexus.reasoning.escalate import EscalateGate

logger = logging.getLogger(__name__)


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

    async def run(self, task: str, tenant_id: str, persona_name: str = None) -> ChainPlan:
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
                activation_time = self.persona_manager.get_activation_time(step_persona_name)

                self.cot_logger.log(cot_key, f"Persona '{step_persona_name}' activated")

                # ── e. ANOMALY CHECK (4 gates) ────────────────────────────────
                anomaly_result = await self.anomaly_engine.check(
                    persona=activated_persona,
                    intent=intent,
                    activation_time=activation_time,
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
                seal = self.notary.create_seal(
                    chain_id=chain.id,
                    step_index=step_index,
                    tenant_id=tenant_id,
                    persona_id=step_persona_name,
                    intent=intent,
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
                    )

                # ── h+i. VERIFY + EXECUTE (executor handles both) ─────────────
                self.cot_logger.log(cot_key, f"Executing tool '{intent.tool_name}'")
                result, error_str = await self.tool_executor.execute(intent)

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
                "completed_at": datetime.utcnow(),
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

            prompt = DECOMPOSE_TASK.format(
                tool_list=tool_list,
                persona_list=persona_list,
                task=task,
            )

            try:
                response = await self.llm_client.complete(
                    messages=[{"role": "user", "content": prompt}]
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
        logger.error(f"[Engine] No personas loaded — using synthetic fallback")
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
        if not self.callbacks:
            return
        for cb in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event, data)
                else:
                    cb(event, data)
            except Exception as cb_exc:
                logger.warning(f"[Engine] Callback error on '{event}': {cb_exc}")
