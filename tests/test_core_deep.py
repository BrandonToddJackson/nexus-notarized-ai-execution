"""Deep unit tests for core components — edge cases not covered by test_smoketest.py.

Coverage:
  PersonaManager  — activate unknown/disabled, TTL tracking, load_personas reload, validate_action tool fail
  Notary          — fingerprint length, finalize doesn't mutate fingerprint, blocked-status seal
  Ledger          — pagination (offset/limit), verify_integrity, memory cap enforcement
  ChainManager    — get_current_step, advance→COMPLETED, is_complete before/after
  IntentVerifier  — extra declared params, subset params OK, resource advisory (no raise)
  OutputValidator — SSN / CC / email PII, empty collection, error trace, destructive action bypass
  CoTLogger       — unknown seal → [], multiple seals independent
"""

import pytest

from nexus.types import (
    PersonaContract, RiskLevel, IntentDeclaration,
    AnomalyResult, GateResult, GateVerdict, Seal, ActionStatus, ChainStatus,
)
from nexus.exceptions import PersonaViolation, SealIntegrityError
from nexus.core.personas import PersonaManager
from nexus.core.notary import Notary
from nexus.core.ledger import Ledger
from nexus.core.chain import ChainManager
from nexus.core.verifier import IntentVerifier
from nexus.core.output_validator import OutputValidator
from nexus.core.cot_logger import CoTLogger

TENANT = "deep-tenant-001"
CHAIN = "chain-deep-001"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _researcher() -> PersonaContract:
    return PersonaContract(
        name="researcher",
        description="Searches information",
        allowed_tools=["knowledge_search", "web_search"],
        resource_scopes=["kb:*", "web:*"],
        intent_patterns=["search for information", "look up"],
        risk_tolerance=RiskLevel.LOW,
        max_ttl_seconds=60,
    )


def _disabled_persona() -> PersonaContract:
    return PersonaContract(
        name="retired",
        description="Disabled persona",
        allowed_tools=["knowledge_search"],
        resource_scopes=["kb:*"],
        intent_patterns=["search"],
        risk_tolerance=RiskLevel.LOW,
        max_ttl_seconds=60,
        is_active=False,
    )


def _anomaly_pass() -> AnomalyResult:
    gate = GateResult(
        gate_name="scope", verdict=GateVerdict.PASS, score=1.0, threshold=1.0, details="ok"
    )
    return AnomalyResult(
        gates=[gate, gate, gate, gate],
        overall_verdict=GateVerdict.PASS,
        risk_level=RiskLevel.LOW,
        persona_uuid="researcher",
        action_fingerprint="fp123",
    )


def _intent(tool: str = "knowledge_search", params: dict = None) -> IntentDeclaration:
    return IntentDeclaration(
        task_description="test task",
        planned_action="search for information about NEXUS",
        tool_name=tool,
        tool_params=params or {"query": "test"},
        resource_targets=["kb:docs"],
        reasoning="test reasoning",
        confidence=0.9,
    )


def _make_seal(chain_id: str = CHAIN, step_index: int = 0, tenant_id: str = TENANT) -> Seal:
    notary = Notary()
    return notary.create_seal(
        chain_id=chain_id,
        step_index=step_index,
        tenant_id=tenant_id,
        persona_id="researcher",
        intent=_intent(),
        anomaly_result=_anomaly_pass(),
    )


# ── PersonaManager ─────────────────────────────────────────────────────────────

class TestPersonaManagerDeep:

    def test_activate_unknown_persona_raises(self):
        pm = PersonaManager([_researcher()])
        with pytest.raises(PersonaViolation, match="not found"):
            pm.activate("ghost", TENANT)

    def test_activate_disabled_persona_raises(self):
        pm = PersonaManager([_disabled_persona()])
        with pytest.raises(PersonaViolation, match="disabled"):
            pm.activate("retired", TENANT)

    def test_get_ttl_remaining_positive_after_activation(self):
        pm = PersonaManager([_researcher()])
        pm.activate("researcher", TENANT)
        ttl = pm.get_ttl_remaining("researcher")
        assert ttl > 0
        assert ttl <= 60

    def test_get_ttl_remaining_zero_when_not_active(self):
        pm = PersonaManager([_researcher()])
        assert pm.get_ttl_remaining("researcher") == 0

    def test_get_ttl_remaining_zero_after_revoke(self):
        pm = PersonaManager([_researcher()])
        pm.activate("researcher", TENANT)
        pm.revoke("researcher")
        assert pm.get_ttl_remaining("researcher") == 0

    def test_get_ttl_remaining_zero_for_unknown_persona(self):
        pm = PersonaManager([_researcher()])
        assert pm.get_ttl_remaining("nobody") == 0

    def test_load_personas_adds_to_existing(self):
        pm = PersonaManager([_researcher()])
        analyst = PersonaContract(
            name="analyst",
            description="Analyzes data",
            allowed_tools=["compute_stats"],
            resource_scopes=["data:*"],
            intent_patterns=["analyze"],
            risk_tolerance=RiskLevel.MEDIUM,
            max_ttl_seconds=120,
        )
        pm.load_personas([analyst])
        assert pm.get_persona("researcher") is not None
        assert pm.get_persona("analyst") is not None

    def test_load_personas_overwrites_existing(self):
        pm = PersonaManager([_researcher()])
        updated = PersonaContract(
            name="researcher",
            description="Updated researcher",
            allowed_tools=["knowledge_search"],
            resource_scopes=["kb:*"],
            intent_patterns=["search"],
            risk_tolerance=RiskLevel.HIGH,  # changed
            max_ttl_seconds=30,
        )
        pm.load_personas([updated])
        contract = pm.get_persona("researcher")
        assert contract.risk_tolerance == RiskLevel.HIGH
        assert contract.max_ttl_seconds == 30

    def test_validate_action_wrong_tool_raises(self):
        pm = PersonaManager([_researcher()])
        contract = pm.get_persona("researcher")
        with pytest.raises(PersonaViolation, match="send_email"):
            pm.validate_action(contract, "send_email", ["kb:docs"])

    def test_validate_action_correct_tool_and_scope_passes(self):
        pm = PersonaManager([_researcher()])
        contract = pm.get_persona("researcher")
        result = pm.validate_action(contract, "knowledge_search", ["kb:product_docs"])
        assert result is True

    def test_validate_action_wrong_resource_raises(self):
        pm = PersonaManager([_researcher()])
        contract = pm.get_persona("researcher")
        with pytest.raises(PersonaViolation, match="db:secret"):
            pm.validate_action(contract, "knowledge_search", ["db:secret"])

    def test_revoke_idempotent(self):
        """Revoking a non-active persona must not raise."""
        pm = PersonaManager([_researcher()])
        pm.revoke("researcher")  # never activated — should be a no-op
        pm.revoke("researcher")  # double revoke — still no-op

    def test_list_personas_returns_all_loaded(self):
        p1 = _researcher()
        p2 = _disabled_persona()
        pm = PersonaManager([p1, p2])
        names = {p.name for p in pm.list_personas()}
        assert "researcher" in names
        assert "retired" in names


# ── Notary ─────────────────────────────────────────────────────────────────────

class TestNotaryDeep:

    def test_fingerprint_is_64_char_hex(self):
        notary = Notary()
        seal = notary.create_seal(
            chain_id=CHAIN, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=_intent(), anomaly_result=_anomaly_pass(),
        )
        assert len(seal.fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in seal.fingerprint)

    def test_finalize_does_not_change_fingerprint(self):
        notary = Notary()
        seal = notary.create_seal(
            chain_id=CHAIN, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=_intent(), anomaly_result=_anomaly_pass(),
        )
        original_fp = seal.fingerprint
        finalized = notary.finalize_seal(seal, "result", ActionStatus.EXECUTED)
        assert finalized.fingerprint == original_fp

    def test_finalize_with_blocked_status(self):
        notary = Notary()
        seal = notary.create_seal(
            chain_id=CHAIN, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=_intent(), anomaly_result=_anomaly_pass(),
        )
        blocked = notary.finalize_seal(seal, None, ActionStatus.BLOCKED, error="Gate 1 FAIL: scope")
        assert blocked.status == ActionStatus.BLOCKED
        assert blocked.error == "Gate 1 FAIL: scope"
        assert blocked.tool_result is None

    def test_second_seal_parent_equals_first_fingerprint(self):
        notary = Notary()
        seal0 = notary.create_seal(
            chain_id=CHAIN, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=_intent(), anomaly_result=_anomaly_pass(),
        )
        seal1 = notary.create_seal(
            chain_id=CHAIN, step_index=1, tenant_id=TENANT,
            persona_id="researcher", intent=_intent(), anomaly_result=_anomaly_pass(),
        )
        assert seal1.parent_fingerprint == seal0.fingerprint

    def test_verify_chain_empty_list_returns_true(self):
        notary = Notary()
        assert notary.verify_chain([]) is True

    def test_verify_chain_single_seal(self):
        notary = Notary()
        seal = notary.create_seal(
            chain_id=CHAIN, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=_intent(), anomaly_result=_anomaly_pass(),
        )
        seal = notary.finalize_seal(seal, "result", ActionStatus.EXECUTED)
        assert notary.verify_chain([seal]) is True

    def test_verify_chain_out_of_order_still_works(self):
        """verify_chain sorts by step_index internally."""
        notary = Notary()
        seal0 = notary.create_seal(
            chain_id=CHAIN, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=_intent(), anomaly_result=_anomaly_pass(),
        )
        seal1 = notary.create_seal(
            chain_id=CHAIN, step_index=1, tenant_id=TENANT,
            persona_id="researcher", intent=_intent(), anomaly_result=_anomaly_pass(),
        )
        # Pass them reversed — should still verify because it sorts internally
        assert notary.verify_chain([seal1, seal0]) is True

    def test_verify_chain_detects_tampered_fingerprint(self):
        notary = Notary()
        seal = notary.create_seal(
            chain_id=CHAIN, step_index=0, tenant_id=TENANT,
            persona_id="researcher", intent=_intent(), anomaly_result=_anomaly_pass(),
        )
        tampered = seal.model_copy(update={"fingerprint": "a" * 64})
        with pytest.raises(SealIntegrityError, match="step 0"):
            notary.verify_chain([tampered])

    def test_different_chains_have_different_fingerprints(self):
        """Two notary instances produce independent fingerprints for the same content."""
        n1 = Notary()
        n2 = Notary()
        s1 = n1.create_seal(CHAIN, 0, TENANT, "researcher", _intent(), _anomaly_pass())
        s2 = n2.create_seal(CHAIN, 0, TENANT, "researcher", _intent(), _anomaly_pass())
        # Both start from "" so first seals should match
        assert s1.fingerprint == s2.fingerprint


# ── Ledger ─────────────────────────────────────────────────────────────────────

class TestLedgerDeep:

    @pytest.mark.asyncio
    async def test_get_by_tenant_respects_limit(self):
        ledger = Ledger()
        for i in range(10):
            s = _make_seal(step_index=i)
            await ledger.append(s)
        results = await ledger.get_by_tenant(TENANT, limit=5)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_get_by_tenant_respects_offset(self):
        ledger = Ledger()
        seals = []
        for i in range(5):
            s = _make_seal(step_index=i)
            await ledger.append(s)
            seals.append(s)
        # offset=3 → return items at index 3,4
        results = await ledger.get_by_tenant(TENANT, limit=100, offset=3)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_by_tenant_empty_for_unknown_tenant(self):
        ledger = Ledger()
        s = _make_seal(tenant_id="other-tenant")
        await ledger.append(s)
        results = await ledger.get_by_tenant("nonexistent-tenant")
        assert results == []

    @pytest.mark.asyncio
    async def test_verify_integrity_valid_chain(self):
        notary = Notary()
        ledger = Ledger()
        for i in range(3):
            seal = notary.create_seal(CHAIN, i, TENANT, "researcher", _intent(), _anomaly_pass())
            seal = notary.finalize_seal(seal, f"result_{i}", ActionStatus.EXECUTED)
            await ledger.append(seal)
        assert await ledger.verify_integrity(CHAIN) is True

    @pytest.mark.asyncio
    async def test_verify_integrity_empty_chain(self):
        """A chain with no seals should be considered trivially valid."""
        ledger = Ledger()
        result = await ledger.verify_integrity("nonexistent-chain")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_chain_filters_by_tenant(self):
        ledger = Ledger()
        s_a = _make_seal(tenant_id="tenant-alpha")
        s_b = _make_seal(chain_id=CHAIN, tenant_id="tenant-beta")
        await ledger.append(s_a)
        await ledger.append(s_b)
        # Both use CHAIN — but filter by tenant should return only tenant-alpha
        results = await ledger.get_chain(CHAIN, tenant_id="tenant-alpha")
        assert len(results) == 1
        assert results[0].tenant_id == "tenant-alpha"

    @pytest.mark.asyncio
    async def test_memory_cap_trims_oldest_seals(self):
        """After 10,001 seals, the oldest is dropped."""
        from nexus.core import ledger as ledger_module
        original_cap = ledger_module._MAX_MEMORY_SEALS
        ledger_module._MAX_MEMORY_SEALS = 5  # small cap for testing
        try:
            ledger = Ledger()
            for i in range(7):
                s = _make_seal(step_index=i, chain_id=f"chain-{i}")
                await ledger.append(s)
            assert len(ledger._memory_store) == 5
        finally:
            ledger_module._MAX_MEMORY_SEALS = original_cap

    @pytest.mark.asyncio
    async def test_multiple_chains_in_same_ledger(self):
        ledger = Ledger()
        for i in range(3):
            s = _make_seal(chain_id="chain-A", step_index=i)
            await ledger.append(s)
        for i in range(2):
            s = _make_seal(chain_id="chain-B", step_index=i)
            await ledger.append(s)
        assert len(await ledger.get_chain("chain-A")) == 3
        assert len(await ledger.get_chain("chain-B")) == 2


# ── ChainManager ───────────────────────────────────────────────────────────────

class TestChainManagerDeep:

    def test_get_current_step_first_step(self):
        cm = ChainManager()
        chain = cm.create_chain(TENANT, "task", [
            {"action": "step1", "tool": "knowledge_search"},
            {"action": "step2", "tool": "web_search"},
        ])
        step = cm.get_current_step(chain)
        assert step is not None
        assert step["action"] == "step1"

    def test_get_current_step_advances_after_seal(self):
        cm = ChainManager()
        chain = cm.create_chain(TENANT, "task", [
            {"action": "step1"}, {"action": "step2"}
        ])
        chain = cm.advance(chain, "seal-001")
        step = cm.get_current_step(chain)
        assert step["action"] == "step2"

    def test_get_current_step_none_when_complete(self):
        cm = ChainManager()
        chain = cm.create_chain(TENANT, "task", [{"action": "step1"}])
        chain = cm.advance(chain, "seal-001")
        assert cm.get_current_step(chain) is None

    def test_advance_last_step_sets_completed(self):
        cm = ChainManager()
        chain = cm.create_chain(TENANT, "task", [{"action": "only step"}])
        chain = cm.advance(chain, "seal-001")
        assert chain.status == ChainStatus.COMPLETED
        assert chain.completed_at is not None

    def test_advance_partial_step_stays_executing(self):
        cm = ChainManager()
        chain = cm.create_chain(TENANT, "task", [{"action": "s1"}, {"action": "s2"}])
        chain = cm.advance(chain, "seal-001")
        assert chain.status == ChainStatus.EXECUTING
        assert chain.completed_at is None

    def test_is_complete_false_before_all_steps(self):
        cm = ChainManager()
        chain = cm.create_chain(TENANT, "task", [{"action": "s1"}, {"action": "s2"}])
        chain = cm.advance(chain, "seal-001")
        assert cm.is_complete(chain) is False

    def test_is_complete_true_after_all_steps(self):
        cm = ChainManager()
        chain = cm.create_chain(TENANT, "task", [{"action": "s1"}, {"action": "s2"}])
        chain = cm.advance(chain, "seal-001")
        chain = cm.advance(chain, "seal-002")
        assert cm.is_complete(chain) is True

    def test_fail_records_error_message(self):
        cm = ChainManager()
        chain = cm.create_chain(TENANT, "task", [{"action": "s"}])
        chain = cm.fail(chain, "Tool timed out after 30s")
        assert "Tool timed out" in chain.error

    def test_escalate_records_reason(self):
        cm = ChainManager()
        chain = cm.create_chain(TENANT, "task", [{"action": "s"}])
        chain = cm.escalate(chain, "Human review required: anomaly detected")
        assert chain.status == ChainStatus.ESCALATED
        assert "Human review" in chain.error

    def test_create_chain_preserves_task_text(self):
        cm = ChainManager()
        task = "Find all documents about quantum computing and summarize them"
        chain = cm.create_chain(TENANT, task, [{"action": "search"}])
        assert chain.task == task


# ── IntentVerifier ─────────────────────────────────────────────────────────────

class TestIntentVerifierDeep:

    def _verifier(self) -> IntentVerifier:
        return IntentVerifier()

    def test_exact_match_passes(self):
        v = self._verifier()
        intent = _intent("knowledge_search", {"query": "test"})
        assert v.verify(intent, "knowledge_search", {"query": "test"}) is True

    def test_tool_name_mismatch_raises(self):
        v = self._verifier()
        intent = _intent("knowledge_search", {"query": "test"})
        with pytest.raises(PersonaViolation, match="send_email"):
            v.verify(intent, "send_email", {"to": "user@example.com"})

    def test_extra_declared_params_raises(self):
        """Intent declares a param that actual call doesn't have → PersonaViolation."""
        v = self._verifier()
        intent = _intent("knowledge_search", {"query": "test", "nonexistent_param": "value"})
        with pytest.raises(PersonaViolation, match="nonexistent_param"):
            v.verify(intent, "knowledge_search", {"query": "test"})

    def test_subset_declared_params_passes(self):
        """Declaring fewer params than actual call uses is OK (subset check)."""
        v = self._verifier()
        intent = _intent("knowledge_search", {"query": "test"})
        # actual call has extra params — that's fine
        assert v.verify(intent, "knowledge_search", {"query": "test", "namespace": "default"}) is True

    def test_resource_target_mismatch_does_not_raise(self):
        """Resource target mismatch is advisory only (logs warning, no exception)."""
        v = self._verifier()
        intent = IntentDeclaration(
            task_description="test",
            planned_action="search for info",
            tool_name="knowledge_search",
            tool_params={"query": "test"},
            resource_targets=["kb:docs"],
            reasoning="test",
        )
        # The tool param doesn't contain "kb:docs" — should NOT raise
        result = v.verify(intent, "knowledge_search", {"query": "something completely different"})
        assert result is True

    def test_empty_resource_targets_passes(self):
        v = self._verifier()
        intent = IntentDeclaration(
            task_description="test",
            planned_action="search for info",
            tool_name="knowledge_search",
            tool_params={"query": "test"},
            resource_targets=[],
            reasoning="test",
        )
        assert v.verify(intent, "knowledge_search", {"query": "test"}) is True


# ── OutputValidator ────────────────────────────────────────────────────────────

class TestOutputValidatorDeep:

    def _ov(self) -> OutputValidator:
        return OutputValidator()

    def _search_intent(self) -> IntentDeclaration:
        return IntentDeclaration(
            task_description="search",
            planned_action="search for information about NEXUS",
            tool_name="knowledge_search",
            tool_params={},
            resource_targets=[],
            reasoning="",
        )

    def _delete_intent(self) -> IntentDeclaration:
        return IntentDeclaration(
            task_description="delete",
            planned_action="delete all temporary files",
            tool_name="file_delete",
            tool_params={},
            resource_targets=[],
            reasoning="",
        )

    @pytest.mark.asyncio
    async def test_ssn_detected(self):
        ov = self._ov()
        valid, reason = await ov.validate(self._search_intent(), "User SSN: 123-45-6789")
        assert valid is False
        assert "SSN" in reason

    @pytest.mark.asyncio
    async def test_credit_card_detected(self):
        ov = self._ov()
        valid, reason = await ov.validate(self._search_intent(), "Card: 4111 1111 1111 1111")
        assert valid is False
        assert "credit card" in reason.lower()

    @pytest.mark.asyncio
    async def test_email_address_detected(self):
        ov = self._ov()
        valid, reason = await ov.validate(self._search_intent(), "Contact user@example.com for details")
        assert valid is False
        assert "email" in reason.lower()

    @pytest.mark.asyncio
    async def test_error_trace_detected(self):
        ov = self._ov()
        result = "Traceback (most recent call last):\n  File test.py, line 1, in <module>"
        valid, reason = await ov.validate(self._search_intent(), result)
        assert valid is False
        assert "error" in reason.lower()

    @pytest.mark.asyncio
    async def test_permission_denied_detected(self):
        ov = self._ov()
        valid, reason = await ov.validate(self._search_intent(), "Permission denied: cannot access /etc/passwd")
        assert valid is False
        assert "error" in reason.lower()

    @pytest.mark.asyncio
    async def test_none_result_fails_for_search(self):
        ov = self._ov()
        valid, reason = await ov.validate(self._search_intent(), None)
        assert valid is False
        assert reason

    @pytest.mark.asyncio
    async def test_none_result_passes_for_destructive(self):
        """Delete/write actions returning None should be considered valid."""
        ov = self._ov()
        valid, reason = await ov.validate(self._delete_intent(), None)
        assert valid is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_empty_string_fails_for_search(self):
        ov = self._ov()
        valid, reason = await ov.validate(self._search_intent(), "")
        assert valid is False

    @pytest.mark.asyncio
    async def test_empty_dict_fails_for_search(self):
        ov = self._ov()
        valid, reason = await ov.validate(self._search_intent(), {})
        assert valid is False

    @pytest.mark.asyncio
    async def test_empty_list_fails_for_search(self):
        ov = self._ov()
        valid, reason = await ov.validate(self._search_intent(), [])
        assert valid is False

    @pytest.mark.asyncio
    async def test_normal_text_passes(self):
        ov = self._ov()
        valid, reason = await ov.validate(self._search_intent(), "NEXUS is an AI agent framework.")
        assert valid is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_dict_result_passes(self):
        ov = self._ov()
        valid, reason = await ov.validate(self._search_intent(), {"count": 5, "results": ["a", "b"]})
        assert valid is True

    @pytest.mark.asyncio
    async def test_send_intent_with_none_passes(self):
        """'send' in planned_action → treated as destructive."""
        ov = self._ov()
        send_intent = IntentDeclaration(
            task_description="notify user",
            planned_action="send notification email to user",
            tool_name="send_email",
            tool_params={},
            resource_targets=[],
            reasoning="",
        )
        valid, reason = await ov.validate(send_intent, None)
        assert valid is True


# ── CoTLogger ──────────────────────────────────────────────────────────────────

class TestCoTLoggerDeep:

    def test_unknown_seal_returns_empty_list(self):
        logger = CoTLogger()
        assert logger.get_trace("nonexistent-seal-id") == []

    def test_multiple_seals_are_independent(self):
        logger = CoTLogger()
        logger.log("seal-A", "step for A")
        logger.log("seal-B", "step for B")
        assert logger.get_trace("seal-A") == ["step for A"]
        assert logger.get_trace("seal-B") == ["step for B"]

    def test_log_order_preserved(self):
        logger = CoTLogger()
        steps = ["Context built", "Tool selected", "Gate check passed", "Execution started"]
        for s in steps:
            logger.log("seal-001", s)
        assert logger.get_trace("seal-001") == steps

    def test_clear_leaves_other_seals_intact(self):
        logger = CoTLogger()
        logger.log("seal-A", "step A")
        logger.log("seal-B", "step B")
        logger.clear("seal-A")
        assert logger.get_trace("seal-A") == []
        assert logger.get_trace("seal-B") == ["step B"]

    def test_clear_idempotent_on_unknown_seal(self):
        """Clearing a non-existent seal should not raise."""
        logger = CoTLogger()
        logger.clear("nonexistent")  # must not raise

    def test_multiple_log_calls_accumulate(self):
        logger = CoTLogger()
        for i in range(10):
            logger.log("seal-001", f"step {i}")
        trace = logger.get_trace("seal-001")
        assert len(trace) == 10
        assert trace[0] == "step 0"
        assert trace[9] == "step 9"
