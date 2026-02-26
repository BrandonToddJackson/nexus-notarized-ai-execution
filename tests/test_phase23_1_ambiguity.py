"""
Phase 23.1: Ambiguity Resolution — Test Suite

Coverage:
- Import smoke tests
- SpecificityScore construction
- AmbiguityResolver._parse_specificity_score (direct + failure fallback)
- AmbiguityResolver._parse_questions (valid + failure fallback)
- AmbiguityResolver._parse_refined_description
- AmbiguityResolver._extract_json (4 strategies)
- AmbiguityResolver._validate_answers (all types, all failure modes)
- AmbiguityResolver._merge_answers
- AmbiguityResolver._build_auto_complete_session
- AmbiguityResolver._seal_plan (fingerprint tamper-evidence)
- AmbiguityResolver.plan_to_generator_context
- AmbiguityResolver.score() (mocked LLM)
- AmbiguityResolver.start_session() (low-score → questions; high-score → auto-complete)
- AmbiguityResolver.submit_answers() (lifecycle: active→complete, active→next-round, expired, non-active)
- AmbiguityResolver.get_session / cancel_session
- WorkflowPlan fingerprint properties
- ClarifyingQuestion / ClarifyingAnswer round-trip serialisation
- DB repository round-trip (using in-memory SQLite)
- expire_abandoned_sessions
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Import smoke tests
# ─────────────────────────────────────────────────────────────────────────────

def test_import_ambiguity_resolver():
    from nexus.workflows.ambiguity import AmbiguityResolver
    assert AmbiguityResolver is not None


def test_import_from_workflows_package():
    from nexus.workflows import AmbiguityResolver
    assert AmbiguityResolver is not None


def test_import_prompts():
    from nexus.llm.prompts import (
        AMBIGUITY_SCORE_PROMPT,
        AMBIGUITY_QUESTIONS_PROMPT,
        AMBIGUITY_SYNTHESISE_PROMPT,
    )
    assert "{available_tools}" in AMBIGUITY_SCORE_PROMPT
    assert "{available_personas}" in AMBIGUITY_SCORE_PROMPT
    assert "{clarification_dimensions}" in AMBIGUITY_SCORE_PROMPT
    assert "{available_tools}" in AMBIGUITY_QUESTIONS_PROMPT
    assert "{max_questions}" in AMBIGUITY_QUESTIONS_PROMPT
    assert "refined" in AMBIGUITY_SYNTHESISE_PROMPT.lower() or "description" in AMBIGUITY_SYNTHESISE_PROMPT.lower()


def test_import_exception():
    from nexus.exceptions import AmbiguityResolutionError, NexusError
    assert issubclass(AmbiguityResolutionError, NexusError)


def test_import_types():
    from nexus.types import (
        AmbiguitySession,
        AmbiguitySessionStatus,
        ClarifyingAnswer,
        ClarifyingQuestion,
        QuestionType,
        SpecificityScore,
        WorkflowPlan,
        WorkflowPlanParameter,
    )
    assert issubclass(AmbiguitySession, object)
    assert issubclass(ClarifyingAnswer, object)
    assert issubclass(ClarifyingQuestion, object)
    assert issubclass(SpecificityScore, object)
    assert issubclass(WorkflowPlan, object)
    assert issubclass(WorkflowPlanParameter, object)
    assert AmbiguitySessionStatus.active == "active"
    assert AmbiguitySessionStatus.complete == "complete"
    assert AmbiguitySessionStatus.abandoned == "abandoned"
    assert AmbiguitySessionStatus.generated == "generated"
    assert QuestionType.single_choice == "single_choice"
    assert QuestionType.multi_choice == "multi_choice"
    assert QuestionType.text == "text"
    assert QuestionType.boolean == "boolean"
    assert QuestionType.number == "number"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.ambiguity_auto_generate_threshold = 0.75
    cfg.max_clarification_rounds = 3
    cfg.max_questions_per_round = 5
    cfg.ambiguity_session_ttl_hours = 24
    cfg.ambiguity_generation_model = None
    cfg.workflow_generation_model = None
    cfg.default_llm_model = "anthropic/claude-test"
    return cfg


@pytest.fixture
def mock_tool_registry():
    registry = MagicMock()
    tool1 = MagicMock()
    tool1.name = "web_search"
    tool2 = MagicMock()
    tool2.name = "http_request"
    registry.list_tools.return_value = [tool1, tool2]
    return registry


@pytest.fixture
def mock_persona_manager():
    pm = MagicMock()
    p1 = MagicMock()
    p1.name = "researcher"
    p2 = MagicMock()
    p2.name = "executor"
    pm.list_personas.return_value = [p1, p2]
    return pm


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": ""})
    return llm


@pytest.fixture
def resolver(mock_llm, mock_tool_registry, mock_persona_manager, mock_config):
    from nexus.workflows.ambiguity import AmbiguityResolver
    return AmbiguityResolver(
        llm_client=mock_llm,
        tool_registry=mock_tool_registry,
        persona_manager=mock_persona_manager,
        config=mock_config,
    )


@pytest.fixture
def sample_question():
    from nexus.types import ClarifyingQuestion, QuestionType
    return ClarifyingQuestion(
        id=str(uuid.uuid4()),
        session_id="sess-1",
        dimension="trigger",
        question="What triggers this workflow?",
        question_type=QuestionType.single_choice,
        options=["webhook", "cron", "manual"],
        required=True,
        default=None,
        hint="Choose how the workflow starts.",
        maps_to_param="trigger_type",
    )


@pytest.fixture
def sample_questions():
    from nexus.types import ClarifyingQuestion, QuestionType
    q1 = ClarifyingQuestion(
        id="q1",
        session_id="sess-1",
        dimension="trigger",
        question="What triggers?",
        question_type=QuestionType.single_choice,
        options=["webhook", "cron", "manual"],
        required=True,
        maps_to_param="trigger_type",
    )
    q2 = ClarifyingQuestion(
        id="q2",
        session_id="sess-1",
        dimension="tools",
        question="Which tools?",
        question_type=QuestionType.multi_choice,
        options=["web_search", "http_request", "slack"],
        required=True,
        maps_to_param="preferred_tools",
    )
    q3 = ClarifyingQuestion(
        id="q3",
        session_id="sess-1",
        dimension="scope",
        question="Autonomous?",
        question_type=QuestionType.boolean,
        required=False,
        maps_to_param="authority_level",
    )
    q4 = ClarifyingQuestion(
        id="q4",
        session_id="sess-1",
        dimension="data_sources",
        question="Describe data source",
        question_type=QuestionType.text,
        required=False,
        maps_to_param="data_sources",
        max_chars=200,
    )
    q5 = ClarifyingQuestion(
        id="q5",
        session_id="sess-1",
        dimension="authority",
        question="Max refund amount?",
        question_type=QuestionType.number,
        required=False,
        maps_to_param="example_data",
        min_value=0,
        max_value=10000,
    )
    return [q1, q2, q3, q4, q5]


@pytest.fixture
def active_session(sample_questions):
    from nexus.types import AmbiguitySession, AmbiguitySessionStatus
    now = datetime.now(timezone.utc)
    return AmbiguitySession(
        id="sess-1",
        tenant_id="tenant-1",
        original_description="handle customer service",
        status=AmbiguitySessionStatus.active,
        questions=sample_questions,
        answers=[],
        current_round=1,
        max_rounds=3,
        specificity_history=[0.08],
        plan=None,
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(hours=24),
    )


# ─────────────────────────────────────────────────────────────────────────────
# _extract_json — 4 strategies
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_strategy1_direct(self, resolver):
        result = resolver._extract_json('{"score": 0.5}')
        assert result == {"score": 0.5}

    def test_strategy1_list(self, resolver):
        result = resolver._extract_json('[{"question": "x"}]')
        assert result == [{"question": "x"}]

    def test_strategy2_json_fence(self, resolver):
        text = '```json\n{"score": 0.8}\n```'
        result = resolver._extract_json(text)
        assert result == {"score": 0.8}

    def test_strategy3_plain_fence(self, resolver):
        text = '```\n{"score": 0.3}\n```'
        result = resolver._extract_json(text)
        assert result == {"score": 0.3}

    def test_strategy4_embedded_dict(self, resolver):
        text = 'Here is the result: {"score": 0.6} end'
        result = resolver._extract_json(text)
        assert result == {"score": 0.6}

    def test_strategy4_embedded_list(self, resolver):
        text = 'Here is the result: [{"q": "hello"}] end'
        result = resolver._extract_json(text)
        assert result == [{"q": "hello"}]

    def test_returns_none_on_failure(self, resolver):
        result = resolver._extract_json("no json here at all")
        assert result is None

    def test_returns_none_on_empty(self, resolver):
        result = resolver._extract_json("")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# _parse_specificity_score
# ─────────────────────────────────────────────────────────────────────────────

class TestParseSpecificityScore:
    def test_valid_response(self, resolver):
        data = {
            "score": 0.45,
            "dimensions_resolved": ["tools", "output_destinations"],
            "dimensions_missing": ["trigger", "scope"],
            "tool_coverage_ratio": 0.6,
            "has_trigger": False,
            "has_success_condition": False,
            "has_scope_boundary": False,
            "reasoning": "Partially specified.",
        }
        score = resolver._parse_specificity_score(json.dumps(data))
        assert score.score == 0.45
        assert score.dimensions_resolved == ["tools", "output_destinations"]
        assert score.dimensions_missing == ["trigger", "scope"]
        assert score.tool_coverage_ratio == 0.6
        assert score.has_trigger is False
        assert score.can_auto_generate is False  # 0.45 < 0.75

    def test_score_above_threshold_auto_generate(self, resolver):
        data = {
            "score": 0.80,
            "dimensions_resolved": ["trigger", "tools", "scope", "success_condition"],
            "dimensions_missing": [],
            "tool_coverage_ratio": 1.0,
            "has_trigger": True,
            "has_success_condition": True,
            "has_scope_boundary": True,
            "reasoning": "Fully specified.",
        }
        score = resolver._parse_specificity_score(json.dumps(data))
        assert score.can_auto_generate is True

    def test_fallback_on_parse_failure(self, resolver):
        score = resolver._parse_specificity_score("not json")
        assert score.score == 0.0
        assert score.can_auto_generate is False
        assert len(score.dimensions_missing) > 0

    def test_fallback_on_non_dict_json(self, resolver):
        score = resolver._parse_specificity_score("[1, 2, 3]")
        assert score.score == 0.0
        assert score.can_auto_generate is False


# ─────────────────────────────────────────────────────────────────────────────
# _parse_questions
# ─────────────────────────────────────────────────────────────────────────────

class TestParseQuestions:
    def test_valid_questions(self, resolver):
        data = [
            {
                "dimension": "trigger",
                "question": "What triggers this?",
                "question_type": "single_choice",
                "options": ["webhook", "cron"],
                "required": True,
                "default": None,
                "hint": "Choose wisely.",
                "maps_to_param": "trigger_type",
                "max_chars": None,
                "min_value": None,
                "max_value": None,
            }
        ]
        questions = resolver._parse_questions(json.dumps(data))
        assert len(questions) == 1
        assert questions[0].dimension == "trigger"
        assert questions[0].question_type.value == "single_choice"
        assert questions[0].options == ["webhook", "cron"]
        assert questions[0].maps_to_param == "trigger_type"
        assert len(questions[0].id) == 36  # UUID format

    def test_truncates_to_max_questions(self, resolver):
        data = [
            {
                "dimension": "trigger",
                "question": f"Q{i}",
                "question_type": "text",
                "options": [],
                "required": True,
                "default": None,
                "hint": None,
                "maps_to_param": "scope_boundary",
            }
            for i in range(10)
        ]
        questions = resolver._parse_questions(json.dumps(data))
        assert len(questions) <= resolver._max_questions

    def test_returns_empty_on_parse_failure(self, resolver):
        questions = resolver._parse_questions("not json")
        assert questions == []

    def test_returns_empty_on_dict_not_list(self, resolver):
        questions = resolver._parse_questions('{"key": "value"}')
        assert questions == []

    def test_skips_malformed_items(self, resolver):
        data = [
            {"dimension": "trigger", "question": "Q1", "question_type": "INVALID_TYPE",
             "maps_to_param": "trigger_type"},
            {"dimension": "scope", "question": "Q2", "question_type": "text",
             "maps_to_param": "scope_boundary", "options": []},
        ]
        questions = resolver._parse_questions(json.dumps(data))
        # First item has invalid question_type → skipped. Second is valid.
        assert len(questions) == 1
        assert questions[0].dimension == "scope"


# ─────────────────────────────────────────────────────────────────────────────
# _parse_refined_description
# ─────────────────────────────────────────────────────────────────────────────

class TestParseRefinedDescription:
    def test_plain_text(self, resolver):
        result = resolver._parse_refined_description("When X happens, do Y.")
        assert result == "When X happens, do Y."

    def test_strips_code_fence(self, resolver):
        result = resolver._parse_refined_description("```\nWhen X happens, do Y.\n```")
        assert result == "When X happens, do Y."

    def test_strips_json_fence(self, resolver):
        result = resolver._parse_refined_description("```json\nWhen X happens\n```")
        assert result == "When X happens"

    def test_empty_returns_fallback(self, resolver):
        result = resolver._parse_refined_description("")
        assert result == "(No refined description produced.)"

    def test_whitespace_only_returns_fallback(self, resolver):
        result = resolver._parse_refined_description("   \n\t  ")
        assert result == "(No refined description produced.)"


# ─────────────────────────────────────────────────────────────────────────────
# _validate_answers
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateAnswers:
    def test_valid_single_choice(self, resolver, sample_questions):
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="q1", session_id="s", value="webhook", answered_at=datetime.now(timezone.utc))]
        resolver._validate_answers(answers, sample_questions)  # Should not raise

    def test_invalid_single_choice_not_in_options(self, resolver, sample_questions):
        from nexus.exceptions import AmbiguityResolutionError
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="q1", session_id="s", value="ftp", answered_at=datetime.now(timezone.utc))]
        with pytest.raises(AmbiguityResolutionError, match="single_choice"):
            resolver._validate_answers(answers, sample_questions)

    def test_valid_multi_choice(self, resolver, sample_questions):
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="q2", session_id="s", value=["web_search", "slack"], answered_at=datetime.now(timezone.utc))]
        resolver._validate_answers(answers, sample_questions)

    def test_invalid_multi_choice_not_list(self, resolver, sample_questions):
        from nexus.exceptions import AmbiguityResolutionError
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="q2", session_id="s", value="web_search", answered_at=datetime.now(timezone.utc))]
        with pytest.raises(AmbiguityResolutionError, match="multi_choice"):
            resolver._validate_answers(answers, sample_questions)

    def test_invalid_multi_choice_bad_options(self, resolver, sample_questions):
        from nexus.exceptions import AmbiguityResolutionError
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="q2", session_id="s", value=["web_search", "INVALID"], answered_at=datetime.now(timezone.utc))]
        with pytest.raises(AmbiguityResolutionError, match="invalid options"):
            resolver._validate_answers(answers, sample_questions)

    def test_valid_boolean(self, resolver, sample_questions):
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="q3", session_id="s", value=True, answered_at=datetime.now(timezone.utc))]
        resolver._validate_answers(answers, sample_questions)

    def test_invalid_boolean_string(self, resolver, sample_questions):
        from nexus.exceptions import AmbiguityResolutionError
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="q3", session_id="s", value="true", answered_at=datetime.now(timezone.utc))]
        with pytest.raises(AmbiguityResolutionError, match="boolean"):
            resolver._validate_answers(answers, sample_questions)

    def test_valid_text(self, resolver, sample_questions):
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="q4", session_id="s", value="Postgres DB", answered_at=datetime.now(timezone.utc))]
        resolver._validate_answers(answers, sample_questions)

    def test_invalid_text_too_long(self, resolver, sample_questions):
        from nexus.exceptions import AmbiguityResolutionError
        from nexus.types import ClarifyingAnswer
        long_text = "x" * 201  # max_chars=200
        answers = [ClarifyingAnswer(question_id="q4", session_id="s", value=long_text, answered_at=datetime.now(timezone.utc))]
        with pytest.raises(AmbiguityResolutionError, match="character limit"):
            resolver._validate_answers(answers, sample_questions)

    def test_valid_number(self, resolver, sample_questions):
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="q5", session_id="s", value=500, answered_at=datetime.now(timezone.utc))]
        resolver._validate_answers(answers, sample_questions)

    def test_invalid_number_above_max(self, resolver, sample_questions):
        from nexus.exceptions import AmbiguityResolutionError
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="q5", session_id="s", value=20000, answered_at=datetime.now(timezone.utc))]
        with pytest.raises(AmbiguityResolutionError, match="exceeds maximum"):
            resolver._validate_answers(answers, sample_questions)

    def test_invalid_number_below_min(self, resolver, sample_questions):
        from nexus.exceptions import AmbiguityResolutionError
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="q5", session_id="s", value=-1, answered_at=datetime.now(timezone.utc))]
        with pytest.raises(AmbiguityResolutionError, match="below minimum"):
            resolver._validate_answers(answers, sample_questions)

    def test_invalid_number_string(self, resolver, sample_questions):
        from nexus.exceptions import AmbiguityResolutionError
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="q5", session_id="s", value="100", answered_at=datetime.now(timezone.utc))]
        with pytest.raises(AmbiguityResolutionError, match="number"):
            resolver._validate_answers(answers, sample_questions)

    def test_unknown_question_id(self, resolver, sample_questions):
        from nexus.exceptions import AmbiguityResolutionError
        from nexus.types import ClarifyingAnswer
        answers = [ClarifyingAnswer(question_id="nonexistent", session_id="s", value="x", answered_at=datetime.now(timezone.utc))]
        with pytest.raises(AmbiguityResolutionError, match="unknown question_id"):
            resolver._validate_answers(answers, sample_questions)

    def test_boolean_with_int_not_bool(self, resolver, sample_questions):
        """int 1 is not a bool (though isinstance(1, int) is True, isinstance(True, bool) is True).
        Booleans in Python ARE ints, so 1 passes the int check but fails bool check.
        We must explicitly reject ints that are not bool."""
        from nexus.exceptions import AmbiguityResolutionError
        from nexus.types import ClarifyingAnswer
        # bool is a subclass of int in Python, but 1 (pure int) is not bool
        answers = [ClarifyingAnswer(question_id="q3", session_id="s", value=1, answered_at=datetime.now(timezone.utc))]
        with pytest.raises(AmbiguityResolutionError, match="boolean"):
            resolver._validate_answers(answers, sample_questions)


# ─────────────────────────────────────────────────────────────────────────────
# _merge_answers
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeAnswers:
    def test_merge_empty(self, resolver):
        result = resolver._merge_answers([], [])
        assert result == []

    def test_merge_new_only(self, resolver):
        from nexus.types import ClarifyingAnswer
        new = [ClarifyingAnswer(question_id="q1", session_id="s", value="x", answered_at=datetime.now(timezone.utc))]
        result = resolver._merge_answers([], new)
        assert len(result) == 1
        assert result[0].question_id == "q1"

    def test_merge_overwrite(self, resolver):
        from nexus.types import ClarifyingAnswer
        now = datetime.now(timezone.utc)
        existing = [ClarifyingAnswer(question_id="q1", session_id="s", value="old", answered_at=now)]
        new = [ClarifyingAnswer(question_id="q1", session_id="s", value="new", answered_at=now)]
        result = resolver._merge_answers(existing, new)
        assert len(result) == 1
        assert result[0].value == "new"

    def test_merge_preserves_existing_not_overwritten(self, resolver):
        from nexus.types import ClarifyingAnswer
        now = datetime.now(timezone.utc)
        existing = [
            ClarifyingAnswer(question_id="q1", session_id="s", value="v1", answered_at=now),
            ClarifyingAnswer(question_id="q2", session_id="s", value="v2", answered_at=now),
        ]
        new = [ClarifyingAnswer(question_id="q1", session_id="s", value="new_v1", answered_at=now)]
        result = resolver._merge_answers(existing, new)
        assert len(result) == 2
        q1 = next(r for r in result if r.question_id == "q1")
        q2 = next(r for r in result if r.question_id == "q2")
        assert q1.value == "new_v1"
        assert q2.value == "v2"


# ─────────────────────────────────────────────────────────────────────────────
# _seal_plan
# ─────────────────────────────────────────────────────────────────────────────

class TestSealPlan:
    def test_fingerprint_is_sha256_hex(self, resolver):
        from nexus.types import WorkflowPlan
        plan = WorkflowPlan(
            id=str(uuid.uuid4()),
            session_id="sess-1",
            tenant_id="tenant-1",
            original_description="Do something",
            refined_description="Do something specific",
            parameters=[],
            specificity_score=0.8,
        )
        sealed = resolver._seal_plan(plan)
        assert sealed.seal_fingerprint is not None
        assert len(sealed.seal_fingerprint) == 64  # SHA256 hex
        int(sealed.seal_fingerprint, 16)  # Valid hex

    def test_fingerprint_tamper_evidence(self, resolver):
        from nexus.types import WorkflowPlan
        plan1 = WorkflowPlan(
            id=str(uuid.uuid4()),
            session_id="sess-1",
            tenant_id="tenant-1",
            original_description="Do something",
            refined_description="Original refined",
            parameters=[],
            specificity_score=0.8,
        )
        plan2 = WorkflowPlan(
            id=str(uuid.uuid4()),
            session_id="sess-1",
            tenant_id="tenant-1",
            original_description="Do something",
            refined_description="Modified refined",  # Changed!
            parameters=[],
            specificity_score=0.8,
        )
        sealed1 = resolver._seal_plan(plan1)
        sealed2 = resolver._seal_plan(plan2)
        assert sealed1.seal_fingerprint != sealed2.seal_fingerprint

    def test_fingerprint_deterministic(self, resolver):
        from nexus.types import WorkflowPlan
        plan = WorkflowPlan(
            id="fixed-id",
            session_id="sess-1",
            tenant_id="tenant-1",
            original_description="Do something",
            refined_description="Do it precisely",
            parameters=[],
            specificity_score=0.8,
        )
        sealed1 = resolver._seal_plan(plan)
        plan.seal_fingerprint = None  # Reset
        sealed2 = resolver._seal_plan(plan)
        assert sealed1.seal_fingerprint == sealed2.seal_fingerprint


# ─────────────────────────────────────────────────────────────────────────────
# _build_auto_complete_session
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAutoCompleteSession:
    def test_returns_complete_session(self, resolver):
        from nexus.types import AmbiguitySessionStatus, SpecificityScore
        score = SpecificityScore(
            score=0.85,
            dimensions_resolved=["trigger", "tools"],
            dimensions_missing=[],
            tool_coverage_ratio=1.0,
            has_trigger=True,
            has_success_condition=True,
            has_scope_boundary=True,
            reasoning="Fully specified.",
            can_auto_generate=True,
        )
        session = resolver._build_auto_complete_session("t1", "Do X every morning", score)
        assert session.status == AmbiguitySessionStatus.complete
        assert session.questions == []
        assert session.plan is not None
        assert session.plan.seal_fingerprint is not None
        assert session.plan.refined_description == "Do X every morning"

    def test_session_id_matches_plan_session_id(self, resolver):
        from nexus.types import SpecificityScore
        score = SpecificityScore(
            score=0.9, dimensions_resolved=[], dimensions_missing=[],
            tool_coverage_ratio=1.0, has_trigger=True, has_success_condition=True,
            has_scope_boundary=True, reasoning="x", can_auto_generate=True,
        )
        session = resolver._build_auto_complete_session("t1", "desc", score)
        assert session.plan.session_id == session.id


# ─────────────────────────────────────────────────────────────────────────────
# plan_to_generator_context
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanToGeneratorContext:
    def _make_plan(self, params):
        from nexus.types import WorkflowPlan, WorkflowPlanParameter
        return WorkflowPlan(
            id=str(uuid.uuid4()),
            session_id="s",
            tenant_id="t",
            original_description="x",
            refined_description="y",
            parameters=[
                WorkflowPlanParameter(key=k, value=v, source="user_answer", confidence=1.0)
                for k, v in params.items()
            ],
            specificity_score=0.8,
        )

    def test_trigger_type(self, resolver):
        plan = self._make_plan({"trigger_type": "webhook"})
        ctx = resolver.plan_to_generator_context(plan)
        assert ctx["trigger_type"] == "webhook"

    def test_preferred_tools_list(self, resolver):
        plan = self._make_plan({"preferred_tools": ["slack", "gmail"]})
        ctx = resolver.plan_to_generator_context(plan)
        assert ctx["preferred_tools"] == ["slack", "gmail"]

    def test_preferred_tools_string_wrapped_in_list(self, resolver):
        plan = self._make_plan({"preferred_tools": "slack"})
        ctx = resolver.plan_to_generator_context(plan)
        assert ctx["preferred_tools"] == ["slack"]

    def test_preferred_personas(self, resolver):
        plan = self._make_plan({"preferred_personas": ["researcher"]})
        ctx = resolver.plan_to_generator_context(plan)
        assert ctx["preferred_personas"] == ["researcher"]

    def test_schedule_expression_sets_cron_trigger(self, resolver):
        plan = self._make_plan({"schedule_expression": "0 9 * * *"})
        ctx = resolver.plan_to_generator_context(plan)
        assert ctx["trigger_type"] == "cron"
        assert ctx["example_data"]["cron_expression"] == "0 9 * * *"

    def test_authority_level_in_example_data(self, resolver):
        plan = self._make_plan({"authority_level": "autonomous"})
        ctx = resolver.plan_to_generator_context(plan)
        assert ctx["example_data"]["authority_level"] == "autonomous"

    def test_error_behavior_in_example_data(self, resolver):
        plan = self._make_plan({"error_behavior": "escalate"})
        ctx = resolver.plan_to_generator_context(plan)
        assert ctx["example_data"]["error_behavior"] == "escalate"

    def test_empty_parameters(self, resolver):
        plan = self._make_plan({})
        ctx = resolver.plan_to_generator_context(plan)
        assert ctx == {}


# ─────────────────────────────────────────────────────────────────────────────
# score() — mocked LLM
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreMethod:
    @pytest.mark.asyncio
    async def test_score_vague_description(self, resolver, mock_llm):
        llm_response = json.dumps({
            "score": 0.08,
            "dimensions_resolved": [],
            "dimensions_missing": ["trigger", "tools", "scope", "data_sources",
                                   "success_condition", "error_handling", "authority"],
            "tool_coverage_ratio": 0.0,
            "has_trigger": False,
            "has_success_condition": False,
            "has_scope_boundary": False,
            "reasoning": "The description 'handle customer service' provides no actionable detail.",
        })
        mock_llm.complete.return_value = {"content": llm_response}

        score = await resolver.score("t1", "handle customer service")
        assert score.score < 0.30
        assert score.can_auto_generate is False
        assert "trigger" in score.dimensions_missing

    @pytest.mark.asyncio
    async def test_score_specific_description(self, resolver, mock_llm):
        llm_response = json.dumps({
            "score": 0.80,
            "dimensions_resolved": ["trigger", "tools", "scope", "success_condition"],
            "dimensions_missing": [],
            "tool_coverage_ratio": 1.0,
            "has_trigger": True,
            "has_success_condition": True,
            "has_scope_boundary": True,
            "reasoning": "Fully specified.",
        })
        mock_llm.complete.return_value = {"content": llm_response}

        score = await resolver.score(
            "t1",
            "Every morning at 9am, fetch top 5 HN posts via HTTP and send to Slack #engineering",
        )
        assert score.score >= 0.75
        assert score.can_auto_generate is True

    @pytest.mark.asyncio
    async def test_score_llm_failure_returns_zero(self, resolver, mock_llm):
        mock_llm.complete.return_value = {"content": "I cannot parse this"}

        score = await resolver.score("t1", "do something")
        assert score.score == 0.0
        assert score.can_auto_generate is False


# ─────────────────────────────────────────────────────────────────────────────
# start_session() — mocked LLM + repository
# ─────────────────────────────────────────────────────────────────────────────

class TestStartSession:
    def _make_mock_repo(self):
        repo = MagicMock()
        repo.create_ambiguity_session = AsyncMock()
        return repo

    @pytest.mark.asyncio
    async def test_low_score_returns_active_session_with_questions(self, resolver, mock_llm):
        # LLM returns low score, then returns questions
        low_score_response = json.dumps({
            "score": 0.08,
            "dimensions_resolved": [],
            "dimensions_missing": ["trigger", "tools", "scope"],
            "tool_coverage_ratio": 0.0,
            "has_trigger": False,
            "has_success_condition": False,
            "has_scope_boundary": False,
            "reasoning": "Very vague.",
        })
        questions_response = json.dumps([
            {
                "dimension": "trigger",
                "question": "What triggers this?",
                "question_type": "single_choice",
                "options": ["webhook", "cron", "manual"],
                "required": True,
                "default": None,
                "hint": None,
                "maps_to_param": "trigger_type",
            }
        ])
        mock_llm.complete.side_effect = [
            {"content": low_score_response},
            {"content": questions_response},
        ]

        repo = self._make_mock_repo()
        from nexus.types import AmbiguitySessionStatus
        session = await resolver.start_session("t1", "handle customer service", repo)

        assert session.status == AmbiguitySessionStatus.active
        assert len(session.questions) >= 1
        assert session.current_round == 1
        repo.create_ambiguity_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_high_score_returns_complete_session(self, resolver, mock_llm):
        high_score_response = json.dumps({
            "score": 0.85,
            "dimensions_resolved": ["trigger", "tools", "scope", "success_condition"],
            "dimensions_missing": [],
            "tool_coverage_ratio": 1.0,
            "has_trigger": True,
            "has_success_condition": True,
            "has_scope_boundary": True,
            "reasoning": "Fully specified.",
        })
        mock_llm.complete.return_value = {"content": high_score_response}

        repo = self._make_mock_repo()
        from nexus.types import AmbiguitySessionStatus
        session = await resolver.start_session(
            "t1",
            "Every morning at 9am, fetch top 5 HN posts via HTTP and send to Slack #engineering",
            repo,
        )
        assert session.status == AmbiguitySessionStatus.complete
        assert session.plan is not None


# ─────────────────────────────────────────────────────────────────────────────
# submit_answers() — mocked repository
# ─────────────────────────────────────────────────────────────────────────────

class TestSubmitAnswers:
    def _make_mock_repo(self, session_obj):
        repo = MagicMock()
        repo.get_ambiguity_session = AsyncMock(return_value=session_obj)
        repo.update_ambiguity_session = AsyncMock(return_value=session_obj)
        return repo

    @pytest.mark.asyncio
    async def test_raises_on_session_not_found(self, resolver):
        repo = MagicMock()
        repo.get_ambiguity_session = AsyncMock(return_value=None)
        from nexus.exceptions import AmbiguityResolutionError
        with pytest.raises(AmbiguityResolutionError, match="not found"):
            await resolver.submit_answers("bad-id", "t1", [], repo)

    @pytest.mark.asyncio
    async def test_raises_on_non_active_session(self, resolver, active_session):
        from nexus.exceptions import AmbiguityResolutionError
        from nexus.types import AmbiguitySessionStatus
        active_session.status = AmbiguitySessionStatus.complete
        repo = self._make_mock_repo(active_session)
        with pytest.raises(AmbiguityResolutionError, match="not active"):
            await resolver.submit_answers("sess-1", "tenant-1", [], repo)

    @pytest.mark.asyncio
    async def test_raises_on_expired_session(self, resolver, active_session, mock_llm):
        from nexus.exceptions import AmbiguityResolutionError
        active_session.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        repo = self._make_mock_repo(active_session)
        repo.update_ambiguity_session = AsyncMock(return_value=active_session)
        with pytest.raises(AmbiguityResolutionError, match="expired"):
            await resolver.submit_answers("sess-1", "tenant-1", [], repo)

    @pytest.mark.asyncio
    async def test_raises_on_invalid_answer(self, resolver, active_session, mock_llm):
        from nexus.exceptions import AmbiguityResolutionError
        from nexus.types import ClarifyingAnswer
        bad_answer = ClarifyingAnswer(question_id="q1", session_id="sess-1", value="INVALID_OPTION", answered_at=datetime.now(timezone.utc))
        repo = self._make_mock_repo(active_session)
        with pytest.raises(AmbiguityResolutionError, match="single_choice"):
            await resolver.submit_answers("sess-1", "tenant-1", [bad_answer], repo)

    @pytest.mark.asyncio
    async def test_threshold_met_returns_complete(self, resolver, active_session, mock_llm):
        from nexus.types import AmbiguitySessionStatus, ClarifyingAnswer

        # Score response: above threshold
        high_score = json.dumps({
            "score": 0.82,
            "dimensions_resolved": ["trigger", "tools", "scope", "success_condition"],
            "dimensions_missing": [],
            "tool_coverage_ratio": 1.0,
            "has_trigger": True, "has_success_condition": True, "has_scope_boundary": True,
            "reasoning": "Fully specified after answers.",
        })
        # Synthesise response
        refined_desc = "When a webhook fires, search using researcher persona and post to Slack."
        mock_llm.complete.side_effect = [
            {"content": high_score},
            {"content": refined_desc},
        ]

        # Update returns a complete session
        complete_session = active_session.model_copy(deep=True)
        complete_session.status = AmbiguitySessionStatus.complete
        from nexus.types import WorkflowPlan
        complete_session.plan = WorkflowPlan(
            id=str(uuid.uuid4()),
            session_id="sess-1",
            tenant_id="tenant-1",
            original_description="handle customer service",
            refined_description=refined_desc,
            parameters=[],
            specificity_score=0.82,
        )
        repo = self._make_mock_repo(active_session)
        repo.update_ambiguity_session = AsyncMock(return_value=complete_session)

        answer = ClarifyingAnswer(question_id="q1", session_id="sess-1", value="webhook", answered_at=datetime.now(timezone.utc))
        result = await resolver.submit_answers("sess-1", "tenant-1", [answer], repo)
        assert result.status == AmbiguitySessionStatus.complete

    @pytest.mark.asyncio
    async def test_rounds_exhausted_returns_complete(self, resolver, active_session, mock_llm):
        from nexus.types import AmbiguitySessionStatus, ClarifyingAnswer
        active_session.current_round = 3  # At max_rounds
        active_session.max_rounds = 3

        low_score = json.dumps({
            "score": 0.40,
            "dimensions_resolved": ["tools"],
            "dimensions_missing": ["trigger", "scope"],
            "tool_coverage_ratio": 0.5,
            "has_trigger": False, "has_success_condition": False, "has_scope_boundary": False,
            "reasoning": "Still partial.",
        })
        refined_desc = "Partially specified workflow."
        mock_llm.complete.side_effect = [
            {"content": low_score},
            {"content": refined_desc},
        ]

        complete_session = active_session.model_copy(deep=True)
        complete_session.status = AmbiguitySessionStatus.complete
        from nexus.types import WorkflowPlan
        complete_session.plan = WorkflowPlan(
            id=str(uuid.uuid4()), session_id="sess-1", tenant_id="tenant-1",
            original_description="x", refined_description=refined_desc,
            parameters=[], specificity_score=0.40,
        )
        repo = self._make_mock_repo(active_session)
        repo.update_ambiguity_session = AsyncMock(return_value=complete_session)

        answer = ClarifyingAnswer(question_id="q1", session_id="sess-1", value="webhook", answered_at=datetime.now(timezone.utc))
        result = await resolver.submit_answers("sess-1", "tenant-1", [answer], repo)
        assert result.status == AmbiguitySessionStatus.complete


# ─────────────────────────────────────────────────────────────────────────────
# get_session / cancel_session
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAndCancelSession:
    @pytest.mark.asyncio
    async def test_get_session_found(self, resolver, active_session):
        repo = MagicMock()
        repo.get_ambiguity_session = AsyncMock(return_value=active_session)
        result = await resolver.get_session("sess-1", "tenant-1", repo)
        assert result.id == "sess-1"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, resolver):
        from nexus.exceptions import AmbiguityResolutionError
        repo = MagicMock()
        repo.get_ambiguity_session = AsyncMock(return_value=None)
        with pytest.raises(AmbiguityResolutionError):
            await resolver.get_session("bad", "t1", repo)

    @pytest.mark.asyncio
    async def test_get_session_tenant_mismatch(self, resolver, active_session):
        from nexus.exceptions import AmbiguityResolutionError
        repo = MagicMock()
        repo.get_ambiguity_session = AsyncMock(return_value=active_session)
        with pytest.raises(AmbiguityResolutionError):
            await resolver.get_session("sess-1", "WRONG-TENANT", repo)

    @pytest.mark.asyncio
    async def test_cancel_active_session(self, resolver, active_session):
        from nexus.types import AmbiguitySessionStatus
        abandoned = active_session.model_copy(deep=True)
        abandoned.status = AmbiguitySessionStatus.abandoned
        repo = MagicMock()
        repo.get_ambiguity_session = AsyncMock(return_value=active_session)
        repo.update_ambiguity_session = AsyncMock(return_value=abandoned)
        result = await resolver.cancel_session("sess-1", "tenant-1", repo)
        assert result.status == AmbiguitySessionStatus.abandoned

    @pytest.mark.asyncio
    async def test_cancel_complete_session_noop(self, resolver, active_session):
        from nexus.types import AmbiguitySessionStatus
        complete_session = active_session.model_copy(deep=True)
        complete_session.status = AmbiguitySessionStatus.complete
        repo = MagicMock()
        repo.get_ambiguity_session = AsyncMock(return_value=complete_session)
        result = await resolver.cancel_session("sess-1", "tenant-1", repo)
        # Should return unchanged (complete stays complete)
        assert result.status == AmbiguitySessionStatus.complete


# ─────────────────────────────────────────────────────────────────────────────
# Config fields
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigFields:
    def test_config_has_ambiguity_fields(self):
        from nexus.config import NexusConfig
        cfg = NexusConfig()
        assert hasattr(cfg, "ambiguity_auto_generate_threshold")
        assert hasattr(cfg, "max_clarification_rounds")
        assert hasattr(cfg, "ambiguity_session_ttl_hours")
        assert hasattr(cfg, "ambiguity_generation_model")
        assert hasattr(cfg, "max_questions_per_round")

    def test_config_defaults(self):
        from nexus.config import NexusConfig
        cfg = NexusConfig()
        assert cfg.ambiguity_auto_generate_threshold == 0.75
        assert cfg.max_clarification_rounds == 3
        assert cfg.ambiguity_session_ttl_hours == 24
        assert cfg.ambiguity_generation_model is None
        assert cfg.max_questions_per_round == 5


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestModelRoundTrip:
    def test_clarifying_question_roundtrip(self):
        from nexus.types import ClarifyingQuestion, QuestionType
        q = ClarifyingQuestion(
            id="test-id",
            session_id="sess",
            dimension="trigger",
            question="What?",
            question_type=QuestionType.single_choice,
            options=["a", "b"],
            required=True,
            maps_to_param="trigger_type",
        )
        data = q.model_dump()
        q2 = ClarifyingQuestion(**data)
        assert q2.id == q.id
        assert q2.question_type == q.question_type

    def test_workflow_plan_roundtrip(self):
        from nexus.types import WorkflowPlan, WorkflowPlanParameter
        plan = WorkflowPlan(
            id=str(uuid.uuid4()),
            session_id="s",
            tenant_id="t",
            original_description="orig",
            refined_description="refined",
            parameters=[
                WorkflowPlanParameter(
                    key="trigger_type", value="webhook",
                    source="user_answer", confidence=1.0,
                    confirmed_at=datetime.now(timezone.utc)
                )
            ],
            specificity_score=0.8,
            seal_fingerprint="abc123",
        )
        data = plan.model_dump()
        plan2 = WorkflowPlan(**data)
        assert plan2.seal_fingerprint == "abc123"
        assert len(plan2.parameters) == 1

    def test_ambiguity_session_roundtrip(self):
        from nexus.types import AmbiguitySession, AmbiguitySessionStatus
        now = datetime.now(timezone.utc)
        session = AmbiguitySession(
            id="sess-rt",
            tenant_id="t",
            original_description="test",
            status=AmbiguitySessionStatus.active,
            questions=[],
            answers=[],
            current_round=1,
            max_rounds=3,
            specificity_history=[0.1, 0.4],
            plan=None,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(hours=24),
        )
        data = session.model_dump()
        session2 = AmbiguitySession(**data)
        assert session2.specificity_history == [0.1, 0.4]
        assert session2.status == AmbiguitySessionStatus.active


# ─────────────────────────────────────────────────────────────────────────────
# DB model (in-memory SQLite round-trip)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRepositoryRoundTrip:
    """Test AmbiguitySession DB round-trip using in-memory SQLite."""

    @pytest.fixture
    async def db_session(self):
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        from nexus.db.models import Base
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with async_session() as session:
            yield session
        await engine.dispose()

    async def test_create_and_retrieve(self, db_session):
        from nexus.db.repository import Repository
        from nexus.types import AmbiguitySession, AmbiguitySessionStatus

        repo = Repository(db_session)

        # Create a tenant first (FK constraint)
        from nexus.db.models import TenantModel
        tenant = TenantModel(id="t1", name="Test", api_key_hash="hash")
        db_session.add(tenant)
        await db_session.commit()

        now = datetime.now(timezone.utc)
        session_obj = AmbiguitySession(
            id="sess-db-1",
            tenant_id="t1",
            original_description="DB test description",
            status=AmbiguitySessionStatus.active,
            questions=[],
            answers=[],
            current_round=1,
            max_rounds=3,
            specificity_history=[0.08],
            plan=None,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(hours=24),
        )
        await repo.create_ambiguity_session(session_obj)
        loaded = await repo.get_ambiguity_session("sess-db-1")

        assert loaded is not None
        assert loaded.id == "sess-db-1"
        assert loaded.tenant_id == "t1"
        assert loaded.status == AmbiguitySessionStatus.active
        assert loaded.specificity_history == [0.08]

    async def test_update_session_status(self, db_session):
        from nexus.db.repository import Repository
        from nexus.types import AmbiguitySession, AmbiguitySessionStatus
        from nexus.db.models import TenantModel

        repo = Repository(db_session)
        tenant = TenantModel(id="t2", name="Test2", api_key_hash="hash2")
        db_session.add(tenant)
        await db_session.commit()

        now = datetime.now(timezone.utc)
        session_obj = AmbiguitySession(
            id="sess-db-2",
            tenant_id="t2",
            original_description="Test",
            status=AmbiguitySessionStatus.active,
            questions=[],
            answers=[],
            current_round=1,
            max_rounds=3,
            specificity_history=[0.1],
            plan=None,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(hours=24),
        )
        await repo.create_ambiguity_session(session_obj)
        updated = await repo.update_ambiguity_session(
            "sess-db-2", {"status": AmbiguitySessionStatus.abandoned}
        )
        assert updated.status == AmbiguitySessionStatus.abandoned

    async def test_expire_abandoned_sessions(self, db_session):
        from nexus.db.repository import Repository
        from nexus.types import AmbiguitySession, AmbiguitySessionStatus
        from nexus.db.models import TenantModel

        repo = Repository(db_session)
        tenant = TenantModel(id="t3", name="Test3", api_key_hash="hash3")
        db_session.add(tenant)
        await db_session.commit()

        now = datetime.now(timezone.utc)
        # Create an expired session
        expired = AmbiguitySession(
            id="sess-expired",
            tenant_id="t3",
            original_description="expired",
            status=AmbiguitySessionStatus.active,
            questions=[],
            answers=[],
            current_round=1,
            max_rounds=3,
            specificity_history=[0.1],
            plan=None,
            created_at=now - timedelta(hours=30),
            updated_at=now - timedelta(hours=30),
            expires_at=now - timedelta(hours=6),  # already expired
        )
        await repo.create_ambiguity_session(expired)

        count = await repo.expire_abandoned_sessions(cutoff=now)
        assert count == 1

        loaded = await repo.get_ambiguity_session("sess-expired")
        assert loaded.status == AmbiguitySessionStatus.abandoned

    async def test_list_sessions(self, db_session):
        from nexus.db.repository import Repository
        from nexus.types import AmbiguitySession, AmbiguitySessionStatus
        from nexus.db.models import TenantModel

        repo = Repository(db_session)
        tenant = TenantModel(id="t4", name="Test4", api_key_hash="hash4")
        db_session.add(tenant)
        await db_session.commit()

        now = datetime.now(timezone.utc)
        for i in range(3):
            s = AmbiguitySession(
                id=f"sess-list-{i}",
                tenant_id="t4",
                original_description=f"desc {i}",
                status=AmbiguitySessionStatus.active,
                questions=[],
                answers=[],
                current_round=1,
                max_rounds=3,
                specificity_history=[0.1],
                plan=None,
                created_at=now + timedelta(seconds=i),
                updated_at=now,
                expires_at=now + timedelta(hours=24),
            )
            await repo.create_ambiguity_session(s)

        sessions = await repo.list_ambiguity_sessions("t4")
        assert len(sessions) == 3


# ─────────────────────────────────────────────────────────────────────────────
# DB model existence
# ─────────────────────────────────────────────────────────────────────────────

def test_ambiguity_session_model_exists():
    from nexus.db.models import AmbiguitySessionModel, Base  # noqa: F401
    assert "ambiguity_sessions" in Base.metadata.tables


# ─────────────────────────────────────────────────────────────────────────────
# CLARIFICATION_DIMENSIONS constant
# ─────────────────────────────────────────────────────────────────────────────

def test_clarification_dimensions_constant():
    from nexus.workflows.ambiguity import CLARIFICATION_DIMENSIONS
    assert "trigger" in CLARIFICATION_DIMENSIONS
    assert "tools" in CLARIFICATION_DIMENSIONS
    assert "scope" in CLARIFICATION_DIMENSIONS
    assert "success_condition" in CLARIFICATION_DIMENSIONS
    assert len(CLARIFICATION_DIMENSIONS) >= 8


# ─────────────────────────────────────────────────────────────────────────────
# HTTP route: POST /v2/workflows/ambiguity/{session_id}/generate returns 422
# when session is not complete (spec acceptance criterion)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_route_returns_422_when_session_not_complete():
    """
    POST /v2/workflows/ambiguity/{session_id}/generate must return HTTP 422
    when the session status is not "complete".
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from nexus.api.routes.workflows import router
    from nexus.types import AmbiguitySession, AmbiguitySessionStatus

    app = FastAPI()
    app.include_router(router)

    # Wire up minimal state
    active_session = AmbiguitySession(
        id="sess-422",
        tenant_id="t1",
        original_description="vague",
        status=AmbiguitySessionStatus.active,
        questions=[],
        answers=[],
        current_round=1,
        max_rounds=3,
        specificity_history=[0.1],
        plan=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    mock_resolver = MagicMock()
    mock_resolver.get_session = AsyncMock(return_value=active_session)

    # Minimal async_session stub so _get_repository doesn't 503
    import contextlib

    @contextlib.asynccontextmanager
    async def _fake_session():
        yield MagicMock()

    mock_async_session = MagicMock(return_value=_fake_session())

    app.state.ambiguity_resolver = mock_resolver
    app.state.workflow_generator = MagicMock()
    app.state.async_session = mock_async_session

    # Middleware stub: inject tenant_id into request.state
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    class _FakeTenantMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request.state.tenant_id = "t1"
            return await call_next(request)

    app.add_middleware(_FakeTenantMiddleware)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/v2/workflows/ambiguity/sess-422/generate",
        json={},
    )
    assert resp.status_code == 422, (
        f"Expected 422 for non-complete session, got {resp.status_code}: {resp.text}"
    )
