"""
nexus/workflows/ambiguity.py

AmbiguityResolver: detects under-specified workflow descriptions, runs structured
clarification sessions, produces confirmed WorkflowPlan objects.

Design contract:
- Stateless computation, stateful storage. All session state lives in DB via repository.
- Never calls WorkflowGenerator directly. Returns a WorkflowPlan. Caller decides what to do with it.
- Never generates a DAG. Never calls WorkflowManager. It is upstream of both.
- The WorkflowPlan is Merkle-notarised via SHA256. The seal is the accountability record
  for scope negotiation.
- All LLM calls use AMBIGUITY_SCORE_PROMPT, AMBIGUITY_QUESTIONS_PROMPT, AMBIGUITY_SYNTHESISE_PROMPT.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from nexus.config import NexusConfig
from nexus.exceptions import AmbiguityResolutionError
from nexus.llm.client import LLMClient
from nexus.llm.prompts import (
    AMBIGUITY_QUESTIONS_PROMPT,
    AMBIGUITY_SCORE_PROMPT,
    AMBIGUITY_SYNTHESISE_PROMPT,
)
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

logger = logging.getLogger(__name__)

# Dimensions that must be resolved for a workflow to be generatable.
CLARIFICATION_DIMENSIONS = [
    "trigger",
    "tools",
    "personas",
    "scope",
    "data_sources",
    "output_destinations",
    "success_condition",
    "error_handling",
    "authority",
    "schedule",
]


class AmbiguityResolver:
    """
    Pre-generation stage that scores description specificity and runs structured
    clarification sessions when needed.

    Usage pattern (called by API layer):

      resolver = AmbiguityResolver(llm, registry, persona_manager, config)

      # Step 1: Assess the description.
      score = await resolver.score(tenant_id, description)
      if score.can_auto_generate:
          return await generator.generate(tenant_id, description)

      # Step 2: Start a session.
      session = await resolver.start_session(tenant_id, description, repository)

      # Step 3 (subsequent API call): User submits answers.
      session = await resolver.submit_answers(session_id, tenant_id, answers, repository)

      # Step 4: Check if session is complete.
      if session.status == AmbiguitySessionStatus.complete:
          plan = session.plan
          wf = await generator.generate(
              tenant_id=plan.tenant_id,
              description=plan.refined_description,
              context=resolver.plan_to_generator_context(plan),
          )
      else:
          return session.questions  # Questions for this round.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry,
        persona_manager,
        config: NexusConfig,
        notary=None,
    ) -> None:
        self._llm = llm_client
        self._tool_registry = tool_registry
        self._persona_manager = persona_manager
        self._notary = notary
        self._config = config
        self._auto_threshold: float = getattr(
            config, "ambiguity_auto_generate_threshold", 0.75
        )
        self._max_rounds: int = getattr(config, "max_clarification_rounds", 3)
        self._max_questions: int = getattr(config, "max_questions_per_round", 5)
        self._session_ttl_hours: int = getattr(config, "ambiguity_session_ttl_hours", 24)
        self._model: Optional[str] = (
            getattr(config, "ambiguity_generation_model", None)
            or getattr(config, "workflow_generation_model", None)
        )

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    async def score(
        self,
        tenant_id: str,
        description: str,
        existing_answers: Optional[list[ClarifyingAnswer]] = None,
    ) -> SpecificityScore:
        """
        Compute a specificity score for a description (optionally enriched with answers).

        This method does NOT persist anything. Safe to call multiple times.
        """
        tool_context = self._build_tool_names_list()
        persona_context = self._build_persona_names_list()
        answers_text = self._format_answers_for_prompt(existing_answers or [])

        system_prompt = AMBIGUITY_SCORE_PROMPT.format(
            available_tools=tool_context,
            available_personas=persona_context,
            clarification_dimensions=json.dumps(CLARIFICATION_DIMENSIONS, indent=2),
        )
        user_message = (
            f"Description: {description}"
            + (f"\n\nAnswers from clarification:\n{answers_text}" if answers_text else "")
        )

        response = await self._call_llm(system_prompt, user_message)
        return self._parse_specificity_score(response)

    async def start_session(
        self,
        tenant_id: str,
        description: str,
        repository,
    ) -> AmbiguitySession:
        """
        Start a new clarification session for a vague description.
        Generates the first round of questions and persists the session to DB.

        If score >= threshold, returns a completed session immediately (no questions).
        """
        logger.info(
            "AmbiguityResolver.start_session | tenant=%s | description_len=%d",
            tenant_id, len(description),
        )

        score = await self.score(tenant_id, description)
        if score.can_auto_generate:
            logger.info("Score %.2f >= threshold. Creating auto-complete session.", score.score)
            session = self._build_auto_complete_session(tenant_id, description, score)
            await repository.create_ambiguity_session(session)
            return session

        questions = await self._generate_questions(
            description=description,
            missing_dimensions=score.dimensions_missing,
            existing_answers=[],
            round_number=1,
        )

        now = datetime.now(timezone.utc)
        session = AmbiguitySession(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            original_description=description,
            status=AmbiguitySessionStatus.active,
            questions=questions,
            answers=[],
            current_round=1,
            max_rounds=self._max_rounds,
            specificity_history=[score.score],
            plan=None,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(hours=self._session_ttl_hours),
        )

        await repository.create_ambiguity_session(session)
        logger.info(
            "Session created | id=%s | questions=%d | score=%.2f",
            session.id, len(questions), score.score,
        )
        return session

    async def submit_answers(
        self,
        session_id: str,
        tenant_id: str,
        answers: list[ClarifyingAnswer],
        repository,
    ) -> AmbiguitySession:
        """
        Accept answers, validate, re-score, and advance the session.

        Returns updated session. Check session.status:
          - "active"   → more questions in session.questions (next round)
          - "complete" → session.plan is populated

        Raises AmbiguityResolutionError for invalid session, non-active status,
        expired session, or answer validation failures.
        """
        session = await repository.get_ambiguity_session(session_id)
        if not session:
            raise AmbiguityResolutionError(f"Session {session_id!r} not found.")
        if session.tenant_id != tenant_id:
            raise AmbiguityResolutionError(f"Session {session_id!r} not found.")
        if session.status != AmbiguitySessionStatus.active:
            raise AmbiguityResolutionError(
                f"Session {session_id!r} is {session.status.value}, not active. "
                "Cannot submit answers to a non-active session."
            )
        if datetime.now(timezone.utc) > session.expires_at:
            await repository.update_ambiguity_session(
                session_id, {"status": AmbiguitySessionStatus.abandoned}
            )
            raise AmbiguityResolutionError(
                f"Session {session_id!r} has expired. Start a new session."
            )

        self._validate_answers(answers, session.questions)

        merged_answers = self._merge_answers(session.answers, answers)

        score = await self.score(
            tenant_id=tenant_id,
            description=session.original_description,
            existing_answers=merged_answers,
        )
        specificity_history = session.specificity_history + [score.score]

        rounds_exhausted = session.current_round >= session.max_rounds
        threshold_met = score.can_auto_generate

        if threshold_met or rounds_exhausted:
            # Pass all accumulated questions so _build_plan can map every answer.
            plan = await self._build_plan(
                all_questions=session.questions,
                all_answers=merged_answers,
                session=session,
                score=score,
            )
            plan = self._seal_plan(plan)

            updated = await repository.update_ambiguity_session(
                session_id,
                {
                    "answers": [a.model_dump() for a in merged_answers],
                    "status": AmbiguitySessionStatus.complete,
                    "current_round": session.current_round,
                    "specificity_history": specificity_history,
                    "plan": plan,  # repo serializes via _plan_to_json
                    "updated_at": datetime.now(timezone.utc),
                },
            )
            logger.info(
                "Session %s complete | score=%.2f | rounds=%d | plan_id=%s",
                session_id, score.score, session.current_round, plan.id,
            )
            return updated

        next_questions = await self._generate_questions(
            description=session.original_description,
            missing_dimensions=score.dimensions_missing,
            existing_answers=merged_answers,
            round_number=session.current_round + 1,
        )

        # Accumulate all questions across rounds so _build_plan can always find
        # the maps_to_param for any answer, regardless of which round it came from.
        accumulated_questions = [q.model_dump() for q in session.questions] + [
            q.model_dump() for q in next_questions
        ]

        updated = await repository.update_ambiguity_session(
            session_id,
            {
                "answers": [a.model_dump() for a in merged_answers],
                "questions": accumulated_questions,
                "current_round": session.current_round + 1,
                "specificity_history": specificity_history,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        logger.info(
            "Session %s | round %d → %d | score=%.2f | new_questions=%d",
            session_id, session.current_round, session.current_round + 1,
            score.score, len(next_questions),
        )
        return updated

    async def get_session(
        self,
        session_id: str,
        tenant_id: str,
        repository,
    ) -> AmbiguitySession:
        """Retrieve an existing session. Raises AmbiguityResolutionError if not found."""
        session = await repository.get_ambiguity_session(session_id)
        if not session or session.tenant_id != tenant_id:
            raise AmbiguityResolutionError(f"Session {session_id!r} not found.")
        return session

    async def cancel_session(
        self,
        session_id: str,
        tenant_id: str,
        repository,
    ) -> AmbiguitySession:
        """Mark a session as abandoned. No-op if already complete."""
        session = await repository.get_ambiguity_session(session_id)
        if not session or session.tenant_id != tenant_id:
            raise AmbiguityResolutionError(f"Session {session_id!r} not found.")
        if session.status == AmbiguitySessionStatus.complete:
            return session
        return await repository.update_ambiguity_session(
            session_id,
            {"status": AmbiguitySessionStatus.abandoned, "updated_at": datetime.now(timezone.utc)},
        )

    def plan_to_generator_context(self, plan: WorkflowPlan) -> dict:
        """
        Convert a WorkflowPlan's parameters into the context dict accepted by
        WorkflowGenerator.generate(). This is the bridge between Phase 23.1 and Phase 23.
        """
        context: dict = {}
        for param in plan.parameters:
            if param.key == "trigger_type":
                context["trigger_type"] = param.value
            elif param.key == "preferred_tools":
                context["preferred_tools"] = (
                    param.value if isinstance(param.value, list) else [param.value]
                )
            elif param.key == "preferred_personas":
                context["preferred_personas"] = (
                    param.value if isinstance(param.value, list) else [param.value]
                )
            elif param.key == "example_data":
                context.setdefault("example_data", {})
                context["example_data"].update(
                    param.value if isinstance(param.value, dict) else {}
                )
            elif param.key == "schedule_expression":
                context["trigger_type"] = "cron"
                context.setdefault("example_data", {})
                context["example_data"]["cron_expression"] = param.value
            elif param.key in ("authority_level", "error_behavior"):
                context.setdefault("example_data", {})
                context["example_data"][param.key] = param.value
        return context

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    async def _generate_questions(
        self,
        description: str,
        missing_dimensions: list[str],
        existing_answers: list[ClarifyingAnswer],
        round_number: int,
    ) -> list[ClarifyingQuestion]:
        """Ask the LLM to generate ClarifyingQuestion objects for the missing dimensions."""
        tool_names = self._build_tool_names_list()
        persona_names = self._build_persona_names_list()
        answers_text = self._format_answers_for_prompt(existing_answers)

        system_prompt = AMBIGUITY_QUESTIONS_PROMPT.format(
            available_tools=tool_names,
            available_personas=persona_names,
            max_questions=self._max_questions,
            round_number=round_number,
            max_rounds=self._max_rounds,
        )
        user_message = (
            f"Description: {description}\n\n"
            f"Dimensions needing clarification: {json.dumps(missing_dimensions)}\n\n"
            + (f"Already answered:\n{answers_text}" if answers_text else "")
        )

        response = await self._call_llm(system_prompt, user_message)
        return self._parse_questions(response)

    async def _build_plan(
        self,
        session: AmbiguitySession,
        all_answers: list[ClarifyingAnswer],
        score: SpecificityScore,
        all_questions: Optional[list[ClarifyingQuestion]] = None,
    ) -> WorkflowPlan:
        """Synthesise a final WorkflowPlan from accumulated answers.

        Args:
            all_questions: All questions from all rounds (accumulated). When None,
                           falls back to session.questions (safe for single-round sessions).
        """
        answers_text = self._format_answers_for_prompt(all_answers)
        system_prompt = AMBIGUITY_SYNTHESISE_PROMPT
        user_message = (
            f"Original description: {session.original_description}\n\n"
            f"Confirmed answers:\n{answers_text}"
        )
        response = await self._call_llm(system_prompt, user_message)
        refined_description = self._parse_refined_description(response)

        parameters: list[WorkflowPlanParameter] = []
        # Use all_questions (all rounds) so answers from earlier rounds are not silently dropped.
        questions_to_map = all_questions if all_questions is not None else session.questions
        question_map = {q.id: q for q in questions_to_map}

        for answer in all_answers:
            question = question_map.get(answer.question_id)
            if not question:
                continue
            parameters.append(
                WorkflowPlanParameter(
                    key=question.maps_to_param,
                    value=answer.value,
                    source="user_answer",
                    confidence=1.0,
                    confirmed_at=answer.answered_at,
                )
            )

        return WorkflowPlan(
            id=str(uuid.uuid4()),
            session_id=session.id,
            tenant_id=session.tenant_id,
            original_description=session.original_description,
            refined_description=refined_description,
            parameters=parameters,
            specificity_score=score.score,
            seal_fingerprint=None,
            created_at=datetime.now(timezone.utc),
            workflow_definition_id=None,
        )

    def _seal_plan(self, plan: WorkflowPlan) -> WorkflowPlan:
        """
        Notarise the WorkflowPlan by computing a SHA256 fingerprint.
        Makes scope negotiation tamper-evident.
        """
        try:
            content = json.dumps(
                {
                    "session_id": plan.session_id,
                    "tenant_id": plan.tenant_id,
                    "original_description": plan.original_description,
                    "refined_description": plan.refined_description,
                    "parameters": sorted(
                        [p.model_dump() for p in plan.parameters], key=lambda x: x["key"]
                    ),
                    "specificity_score": plan.specificity_score,
                },
                sort_keys=True,
                default=str,
            )
            plan.seal_fingerprint = hashlib.sha256(content.encode()).hexdigest()
        except Exception as exc:
            logger.error("Failed to seal WorkflowPlan %s: %s", plan.id, exc)
        return plan

    def _validate_answers(
        self,
        answers: list[ClarifyingAnswer],
        session_questions: list[ClarifyingQuestion],
    ) -> None:
        """
        Validate each answer against its question's type constraints.
        Raises AmbiguityResolutionError on the first failure (fail-fast).
        """
        question_map = {q.id: q for q in session_questions}
        for answer in answers:
            q = question_map.get(answer.question_id)
            if not q:
                raise AmbiguityResolutionError(
                    f"Answer references unknown question_id={answer.question_id!r}."
                )
            v = answer.value

            if q.question_type == QuestionType.single_choice:
                if not isinstance(v, str) or v not in q.options:
                    raise AmbiguityResolutionError(
                        f"Question {q.id!r}: single_choice answer must be one of "
                        f"{q.options!r}. Got {v!r}."
                    )
            elif q.question_type == QuestionType.multi_choice:
                if not isinstance(v, list):
                    raise AmbiguityResolutionError(
                        f"Question {q.id!r}: multi_choice answer must be a list. "
                        f"Got {type(v).__name__}."
                    )
                invalid = [x for x in v if x not in q.options]
                if invalid:
                    raise AmbiguityResolutionError(
                        f"Question {q.id!r}: invalid options {invalid!r}. "
                        f"Valid: {q.options!r}."
                    )
            elif q.question_type == QuestionType.boolean:
                if not isinstance(v, bool):
                    raise AmbiguityResolutionError(
                        f"Question {q.id!r}: boolean answer must be True or False (bool). "
                        f"Got {type(v).__name__}: {v!r}."
                    )
            elif q.question_type == QuestionType.number:
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise AmbiguityResolutionError(
                        f"Question {q.id!r}: number answer must be int or float. "
                        f"Got {type(v).__name__}."
                    )
                if q.min_value is not None and v < q.min_value:
                    raise AmbiguityResolutionError(
                        f"Question {q.id!r}: value {v} is below minimum {q.min_value}."
                    )
                if q.max_value is not None and v > q.max_value:
                    raise AmbiguityResolutionError(
                        f"Question {q.id!r}: value {v} exceeds maximum {q.max_value}."
                    )
            elif q.question_type == QuestionType.text:
                if not isinstance(v, str):
                    raise AmbiguityResolutionError(
                        f"Question {q.id!r}: text answer must be a string. "
                        f"Got {type(v).__name__}."
                    )
                max_chars = q.max_chars or 500
                if len(v) > max_chars:
                    raise AmbiguityResolutionError(
                        f"Question {q.id!r}: text answer exceeds {max_chars} character limit "
                        f"(got {len(v)} chars)."
                    )

    def _merge_answers(
        self,
        existing: list[ClarifyingAnswer],
        new: list[ClarifyingAnswer],
    ) -> list[ClarifyingAnswer]:
        """Merge new answers into existing. New answers overwrite existing for same question_id."""
        existing_map = {a.question_id: a for a in existing}
        for a in new:
            existing_map[a.question_id] = a
        return list(existing_map.values())

    def _build_auto_complete_session(
        self,
        tenant_id: str,
        description: str,
        score: SpecificityScore,
    ) -> AmbiguitySession:
        """Build a synthetic completed session for descriptions that pass the auto-generate threshold."""
        now = datetime.now(timezone.utc)
        session_id = str(uuid.uuid4())
        plan = WorkflowPlan(
            id=str(uuid.uuid4()),
            session_id=session_id,
            tenant_id=tenant_id,
            original_description=description,
            refined_description=description,
            parameters=[],
            specificity_score=score.score,
            seal_fingerprint=None,
            created_at=now,
            workflow_definition_id=None,
        )
        plan = self._seal_plan(plan)
        return AmbiguitySession(
            id=session_id,
            tenant_id=tenant_id,
            original_description=description,
            status=AmbiguitySessionStatus.complete,
            questions=[],
            answers=[],
            current_round=0,
            max_rounds=self._max_rounds,
            specificity_history=[score.score],
            plan=plan,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(hours=self._session_ttl_hours),
        )

    async def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Single LLM call. Returns raw text. Uses self._model override if set."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        kwargs: dict = {}
        if self._model:
            kwargs["model"] = self._model
        response = await self._llm.complete(
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
            **kwargs,
        )
        return response["content"]

    def _build_tool_names_list(self) -> str:
        """Return a newline-separated list of registered tool names."""
        try:
            tools = self._tool_registry.list_tools()
            return "\n".join(t.name for t in tools) if tools else "(no tools registered)"
        except Exception:
            return "(tool registry unavailable)"

    def _build_persona_names_list(self) -> str:
        """Return a newline-separated list of persona names."""
        try:
            personas = self._persona_manager.list_personas()
            return "\n".join(p.name for p in personas) if personas else "(no personas registered)"
        except Exception:
            return "(persona manager unavailable)"

    def _format_answers_for_prompt(self, answers: list[ClarifyingAnswer]) -> str:
        """Format existing answers as readable text for LLM prompt injection."""
        if not answers:
            return ""
        return "\n".join(
            f"- Question {a.question_id}: {json.dumps(a.value)}"
            for a in answers
        )

    def _parse_specificity_score(self, llm_response: str) -> SpecificityScore:
        """
        Parse LLM response from AMBIGUITY_SCORE_PROMPT into a SpecificityScore.
        Falls back to conservative zero-score if parsing fails.
        """
        data = self._extract_json(llm_response)
        if not data or not isinstance(data, dict):
            logger.warning("Failed to parse specificity score from LLM. Defaulting to 0.0.")
            return SpecificityScore(
                score=0.0,
                dimensions_resolved=[],
                dimensions_missing=CLARIFICATION_DIMENSIONS[:],
                tool_coverage_ratio=0.0,
                has_trigger=False,
                has_success_condition=False,
                has_scope_boundary=False,
                reasoning="Score parsing failed. Defaulting to full clarification.",
                can_auto_generate=False,
            )

        score_val = float(data.get("score", 0.0))
        return SpecificityScore(
            score=score_val,
            dimensions_resolved=data.get("dimensions_resolved", []),
            dimensions_missing=data.get("dimensions_missing", CLARIFICATION_DIMENSIONS[:]),
            tool_coverage_ratio=float(data.get("tool_coverage_ratio", 0.0)),
            has_trigger=bool(data.get("has_trigger", False)),
            has_success_condition=bool(data.get("has_success_condition", False)),
            has_scope_boundary=bool(data.get("has_scope_boundary", False)),
            reasoning=data.get("reasoning", ""),
            can_auto_generate=score_val >= self._auto_threshold,
        )

    def _parse_questions(self, llm_response: str) -> list[ClarifyingQuestion]:
        """
        Parse LLM response from AMBIGUITY_QUESTIONS_PROMPT into list[ClarifyingQuestion].
        Returns empty list if parsing fails.
        """
        data = self._extract_json(llm_response)
        if not data or not isinstance(data, list):
            logger.warning("Failed to parse questions from LLM. Returning empty list.")
            return []

        questions = []
        for item in data[: self._max_questions]:
            try:
                q = ClarifyingQuestion(
                    id=str(uuid.uuid4()),
                    session_id="pending",
                    dimension=item.get("dimension", "scope"),
                    question=item.get("question", ""),
                    question_type=QuestionType(item.get("question_type", "text")),
                    options=item.get("options") or [],
                    required=bool(item.get("required", True)),
                    default=item.get("default"),
                    hint=item.get("hint"),
                    maps_to_param=item.get("maps_to_param", "scope_boundary"),
                    max_chars=item.get("max_chars"),
                    min_value=item.get("min_value"),
                    max_value=item.get("max_value"),
                )
                questions.append(q)
            except Exception as exc:
                logger.warning("Skipping malformed question item: %s | error: %s", item, exc)
        return questions

    def _parse_refined_description(self, llm_response: str) -> str:
        """
        Parse LLM response from AMBIGUITY_SYNTHESISE_PROMPT.
        Expected: plain text. Falls back gracefully if empty.
        """
        text = llm_response.strip()
        if not text:
            return "(No refined description produced.)"
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()

    def _extract_json(self, text: str) -> Any:
        """
        4-strategy JSON extraction (same pattern as Phase 23).
        Returns parsed object (dict or list) or None on failure.
        """
        text = text.strip()
        # Strategy 1: Direct parse.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Strategy 2: ```json fence.
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Strategy 3: ``` fence (no language).
        m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Strategy 4: First structural bracket to last.
        # Try whichever bracket appears earliest in the text first.
        brace_pos = text.find("{")
        bracket_pos = text.find("[")
        if brace_pos == -1:
            order = [("[", "]"), ("{", "}")]
        elif bracket_pos == -1:
            order = [("{", "}"), ("[", "]")]
        elif bracket_pos < brace_pos:
            order = [("[", "]"), ("{", "}")]
        else:
            order = [("{", "}"), ("[", "]")]
        for open_c, close_c in order:
            start = text.find(open_c)
            end = text.rfind(close_c)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start: end + 1])
                except json.JSONDecodeError:
                    pass
        return None
