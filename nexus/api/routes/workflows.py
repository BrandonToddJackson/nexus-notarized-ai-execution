"""Workflow and ambiguity resolution API routes."""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from nexus.exceptions import AmbiguityResolutionError
from nexus.types import AmbiguitySessionStatus, ClarifyingAnswer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["workflows"])


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class AmbiguityScoreRequest(BaseModel):
    description: str = Field(..., min_length=10, max_length=5000)


class AmbiguityScoreResponse(BaseModel):
    score: float
    can_auto_generate: bool
    dimensions_resolved: list[str]
    dimensions_missing: list[str]
    reasoning: str


class AmbiguityStartRequest(BaseModel):
    description: str = Field(..., min_length=10, max_length=5000)


class AmbiguityStartResponse(BaseModel):
    session_id: str
    status: str
    questions: list[dict]
    specificity_score: float
    can_skip_to_generate: bool


class AmbiguityAnswersRequest(BaseModel):
    answers: list[dict]  # [{question_id: str, value: Any}]


class AmbiguityAnswersResponse(BaseModel):
    session_id: str
    status: str
    current_round: int
    max_rounds: int
    specificity_score: float
    questions: list[dict]
    plan_id: Optional[str] = None
    plan_seal_fingerprint: Optional[str] = None


class AmbiguityGenerateRequest(BaseModel):
    description_override: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Dependency helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_tenant(request: Request) -> str:
    tenant = getattr(request.state, "tenant_id", None)
    if not tenant:
        raise HTTPException(status_code=401, detail="Tenant not authenticated.")
    return tenant


def _get_resolver(request: Request):
    resolver = getattr(request.app.state, "ambiguity_resolver", None)
    if resolver is None:
        raise HTTPException(status_code=503, detail="AmbiguityResolver not initialised.")
    return resolver


async def _get_repository(request: Request):
    async_session = getattr(request.app.state, "async_session", None)
    if async_session is None:
        raise HTTPException(status_code=503, detail="Database not initialised.")
    async with async_session() as session:
        from nexus.db.repository import Repository
        yield Repository(session)


def _get_generator(request: Request):
    gen = getattr(request.app.state, "workflow_generator", None)
    if gen is None:
        raise HTTPException(status_code=503, detail="WorkflowGenerator not initialised.")
    return gen


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/v2/workflows/ambiguity/score", response_model=AmbiguityScoreResponse)
async def score_description(
    body: AmbiguityScoreRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    resolver=Depends(_get_resolver),
):
    """Score a description's specificity. No session is created."""
    score = await resolver.score(tenant_id=tenant_id, description=body.description)
    return AmbiguityScoreResponse(
        score=score.score,
        can_auto_generate=score.can_auto_generate,
        dimensions_resolved=score.dimensions_resolved,
        dimensions_missing=score.dimensions_missing,
        reasoning=score.reasoning,
    )


@router.post("/v2/workflows/ambiguity/start", response_model=AmbiguityStartResponse)
async def start_session(
    body: AmbiguityStartRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    resolver=Depends(_get_resolver),
    repository=Depends(_get_repository),
):
    """Start a clarification session. Returns questions or auto-completes if high-specificity."""
    session = await resolver.start_session(tenant_id, body.description, repository)
    return AmbiguityStartResponse(
        session_id=session.id,
        status=session.status.value,
        questions=[q.model_dump() for q in session.questions],
        specificity_score=session.specificity_history[-1] if session.specificity_history else 0.0,
        can_skip_to_generate=(session.status == AmbiguitySessionStatus.complete),
    )


@router.get("/v2/workflows/ambiguity")
async def list_sessions(
    request: Request,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    tenant_id: str = Depends(_get_tenant),
    repository=Depends(_get_repository),
):
    """List clarification sessions for the tenant."""
    sessions = await repository.list_ambiguity_sessions(
        tenant_id=tenant_id, status=status, limit=limit, offset=offset
    )
    return {"sessions": [s.model_dump() for s in sessions], "total": len(sessions)}


@router.get("/v2/workflows/ambiguity/{session_id}")
async def get_session(
    session_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    resolver=Depends(_get_resolver),
    repository=Depends(_get_repository),
):
    """Get full session state including current questions and plan if complete."""
    try:
        session = await resolver.get_session(session_id, tenant_id, repository)
    except AmbiguityResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return session.model_dump()


@router.post("/v2/workflows/ambiguity/{session_id}/answers", response_model=AmbiguityAnswersResponse)
async def submit_answers(
    session_id: str,
    body: AmbiguityAnswersRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    resolver=Depends(_get_resolver),
    repository=Depends(_get_repository),
):
    """Submit answers to advance the clarification session."""
    answers = [
        ClarifyingAnswer(
            question_id=a["question_id"],
            session_id=session_id,
            value=a["value"],
            answered_at=datetime.now(timezone.utc),
        )
        for a in body.answers
    ]
    try:
        session = await resolver.submit_answers(session_id, tenant_id, answers, repository)
    except AmbiguityResolutionError as exc:
        msg = str(exc)
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg)
        if "expired" in msg:
            raise HTTPException(status_code=410, detail=msg)
        raise HTTPException(status_code=422, detail=msg)

    return AmbiguityAnswersResponse(
        session_id=session.id,
        status=session.status.value,
        current_round=session.current_round,
        max_rounds=session.max_rounds,
        specificity_score=session.specificity_history[-1] if session.specificity_history else 0.0,
        questions=[q.model_dump() for q in session.questions],
        plan_id=session.plan.id if session.plan else None,
        plan_seal_fingerprint=session.plan.seal_fingerprint if session.plan else None,
    )


@router.delete("/v2/workflows/ambiguity/{session_id}")
async def cancel_session(
    session_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    resolver=Depends(_get_resolver),
    repository=Depends(_get_repository),
):
    """Cancel (abandon) a clarification session. No-op if already complete."""
    try:
        session = await resolver.cancel_session(session_id, tenant_id, repository)
    except AmbiguityResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"session_id": session.id, "status": session.status.value}


@router.post("/v2/workflows/ambiguity/{session_id}/generate")
async def generate_from_plan(
    session_id: str,
    body: AmbiguityGenerateRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    resolver=Depends(_get_resolver),
    repository=Depends(_get_repository),
    generator=Depends(_get_generator),
):
    """Trigger workflow generation from a completed clarification plan."""
    try:
        session = await resolver.get_session(session_id, tenant_id, repository)
    except AmbiguityResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if session.status != AmbiguitySessionStatus.complete:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Session {session_id!r} is not complete "
                f"(status={session.status.value}). "
                "Submit all required answers before generating."
            ),
        )

    plan = session.plan
    description = body.description_override or plan.refined_description
    context = resolver.plan_to_generator_context(plan)

    workflow = await generator.generate(
        tenant_id=tenant_id,
        description=description,
        context=context,
    )

    # Record the workflow_definition_id on the plan for audit trail.
    # Pass the Pydantic model directly; repo's _plan_to_json handles serialization.
    updated_plan = plan.model_copy(update={"workflow_definition_id": workflow.id})
    await repository.update_ambiguity_session(session_id, {"plan": updated_plan})
    await repository.update_ambiguity_session(
        session_id, {"status": AmbiguitySessionStatus.generated}
    )

    return workflow.model_dump()
