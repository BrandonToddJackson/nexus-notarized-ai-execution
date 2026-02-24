"""Workflow and ambiguity resolution API routes."""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from nexus.exceptions import AmbiguityResolutionError, WorkflowGenerationError, WorkflowNotFound
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


def _get_manager(request: Request):
    mgr = getattr(request.app.state, "workflow_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="WorkflowManager not initialised.")
    return mgr


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


# ─────────────────────────────────────────────────────────────────────────────
# Workflow CRUD routes
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowCreateRequest(BaseModel):
    name: str
    description: str = ""
    steps: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    created_by: str = ""
    tags: list[str] = Field(default_factory=list)
    settings: dict = Field(default_factory=dict)
    trigger_config: dict = Field(default_factory=dict)


class WorkflowStatusRequest(BaseModel):
    status: str


class WorkflowGenerateRequest(BaseModel):
    description: str = Field(..., min_length=10, max_length=5000)


@router.get("/v2/workflows")
async def list_workflows(
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """List all workflows for the tenant."""
    workflows = await manager.list(tenant_id)
    return {"workflows": [w.model_dump() for w in workflows]}


@router.post("/v2/workflows", status_code=201)
async def create_workflow(
    body: WorkflowCreateRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Create a new workflow."""
    workflow = await manager.create(
        tenant_id=tenant_id,
        name=body.name,
        description=body.description,
        steps=body.steps,
        edges=body.edges,
        created_by=body.created_by,
        tags=body.tags,
        settings=body.settings,
        trigger_config=body.trigger_config,
    )
    return workflow.model_dump()


@router.get("/v2/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Get a single workflow by ID."""
    try:
        workflow = await manager.get(workflow_id, tenant_id)
    except WorkflowNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow.model_dump()


@router.put("/v2/workflows/{workflow_id}")
async def update_workflow(
    workflow_id: str,
    body: dict,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Full update of a workflow."""
    try:
        workflow = await manager.update(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            name=body.get("name"),
            description=body.get("description"),
            steps=body.get("steps"),
            edges=body.get("edges"),
            tags=body.get("tags"),
            settings=body.get("settings"),
            trigger_config=body.get("trigger_config"),
        )
    except WorkflowNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow.model_dump()


@router.patch("/v2/workflows/{workflow_id}/status")
async def update_workflow_status(
    workflow_id: str,
    body: WorkflowStatusRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Update workflow status (active, paused, archived)."""
    status_actions = {
        "active": manager.activate,
        "paused": manager.pause,
        "archived": manager.archive,
    }
    action = status_actions.get(body.status)
    if action is None:
        raise HTTPException(status_code=400, detail="Invalid status")
    try:
        workflow = await action(workflow_id, tenant_id)
    except WorkflowNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow.model_dump()


@router.post("/v2/workflows/generate")
async def generate_workflow(
    body: WorkflowGenerateRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    generator=Depends(_get_generator),
):
    """Generate a workflow from a natural language description."""
    try:
        workflow = await generator.generate(
            description=body.description,
            tenant_id=tenant_id,
        )
    except WorkflowGenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return workflow.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Templates, import/export, duplicate
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATES = [
    {
        "id": "data-pipeline",
        "name": "Data Pipeline",
        "description": "Extract, transform, and load data from an API source",
        "steps": [
            {"id": "extract", "step_type": "action", "name": "Extract Data", "tool_name": "http_request", "persona_name": "researcher"},
            {"id": "transform", "step_type": "action", "name": "Transform Data", "tool_name": "data_transform", "persona_name": "analyst"},
        ],
        "edges": [
            {"source_step_id": "extract", "target_step_id": "transform", "edge_type": "default"},
        ],
        "tags": ["template", "data"],
    },
    {
        "id": "research-report",
        "name": "Research Report",
        "description": "Search knowledge base and generate a summary report",
        "steps": [
            {"id": "search", "step_type": "action", "name": "Search Knowledge", "tool_name": "knowledge_search", "persona_name": "researcher"},
            {"id": "summarize", "step_type": "action", "name": "Summarize Findings", "tool_name": "knowledge_search", "persona_name": "researcher"},
        ],
        "edges": [
            {"source_step_id": "search", "target_step_id": "summarize", "edge_type": "default"},
        ],
        "tags": ["template", "research"],
    },
    {
        "id": "approval-flow",
        "name": "Approval Flow",
        "description": "Process with human approval gate before execution",
        "steps": [
            {"id": "prepare", "step_type": "action", "name": "Prepare Request", "tool_name": "knowledge_search", "persona_name": "researcher"},
            {"id": "approve", "step_type": "human_approval", "name": "Manager Approval", "config": {"message": "Please review and approve"}},
            {"id": "execute", "step_type": "action", "name": "Execute Action", "tool_name": "http_request", "persona_name": "operator"},
        ],
        "edges": [
            {"source_step_id": "prepare", "target_step_id": "approve", "edge_type": "default"},
            {"source_step_id": "approve", "target_step_id": "execute", "edge_type": "default"},
        ],
        "tags": ["template", "approval"],
    },
]


@router.get("/v2/workflows/templates")
async def list_templates(request: Request):
    """Return static list of workflow templates."""
    return {"templates": _TEMPLATES}


@router.post("/v2/workflows/from-template/{template_id}", status_code=201)
async def create_from_template(
    template_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Create a workflow from a template."""
    template = next((t for t in _TEMPLATES if t["id"] == template_id), None)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    workflow = await manager.create(
        tenant_id=tenant_id,
        name=template["name"],
        description=template.get("description", ""),
        steps=template.get("steps", []),
        edges=template.get("edges", []),
        tags=template.get("tags", []),
    )
    return workflow.model_dump()


@router.get("/v2/workflows/{workflow_id}/export")
async def export_workflow(
    workflow_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Export a workflow as JSON."""
    try:
        data = await manager.export_json(workflow_id, tenant_id)
    except WorkflowNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {"data": data}


@router.post("/v2/workflows/import", status_code=201)
async def import_workflow(
    body: dict,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Import a workflow from JSON."""
    json_str = body.get("data", "")
    if not json_str:
        raise HTTPException(status_code=422, detail="Missing 'data' field")
    try:
        workflow = await manager.import_json(json_str, tenant_id)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return workflow.model_dump()


@router.post("/v2/workflows/{workflow_id}/duplicate", status_code=201)
async def duplicate_workflow(
    workflow_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Duplicate a workflow."""
    try:
        workflow = await manager.duplicate(workflow_id, tenant_id)
    except WorkflowNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow.model_dump()


@router.patch("/v2/workflows/{workflow_id}")
async def patch_workflow(
    workflow_id: str,
    body: dict,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Partial update of a workflow, including active/status bool support."""
    from nexus.types import WorkflowStatus

    status = None
    if "status" in body:
        try:
            status = WorkflowStatus(body["status"])
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid status: {body['status']}")
    if "active" in body:
        status = WorkflowStatus.ACTIVE if body["active"] else WorkflowStatus.ARCHIVED

    try:
        workflow = await manager.update(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            name=body.get("name"),
            description=body.get("description"),
            steps=body.get("steps"),
            edges=body.get("edges"),
            tags=body.get("tags"),
            settings=body.get("settings"),
            trigger_config=body.get("trigger_config"),
            status=status,
        )
    except WorkflowNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow.model_dump()
