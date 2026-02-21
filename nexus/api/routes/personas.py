"""CRUD /v1/personas — Behavioral contract management."""

from fastapi import APIRouter, Request, HTTPException
from nexus.api.schemas import CreatePersonaRequest, PersonaResponse
from nexus.types import PersonaContract, RiskLevel

router = APIRouter(tags=["personas"])


def _contract_to_response(p: PersonaContract) -> PersonaResponse:
    return PersonaResponse(
        id=str(p.id),
        name=p.name,
        description=p.description,
        allowed_tools=p.allowed_tools,
        resource_scopes=p.resource_scopes,
        intent_patterns=p.intent_patterns,
        max_ttl_seconds=p.max_ttl_seconds,
        risk_tolerance=p.risk_tolerance.value if hasattr(p.risk_tolerance, "value") else p.risk_tolerance,
        version=p.version,
        is_active=p.is_active,
    )


@router.get("/personas")
async def list_personas(request: Request):
    """List all personas for the authenticated tenant."""
    persona_manager = request.app.state.persona_manager
    personas = persona_manager.list_personas()
    return {"personas": [_contract_to_response(p) for p in personas]}


@router.post("/personas", response_model=PersonaResponse)
async def create_persona(request: Request, body: CreatePersonaRequest):
    """Create a new persona contract."""
    tenant_id = getattr(request.state, "tenant_id", "demo")

    try:
        risk = RiskLevel(body.risk_tolerance)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid risk_tolerance: {body.risk_tolerance}")

    contract = PersonaContract(
        name=body.name,
        description=body.description,
        allowed_tools=body.allowed_tools,
        resource_scopes=body.resource_scopes,
        intent_patterns=body.intent_patterns,
        max_ttl_seconds=body.max_ttl_seconds,
        risk_tolerance=risk,
    )

    # Persist to DB
    async_session = request.app.state.async_session
    async with async_session() as session:
        from nexus.db.repository import Repository
        repo = Repository(session)
        await repo.upsert_persona(tenant_id, {
            "name": body.name,
            "description": body.description,
            "allowed_tools": body.allowed_tools,
            "resource_scopes": body.resource_scopes,
            "intent_patterns": body.intent_patterns,
            "max_ttl_seconds": body.max_ttl_seconds,
            "risk_tolerance": body.risk_tolerance,
        })

    # Load into in-memory manager
    persona_manager = request.app.state.persona_manager
    persona_manager.load_personas([contract])

    return _contract_to_response(contract)


@router.put("/personas/{persona_id}", response_model=PersonaResponse)
async def update_persona(request: Request, persona_id: str, body: CreatePersonaRequest):
    """Update an existing persona contract."""
    tenant_id = getattr(request.state, "tenant_id", "demo")

    try:
        risk = RiskLevel(body.risk_tolerance)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid risk_tolerance: {body.risk_tolerance}")

    # Update in DB
    async_session = request.app.state.async_session
    async with async_session() as session:
        from nexus.db.repository import Repository
        repo = Repository(session)
        db_persona = await repo.get_persona(tenant_id, persona_id)
        if db_persona is None:
            raise HTTPException(status_code=404, detail=f"Persona {persona_id} not found")
        if getattr(db_persona, "tenant_id", None) != tenant_id:
            raise HTTPException(status_code=403, detail="Forbidden")

        await repo.upsert_persona(tenant_id, {
            "name": body.name,
            "description": body.description,
            "allowed_tools": body.allowed_tools,
            "resource_scopes": body.resource_scopes,
            "intent_patterns": body.intent_patterns,
            "max_ttl_seconds": body.max_ttl_seconds,
            "risk_tolerance": body.risk_tolerance,
        })

    # Update in-memory manager
    contract = PersonaContract(
        name=body.name,
        description=body.description,
        allowed_tools=body.allowed_tools,
        resource_scopes=body.resource_scopes,
        intent_patterns=body.intent_patterns,
        max_ttl_seconds=body.max_ttl_seconds,
        risk_tolerance=risk,
    )
    persona_manager = request.app.state.persona_manager
    persona_manager.load_personas([contract])

    return _contract_to_response(contract)
