"""CRUD /v1/personas — Behavioral contract management."""

from fastapi import APIRouter, Request, HTTPException
from nexus.api.schemas import CreatePersonaRequest, PersonaResponse

router = APIRouter(tags=["personas"])


@router.get("/personas")
async def list_personas(request: Request):
    """List all personas for the authenticated tenant."""
    # TODO: Query personas from repository
    return {"personas": []}


@router.post("/personas")
async def create_persona(request: Request, body: CreatePersonaRequest):
    """Create a new persona contract."""
    # TODO: Create persona via repository
    return {"id": "new", "name": body.name, "status": "created"}


@router.put("/personas/{persona_id}")
async def update_persona(request: Request, persona_id: str, body: CreatePersonaRequest):
    """Update an existing persona contract."""
    # TODO: Update persona via repository
    return {"id": persona_id, "status": "updated"}
