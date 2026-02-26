"""Skill CRUD, import/export, invocation history API routes."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from nexus.exceptions import SkillNotFound, SkillValidationError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["skills"])


# ── Request / Response schemas ────────────────────────────────────────────────

class SkillCreateRequest(BaseModel):
    name: str
    display_name: str
    description: str = ""
    content: str
    allowed_tools: list[str] = Field(default_factory=list)
    allowed_personas: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class SkillUpdateRequest(BaseModel):
    change_note: str
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    allowed_tools: Optional[list[str]] = None
    allowed_personas: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    active: Optional[bool] = None


class SkillImportRequest(BaseModel):
    data: str  # JSON string or SKILL.md with YAML frontmatter


# ── Dependencies ──────────────────────────────────────────────────────────────

def _get_tenant(request: Request) -> str:
    tenant = getattr(request.state, "tenant_id", None)
    if not tenant:
        raise HTTPException(status_code=401, detail="Tenant not authenticated.")
    return tenant


def _get_skill_manager(request: Request):
    mgr = getattr(request.app.state, "skill_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="SkillManager not initialised.")
    return mgr


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/skills")
async def list_skills(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    active: Optional[bool] = None,
    tag: Optional[str] = None,
    search: Optional[str] = None,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_skill_manager),
):
    active_only = active if active is not None else False
    skills = await manager.list(tenant_id, active_only=active_only, limit=limit, offset=offset)
    if tag:
        skills = [s for s in skills if tag in s.tags]
    if search:
        search_lower = search.lower()
        skills = [
            s for s in skills
            if search_lower in s.name.lower()
            or search_lower in s.display_name.lower()
            or search_lower in s.description.lower()
        ]
    return {"skills": [s.model_dump(mode="json") for s in skills], "total": len(skills)}


@router.post("/skills", status_code=201)
async def create_skill(
    body: SkillCreateRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_skill_manager),
):
    try:
        skill = await manager.create(
            tenant_id=tenant_id,
            name=body.name,
            display_name=body.display_name,
            description=body.description,
            content=body.content,
            allowed_tools=body.allowed_tools,
            allowed_personas=body.allowed_personas,
            tags=body.tags,
        )
    except SkillValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return skill.model_dump(mode="json")


@router.get("/skills/{skill_id}")
async def get_skill(
    skill_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_skill_manager),
):
    try:
        skill = await manager.get(skill_id, tenant_id)
    except SkillNotFound:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill.model_dump(mode="json")


@router.patch("/skills/{skill_id}")
async def update_skill(
    skill_id: str,
    body: SkillUpdateRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_skill_manager),
):
    try:
        skill = await manager.update(
            skill_id=skill_id,
            tenant_id=tenant_id,
            change_note=body.change_note,
            name=body.name,
            display_name=body.display_name,
            description=body.description,
            content=body.content,
            allowed_tools=body.allowed_tools,
            allowed_personas=body.allowed_personas,
            tags=body.tags,
            active=body.active,
        )
    except SkillNotFound:
        raise HTTPException(status_code=404, detail="Skill not found")
    except SkillValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return skill.model_dump(mode="json")


@router.delete("/skills/{skill_id}")
async def delete_skill(
    skill_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_skill_manager),
):
    try:
        await manager.delete(skill_id, tenant_id)
    except SkillNotFound:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"deleted": True}


@router.get("/skills/{skill_id}/export")
async def export_skill(
    skill_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_skill_manager),
):
    try:
        skill = await manager.get(skill_id, tenant_id)
    except SkillNotFound:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"data": manager.export_json(skill)}


@router.post("/skills/import", status_code=201)
async def import_skill(
    body: SkillImportRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_skill_manager),
):
    try:
        skill = await manager.import_json(body.data, tenant_id)
    except (SkillValidationError, ValueError, Exception) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return skill.model_dump(mode="json")


@router.get("/skills/{skill_id}/invocations")
async def list_invocations(
    skill_id: str,
    request: Request,
    limit: int = 50,
    offset: int = 0,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_skill_manager),
):
    try:
        await manager.get(skill_id, tenant_id)
    except SkillNotFound:
        raise HTTPException(status_code=404, detail="Skill not found")
    if manager._repository:
        invocations = await manager._repository.list_skill_invocations(
            skill_id, tenant_id, limit=limit, offset=offset
        )
    else:
        invocations = []
    return {"invocations": [i.model_dump(mode="json") for i in invocations], "total": len(invocations)}


@router.post("/skills/{skill_id}/duplicate", status_code=201)
async def duplicate_skill(
    skill_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_skill_manager),
):
    try:
        skill = await manager.duplicate(skill_id, tenant_id)
    except SkillNotFound:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill.model_dump(mode="json")


@router.get("/skills/{skill_id}/diff")
async def diff_versions(
    skill_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_skill_manager),
    from_version: int = 1,
    to_version: Optional[int] = None,
):
    """Diff two versions of a skill. Uses query params ?from=v&to=v."""
    try:
        skill = await manager.get(skill_id, tenant_id)
    except SkillNotFound:
        raise HTTPException(status_code=404, detail="Skill not found")

    history = skill.version_history
    from_entry = next((v for v in history if v.version == from_version), None)
    to_content = skill.content
    to_ver = skill.version
    if to_version is not None:
        to_entry = next((v for v in history if v.version == to_version), None)
        if to_entry is None:
            raise HTTPException(status_code=404, detail=f"Version {to_version} not found")
        to_content = to_entry.content
        to_ver = to_entry.version

    from_content = from_entry.content if from_entry else ""
    return {
        "skill_id": skill_id,
        "from_version": from_version,
        "to_version": to_ver,
        "from_content": from_content,
        "to_content": to_content,
    }
