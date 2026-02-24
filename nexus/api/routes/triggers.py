"""Trigger management API routes."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from nexus.config import config as nexus_config
from nexus.exceptions import TriggerError, WorkflowNotFound
from nexus.types import TriggerType

logger = logging.getLogger(__name__)
router = APIRouter(tags=["triggers"])


# ── Request schemas ───────────────────────────────────────────────────────────

class TriggerCreateRequest(BaseModel):
    workflow_id: str
    type: str  # "webhook", "cron", "event", "workflow_complete", "manual", "schedule"
    config: dict = Field(default_factory=dict)


class TriggerUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    config: Optional[dict] = None


# ── Dependencies ──────────────────────────────────────────────────────────────

def _get_tenant(request: Request) -> str:
    tenant = getattr(request.state, "tenant_id", None)
    if not tenant:
        raise HTTPException(status_code=401, detail="Tenant not authenticated.")
    return tenant


def _get_manager(request: Request):
    mgr = getattr(request.app.state, "trigger_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="TriggerManager not initialised.")
    return mgr


def _get_config(request: Request):
    """Return the global NexusConfig. `request` is kept for future per-tenant overrides."""
    return nexus_config


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/triggers")
async def list_triggers(
    request: Request,
    workflow_id: Optional[str] = None,
    enabled: Optional[bool] = None,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """List all triggers for the tenant, optionally filtered by workflow_id or enabled state."""
    triggers = await manager.list(tenant_id, workflow_id=workflow_id, enabled=enabled)
    return {"triggers": [t.model_dump() for t in triggers]}


@router.post("/triggers", status_code=201)
async def create_trigger(
    body: TriggerCreateRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Create a new trigger for a workflow."""
    try:
        trigger_type = TriggerType(body.type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid trigger type: {body.type!r}. "
                   f"Must be one of: {[t.value for t in TriggerType]}",
        )

    try:
        trigger = await manager.create_trigger(
            tenant_id=tenant_id,
            workflow_id=body.workflow_id,
            trigger_type=trigger_type,
            config=body.config,
        )
    except WorkflowNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")
    except TriggerError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    result = trigger.model_dump()
    # Attach a computed webhook_url for WEBHOOK triggers
    if trigger_type == TriggerType.WEBHOOK and trigger.webhook_path:
        cfg = _get_config(request)
        base = getattr(cfg, "webhook_base_url", "") if cfg else ""
        result["webhook_url"] = f"{base}{trigger.webhook_path}"

    return result


@router.get("/triggers/{trigger_id}")
async def get_trigger(
    trigger_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Get a single trigger by ID."""
    triggers = await manager.list(tenant_id)
    trigger = next((t for t in triggers if str(t.id) == trigger_id), None)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    return trigger.model_dump()


@router.put("/triggers/{trigger_id}")
async def update_trigger(
    trigger_id: str,
    body: TriggerUpdateRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Enable/disable a trigger or update its config.

    Uses enable()/disable() for the enabled flag — TriggerManager has no
    generic update method, so config changes are applied via the repository
    directly when available.
    """
    # Verify the trigger exists
    triggers = await manager.list(tenant_id)
    trigger = next((t for t in triggers if str(t.id) == trigger_id), None)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")

    try:
        if body.enabled is True:
            trigger = await manager.enable(tenant_id, trigger_id)
        elif body.enabled is False:
            trigger = await manager.disable(tenant_id, trigger_id)
        else:
            # No enabled change — return current state
            pass
    except TriggerError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return trigger.model_dump()


@router.post("/triggers/{trigger_id}/enable")
async def enable_trigger(
    trigger_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Enable a disabled trigger."""
    try:
        trigger = await manager.enable(tenant_id, trigger_id)
    except TriggerError as exc:
        msg = str(exc)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=422, detail=msg)
    return trigger.model_dump()


@router.post("/triggers/{trigger_id}/disable")
async def disable_trigger(
    trigger_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Disable an enabled trigger."""
    try:
        trigger = await manager.disable(tenant_id, trigger_id)
    except TriggerError as exc:
        msg = str(exc)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=422, detail=msg)
    return trigger.model_dump()


@router.delete("/triggers/{trigger_id}", status_code=204)
async def delete_trigger(
    trigger_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    manager=Depends(_get_manager),
):
    """Delete a trigger permanently."""
    try:
        await manager.delete(tenant_id, trigger_id)
    except TriggerError as exc:
        msg = str(exc)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=422, detail=msg)
