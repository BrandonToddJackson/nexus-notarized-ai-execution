"""Credential vault API routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from nexus.exceptions import CredentialError, CredentialNotFound
from nexus.types import CredentialType

logger = logging.getLogger(__name__)
router = APIRouter(tags=["credentials"])


# ── Request schemas ───────────────────────────────────────────────────────────

class CredentialCreateRequest(BaseModel):
    name: str
    credential_type: str
    service_name: str
    data: dict
    scoped_personas: list[str] = Field(default_factory=list)


class CredentialTestRequest(BaseModel):
    credential_type: str
    service_name: str
    data: dict


class CredentialRotateRequest(BaseModel):
    data: dict


# ── Dependencies ──────────────────────────────────────────────────────────────

def _get_tenant(request: Request) -> str:
    tenant = getattr(request.state, "tenant_id", None)
    if not tenant:
        raise HTTPException(status_code=401, detail="Tenant not authenticated.")
    return tenant


def _get_vault(request: Request):
    vault = getattr(request.app.state, "vault", None)
    if vault is None:
        raise HTTPException(status_code=503, detail="CredentialVault not initialised.")
    return vault


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/credentials")
async def list_credentials(
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    vault=Depends(_get_vault),
):
    creds = vault.list(tenant_id)
    return {"credentials": [c.model_dump(mode="json") for c in creds]}


@router.post("/credentials", status_code=201)
async def create_credential(
    body: CredentialCreateRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    vault=Depends(_get_vault),
):
    try:
        ctype = CredentialType(body.credential_type)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid credential_type: {body.credential_type}")
    try:
        record = vault.store(
            tenant_id=tenant_id,
            name=body.name,
            credential_type=ctype,
            service_name=body.service_name,
            data=body.data,
            scoped_personas=body.scoped_personas,
        )
    except CredentialError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return record.model_dump(mode="json")


@router.delete("/credentials/{credential_id}")
async def delete_credential(
    credential_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    vault=Depends(_get_vault),
):
    try:
        vault.delete(credential_id, tenant_id)
    except CredentialNotFound:
        raise HTTPException(status_code=404, detail="Credential not found")
    return {"deleted": True}


@router.get("/credentials/types")
async def list_credential_types(request: Request):
    return {
        "types": [
            {"value": ct.value, "label": ct.value.replace("_", " ").title()}
            for ct in CredentialType
        ]
    }


@router.post("/credentials/test")
async def test_credential(
    body: CredentialTestRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
):
    try:
        CredentialType(body.credential_type)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid credential_type: {body.credential_type}")
    return {"valid": True, "message": "Credential format is valid"}


@router.get("/credentials/{credential_id}/usage")
async def credential_usage(
    credential_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    vault=Depends(_get_vault),
):
    try:
        vault.retrieve(credential_id, tenant_id)
    except CredentialNotFound:
        raise HTTPException(status_code=404, detail="Credential not found")
    # Count MCP servers using this credential
    adapter = getattr(request.app.state, "mcp_adapter", None)
    count = 0
    if adapter:
        for server in adapter.list_servers(tenant_id):
            if server.credential_id == credential_id:
                count += 1
    return {"credential_id": credential_id, "usage_count": count}


@router.post("/credentials/{credential_id}/peek")
async def peek_credential(
    credential_id: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    vault=Depends(_get_vault),
):
    try:
        data = vault.retrieve(credential_id, tenant_id)
    except CredentialNotFound:
        raise HTTPException(status_code=404, detail="Credential not found")
    # Return last 4 chars of the first string value as a hint
    hint = ""
    for v in data.values():
        if isinstance(v, str) and len(v) >= 4:
            hint = f"...{v[-4:]}"
            break
    return {"credential_id": credential_id, "hint": hint}


@router.post("/credentials/{credential_id}/rotate")
async def rotate_credential(
    credential_id: str,
    body: CredentialRotateRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    vault=Depends(_get_vault),
):
    try:
        record = vault.update(credential_id, tenant_id, body.data)
    except CredentialNotFound:
        raise HTTPException(status_code=404, detail="Credential not found")
    return record.model_dump(mode="json")


@router.get("/oauth/authorize")
async def oauth_authorize(request: Request):
    raise HTTPException(status_code=501, detail="OAuth authorization not implemented in v1")


@router.get("/oauth/callback")
async def oauth_callback(request: Request):
    raise HTTPException(status_code=501, detail="OAuth callback not implemented in v1")
