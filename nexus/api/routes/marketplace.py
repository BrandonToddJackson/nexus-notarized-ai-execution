"""Plugin Marketplace API routes."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from nexus.marketplace.plugin_sdk import (
    PluginError,
    PluginInstallError,
    PluginImportError,
    PluginManifestError,
    PluginNotFoundError,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["marketplace"])


# ── Request schemas ───────────────────────────────────────────────────────────

class InstallRequest(BaseModel):
    package_name: str = Field(..., min_length=1)
    version: Optional[str] = None
    force: bool = False


# ── Dependencies ──────────────────────────────────────────────────────────────

def _get_tenant(request: Request) -> str:
    tenant = getattr(request.state, "tenant_id", None)
    if not tenant:
        raise HTTPException(status_code=401, detail="Tenant not authenticated.")
    return tenant


def _get_registry(request: Request):
    registry = getattr(request.app.state, "plugin_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="PluginRegistry not initialised.")
    return registry


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/marketplace/search")
async def search_plugins(
    request: Request,
    q: str = "",
    limit: int = 20,
    tenant_id: str = Depends(_get_tenant),
    registry=Depends(_get_registry),
):
    """Search PyPI for NEXUS plugins matching a query."""
    results = await registry.search(query=q, limit=limit)
    return {"results": results, "query": q, "count": len(results)}


@router.get("/marketplace/installed")
async def list_installed(
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    registry=Depends(_get_registry),
):
    """List all installed plugins."""
    plugins = registry.list_installed()
    return {
        "plugins": [p.model_dump() for p in plugins],
        "count": len(plugins),
    }


@router.post("/marketplace/install", status_code=201)
async def install_plugin(
    body: InstallRequest,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    registry=Depends(_get_registry),
):
    """Install a plugin from PyPI."""
    try:
        manifest = await registry.install(
            package_name=body.package_name,
            version=body.version,
            force=body.force,
        )
    except PluginInstallError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except PluginManifestError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except PluginImportError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except PluginError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return manifest.model_dump()


@router.delete("/marketplace/{plugin_name}", status_code=204)
async def uninstall_plugin(
    plugin_name: str,
    request: Request,
    tenant_id: str = Depends(_get_tenant),
    registry=Depends(_get_registry),
):
    """Uninstall a plugin by name."""
    try:
        await registry.uninstall(plugin_name)
    except PluginNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PluginError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
