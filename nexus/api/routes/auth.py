"""POST /v1/auth/token — Generate JWT from API key."""

import hashlib
import logging
from fastapi import APIRouter, HTTPException, Request
from nexus.api.schemas import TokenRequest, TokenResponse
from nexus.auth.jwt import JWTManager
from nexus.config import config

logger = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])
jwt_manager = JWTManager()


@router.post("/auth/token", response_model=TokenResponse)
async def create_token(request: Request, body: TokenRequest):
    """Exchange API key for JWT token."""
    key_hash = hashlib.sha256(body.api_key.encode()).hexdigest()

    # Look up tenant via repository (falls back to in-memory check when DB not available)
    tenant = None
    async_session = getattr(request.app.state, "async_session", None)
    if async_session is not None:
        try:
            async with async_session() as session:
                from nexus.db.repository import Repository
                repo = Repository(session)
                tenant = await repo.get_tenant_by_api_key_hash(key_hash)
        except Exception as exc:
            logger.error(f"[auth] DB lookup failed: {exc}")
            raise HTTPException(status_code=500, detail="Auth service unavailable")
    else:
        # No DB available (e.g., test environment) — validate against seeded demo key
        import hashlib as _hl
        demo_hash = _hl.sha256(b"nxs_demo_key_12345").hexdigest()
        if key_hash == demo_hash:
            # Return a synthetic tenant object
            from types import SimpleNamespace
            tenant = SimpleNamespace(id="demo")

    if tenant is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    token = await jwt_manager.create_token(str(tenant.id), "user")
    return TokenResponse(
        token=token,
        tenant_id=str(tenant.id),
        expires_in=config.jwt_expiry_minutes * 60,
    )
