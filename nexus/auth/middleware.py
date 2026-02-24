"""FastAPI authentication middleware.

Checks Authorization header:
- Bearer JWT token
- API key starting with "nxs_"

Extracts tenant_id and sets request.state.tenant_id.
"""

import hashlib
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from nexus.auth.jwt import JWTManager


class AuthMiddleware(BaseHTTPMiddleware):
    """Tenant isolation middleware."""

    # Paths that don't require auth
    PUBLIC_PATHS = {"/v1/health", "/v1/auth/token", "/docs", "/openapi.json", "/redoc"}
    # Webhook prefix — matched by startswith so all /v2/webhooks/* paths bypass JWT
    PUBLIC_PREFIXES = ("/v2/webhooks",)

    def __init__(self, app, jwt_manager: JWTManager = None, repository=None):
        super().__init__(app)
        self.jwt_manager = jwt_manager or JWTManager()
        self.repository = repository

    async def dispatch(self, request: Request, call_next):
        """Check auth and set tenant context.

        1. Skip auth for public paths
        2. Check Authorization header
        3. If Bearer token: verify JWT, extract tenant_id
        4. If API key (starts with "nxs_"): look up tenant by key hash
        5. Set request.state.tenant_id
        6. Return 401 if invalid

        Args:
            request: FastAPI request
            call_next: Next middleware/handler
        """
        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)
        if any(request.url.path.startswith(p) for p in self.PUBLIC_PREFIXES):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")

        if not auth_header:
            return JSONResponse({"detail": "Missing Authorization header"}, status_code=401)

        try:
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                payload = await self.jwt_manager.verify_token(token)
                request.state.tenant_id = payload["tenant_id"]
                request.state.role = payload["role"]
            elif auth_header.startswith("nxs_"):
                # API key auth — hash and look up
                key_hash = hashlib.sha256(auth_header.encode()).hexdigest()
                tenant = None
                if self.repository is not None:
                    tenant = await self.repository.get_tenant_by_api_key_hash(key_hash)
                else:
                    # Per-request DB lookup via app.state.async_session
                    async_session = getattr(request.app.state, "async_session", None)
                    if async_session is None:
                        return JSONResponse({"detail": "Repository not configured"}, status_code=401)
                    from nexus.db.repository import Repository
                    async with async_session() as session:
                        repo = Repository(session)
                        tenant = await repo.get_tenant_by_api_key_hash(key_hash)
                if tenant is None:
                    return JSONResponse({"detail": "Invalid API key"}, status_code=401)
                request.state.tenant_id = str(tenant.id)
                request.state.role = "user"
            else:
                return JSONResponse({"detail": "Invalid auth format"}, status_code=401)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[AuthMiddleware] Auth error: {e}")
            return JSONResponse({"detail": "Authentication failed"}, status_code=401)

        return await call_next(request)
