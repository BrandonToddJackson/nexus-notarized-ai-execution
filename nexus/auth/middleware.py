"""FastAPI authentication middleware.

Checks Authorization header:
- Bearer JWT token
- API key starting with "nxs_"

Extracts tenant_id and sets request.state.tenant_id.
"""

import hashlib
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from nexus.auth.jwt import JWTManager
from nexus.config import config


class AuthMiddleware(BaseHTTPMiddleware):
    """Tenant isolation middleware."""

    # Paths that don't require auth
    PUBLIC_PATHS = {"/v1/health", "/v1/auth/token", "/docs", "/openapi.json", "/redoc"}

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

        # TODO: Implement auth check
        # For now, pass through (implement after API routes are built)
        auth_header = request.headers.get("Authorization", "")

        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")

        try:
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                payload = await self.jwt_manager.verify_token(token)
                request.state.tenant_id = payload["tenant_id"]
                request.state.role = payload["role"]
            elif auth_header.startswith("nxs_"):
                # API key auth — hash and look up
                key_hash = hashlib.sha256(auth_header.encode()).hexdigest()
                # TODO: Look up tenant by key_hash in repository
                request.state.tenant_id = "demo"  # placeholder
                request.state.role = "user"
            else:
                raise HTTPException(status_code=401, detail="Invalid auth format")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))

        return await call_next(request)
