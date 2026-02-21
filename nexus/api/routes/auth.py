"""POST /v1/auth/token — Generate JWT from API key."""

import hashlib
from fastapi import APIRouter, HTTPException
from nexus.api.schemas import TokenRequest, TokenResponse
from nexus.auth.jwt import JWTManager
from nexus.config import config

router = APIRouter(tags=["auth"])
jwt_manager = JWTManager()


@router.post("/auth/token", response_model=TokenResponse)
async def create_token(body: TokenRequest):
    """Exchange API key for JWT token.

    Args:
        body: TokenRequest with api_key

    Returns:
        TokenResponse with JWT token
    """
    # TODO: Look up tenant by API key hash
    # For demo: accept "nxs_demo_key_12345"
    if body.api_key == "nxs_demo_key_12345":
        token = await jwt_manager.create_token("demo", "admin")
        return TokenResponse(
            token=token,
            tenant_id="demo",
            expires_in=config.jwt_expiry_minutes * 60,
        )
    raise HTTPException(status_code=401, detail="Invalid API key")
