"""JWT token management: create, validate, refresh."""

import jwt
from datetime import datetime, timedelta, timezone

from nexus.config import config
from nexus.exceptions import NexusError


class JWTManager:
    """JWT token management."""

    async def create_token(self, tenant_id: str, role: str = "user") -> str:
        """Create a JWT token.

        Args:
            tenant_id: Tenant to encode in token
            role: User role ("user" or "admin")

        Returns:
            JWT token string
        """
        payload = {
            "tenant_id": tenant_id,
            "role": role,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=config.jwt_expiry_minutes),
            "iat": datetime.now(timezone.utc),
        }
        return jwt.encode(payload, config.secret_key, algorithm=config.jwt_algorithm)

    async def verify_token(self, token: str) -> dict:
        """Verify and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            {"tenant_id": str, "role": str}

        Raises:
            NexusError: On invalid/expired token
        """
        try:
            payload = jwt.decode(token, config.secret_key, algorithms=[config.jwt_algorithm])
            return {"tenant_id": payload["tenant_id"], "role": payload["role"]}
        except jwt.ExpiredSignatureError:
            raise NexusError("Token expired")
        except jwt.InvalidTokenError:
            raise NexusError("Invalid token")
