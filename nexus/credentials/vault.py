"""CredentialVault — tenant-scoped, encrypted secret storage.

Credentials are NEVER exposed to the AI planner or written to the ledger.
The vault is called by the executor *after* anomaly gates pass and *before*
the tool runs.  The executor sanitizes tool_params before the Notary seals
the record so that secrets never appear in the audit trail.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

from nexus.credentials.encryption import CredentialEncryption
from nexus.exceptions import CredentialError, CredentialNotFound
from nexus.types import CredentialRecord, CredentialType

logger = logging.getLogger(__name__)

# Keys that must be stripped before sealing params into the ledger.
_SENSITIVE_KEYS: frozenset[str] = frozenset({
    "Authorization",
    "authorization",
    "password",
    "token",
    "api_key",
    "access_token",
    "refresh_token",
    "secret",
    "credentials",
    "client_secret",
})


def sanitize_tool_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *params* with sensitive values replaced by ``'***'``.

    Call this before passing tool_params to Notary.seal() so that secrets
    never appear in the immutable audit ledger.
    """
    return {k: "***" if k in _SENSITIVE_KEYS else v for k, v in params.items()}


class CredentialVault:
    """In-process, encrypted credential store scoped to tenants and personas.

    In v1 credentials live in memory.  A future phase will back this with the
    PostgreSQL ``credentials`` table via the async repository.

    Args:
        encryption: A configured :class:`CredentialEncryption` instance.
    """

    def __init__(self, encryption: CredentialEncryption) -> None:
        self._enc = encryption
        self._store: dict[str, CredentialRecord] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def store(
        self,
        *,
        tenant_id: str,
        name: str,
        credential_type: CredentialType,
        service_name: str,
        data: dict[str, Any],
        scoped_personas: Optional[list[str]] = None,
        expires_at: Optional[datetime] = None,
    ) -> CredentialRecord:
        """Encrypt *data* and persist the credential record.

        Returns the stored record **without** ``encrypted_data`` so callers
        can safely log or return the metadata to API consumers.

        Raises:
            CredentialError: if *data* is not a dict.
        """
        if not isinstance(data, dict):
            raise CredentialError("Credential data must be a plain dict")

        encrypted = self._enc.encrypt(json.dumps(data))
        record = CredentialRecord(
            tenant_id=tenant_id,
            name=name,
            credential_type=credential_type,
            service_name=service_name,
            encrypted_data=encrypted,
            scoped_personas=scoped_personas or [],
            expires_at=expires_at,
        )
        self._store[record.id] = record
        logger.debug("[Vault] Stored credential %s (%s) for tenant %s", record.id, service_name, tenant_id)
        return record.model_copy(update={"encrypted_data": ""})

    def retrieve(
        self,
        credential_id: str,
        tenant_id: str,
        persona_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Load, authorise, and decrypt a credential.

        Args:
            credential_id: UUID of the stored credential.
            tenant_id: Calling tenant — must match stored tenant.
            persona_name: Active persona — checked against ``scoped_personas``
                when the list is non-empty.

        Returns:
            Decrypted credential data dict.

        Raises:
            CredentialNotFound: unknown ID or tenant mismatch.
            CredentialError: persona not in scope, credential expired, or
                decryption failure.
        """
        record = self._store.get(credential_id)
        if record is None or record.tenant_id != tenant_id:
            raise CredentialNotFound(
                f"Credential '{credential_id}' not found",
                credential_id=credential_id,
            )

        if record.scoped_personas and persona_name not in record.scoped_personas:
            raise CredentialError(
                f"Persona '{persona_name}' is not permitted to access credential '{credential_id}'",
                credential_id=credential_id,
            )

        if record.expires_at and record.expires_at < datetime.utcnow():
            raise CredentialError(
                f"Credential '{credential_id}' expired at {record.expires_at.isoformat()}",
                credential_id=credential_id,
            )

        return json.loads(self._enc.decrypt(record.encrypted_data))

    def update(
        self,
        credential_id: str,
        tenant_id: str,
        data: dict[str, Any],
    ) -> CredentialRecord:
        """Re-encrypt and overwrite the credential data.

        Returns updated record without ``encrypted_data``.

        Raises:
            CredentialNotFound: unknown ID or tenant mismatch.
        """
        record = self._store.get(credential_id)
        if record is None or record.tenant_id != tenant_id:
            raise CredentialNotFound(
                f"Credential '{credential_id}' not found",
                credential_id=credential_id,
            )

        encrypted = self._enc.encrypt(json.dumps(data))
        updated = record.model_copy(update={"encrypted_data": encrypted, "updated_at": datetime.utcnow()})
        self._store[credential_id] = updated
        return updated.model_copy(update={"encrypted_data": ""})

    def delete(self, credential_id: str, tenant_id: str) -> bool:
        """Remove a credential from the vault.

        Returns:
            ``True`` on success.

        Raises:
            CredentialNotFound: unknown ID or tenant mismatch.
        """
        record = self._store.get(credential_id)
        if record is None or record.tenant_id != tenant_id:
            raise CredentialNotFound(
                f"Credential '{credential_id}' not found",
                credential_id=credential_id,
            )
        del self._store[credential_id]
        logger.debug("[Vault] Deleted credential %s for tenant %s", credential_id, tenant_id)
        return True

    def list(self, tenant_id: str) -> list[CredentialRecord]:
        """Return metadata for all credentials belonging to *tenant_id*.

        ``encrypted_data`` is always empty in the returned records.
        """
        return [
            r.model_copy(update={"encrypted_data": ""})
            for r in self._store.values()
            if r.tenant_id == tenant_id
        ]

    # ------------------------------------------------------------------
    # Injection & OAuth2 refresh
    # ------------------------------------------------------------------

    def inject_credentials(
        self,
        credential_id: str,
        tenant_id: str,
        persona_name: str,
        tool_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge decrypted credential fields into *tool_params*.

        The merge strategy depends on ``credential_type``:

        * ``api_key``      → ``tool_params["api_key"]``
        * ``oauth2``       → ``tool_params["access_token"]``
        * ``basic_auth``   → ``tool_params["username"]``, ``tool_params["password"]``
        * ``bearer_token`` → ``tool_params["token"]``
        * ``custom``       → shallow-merge all keys from the credential data

        Returns a *new* dict; the original *tool_params* is not mutated.

        Raises:
            CredentialNotFound: if the credential does not exist or tenant mismatch.
            CredentialError: if persona scope / expiry checks fail.
        """
        record = self._store.get(credential_id)
        if record is None or record.tenant_id != tenant_id:
            raise CredentialNotFound(
                f"Credential '{credential_id}' not found",
                credential_id=credential_id,
            )

        data = self.retrieve(credential_id, tenant_id, persona_name)
        merged = dict(tool_params)

        ctype = record.credential_type
        if ctype == CredentialType.API_KEY:
            value = data.get("api_key")
            if not value:
                logger.warning("[Vault] Credential '%s' has no 'api_key' field — injecting empty string", credential_id)
            merged["api_key"] = value or ""
        elif ctype == CredentialType.OAUTH2:
            value = data.get("access_token")
            if not value:
                logger.warning("[Vault] Credential '%s' has no 'access_token' field — injecting empty string", credential_id)
            merged["access_token"] = value or ""
        elif ctype == CredentialType.BASIC_AUTH:
            if not data.get("username") or not data.get("password"):
                logger.warning("[Vault] Credential '%s' missing 'username' or 'password' field", credential_id)
            merged["username"] = data.get("username") or ""
            merged["password"] = data.get("password") or ""
        elif ctype == CredentialType.BEARER_TOKEN:
            value = data.get("token")
            if not value:
                logger.warning("[Vault] Credential '%s' has no 'token' field — injecting empty string", credential_id)
            merged["token"] = value or ""
        elif ctype == CredentialType.CUSTOM:
            merged.update(data)

        return merged

    async def refresh_oauth2(
        self,
        credential_id: str,
        tenant_id: str,
        token_url: str,
    ) -> CredentialRecord:
        """Exchange a stored refresh_token for a new access_token.

        POSTs to *token_url* with ``grant_type=refresh_token``, updates the
        stored credential, and returns the updated metadata record (no
        ``encrypted_data``).

        Raises:
            CredentialNotFound: unknown credential.
            CredentialError: HTTP failure or missing ``access_token`` in response.
        """
        data = self.retrieve(credential_id, tenant_id)
        refresh_token = data.get("refresh_token", "")
        if not refresh_token:
            raise CredentialError(
                f"Credential '{credential_id}' has no refresh_token for OAuth2 refresh",
                credential_id=credential_id,
            )

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    token_url,
                    data={"grant_type": "refresh_token", "refresh_token": refresh_token},
                )
                resp.raise_for_status()
                token_data: dict[str, Any] = resp.json()
        except httpx.HTTPStatusError as exc:
            raise CredentialError(
                f"OAuth2 token refresh failed: {exc.response.status_code} {exc.response.text}",
                credential_id=credential_id,
            ) from exc
        except Exception as exc:
            raise CredentialError(
                f"OAuth2 token refresh request error: {exc}",
                credential_id=credential_id,
            ) from exc

        if "access_token" not in token_data:
            raise CredentialError(
                "OAuth2 token response missing 'access_token'",
                credential_id=credential_id,
            )

        data["access_token"] = token_data["access_token"]
        if "refresh_token" in token_data:
            data["refresh_token"] = token_data["refresh_token"]

        updated_record = self.update(credential_id, tenant_id, data)

        # Apply expires_at from expires_in if provided
        if "expires_in" in token_data:
            record = self._store[credential_id]  # guaranteed to exist after update()
            new_expires = datetime.utcnow() + timedelta(seconds=int(token_data["expires_in"]))
            self._store[credential_id] = record.model_copy(update={"expires_at": new_expires})
            return self._store[credential_id].model_copy(update={"encrypted_data": ""})

        return updated_record
