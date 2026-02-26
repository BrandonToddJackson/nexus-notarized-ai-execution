"""Phase 18 — Credential Vault tests.

Covers:
- CredentialEncryption: generate key, custom key, encrypt/decrypt, tamper detection
- CredentialVault: store, retrieve, update, delete, list, inject_credentials, refresh_oauth2
- sanitize_tool_params: sensitive key stripping
- ToolExecutor: vault integration (inject on credential_id, skip when no vault)
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from cryptography.fernet import Fernet

from nexus.credentials.encryption import CredentialEncryption
from nexus.credentials.vault import CredentialVault, sanitize_tool_params
from nexus.exceptions import CredentialError, CredentialNotFound
from nexus.types import CredentialType


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fernet_key() -> str:
    return Fernet.generate_key().decode()


@pytest.fixture
def enc(fernet_key) -> CredentialEncryption:
    return CredentialEncryption(key=fernet_key)


@pytest.fixture
def vault(enc) -> CredentialVault:
    return CredentialVault(encryption=enc)


def _store_api_key(vault: CredentialVault, tenant: str = "t1", persona: str = "") -> str:
    """Helper — store an api_key credential and return its id."""
    scoped = [persona] if persona else []
    rec = vault.store(
        tenant_id=tenant,
        name="Test API Key",
        credential_type=CredentialType.API_KEY,
        service_name="testservice",
        data={"api_key": "sk-secret-123"},
        scoped_personas=scoped,
    )
    return rec.id


# ── CredentialEncryption ──────────────────────────────────────────────────────

class TestCredentialEncryption:
    def test_encrypt_decrypt_roundtrip(self, enc):
        plaintext = '{"api_key": "sk-test"}'
        ciphertext = enc.encrypt(plaintext)
        assert ciphertext != plaintext
        assert enc.decrypt(ciphertext) == plaintext

    def test_ciphertext_is_string(self, enc):
        result = enc.encrypt("hello")
        assert isinstance(result, str)

    def test_different_encryptions_produce_different_ciphertext(self, enc):
        # Fernet uses random IV per call
        c1 = enc.encrypt("same")
        c2 = enc.encrypt("same")
        assert c1 != c2

    def test_tampered_ciphertext_raises_credential_error(self, enc):
        ciphertext = enc.encrypt("data")
        tampered = ciphertext[:-4] + "XXXX"
        with pytest.raises(CredentialError):
            enc.decrypt(tampered)

    def test_empty_key_generates_ephemeral_key(self):
        # Should not raise; should log warning
        enc = CredentialEncryption(key="")
        result = enc.encrypt("test")
        assert enc.decrypt(result) == "test"

    def test_invalid_key_raises_credential_error(self):
        with pytest.raises(CredentialError, match="Invalid Fernet key"):
            CredentialEncryption(key="not-a-valid-key")

    def test_different_key_cannot_decrypt(self, fernet_key):
        enc1 = CredentialEncryption(key=fernet_key)
        enc2 = CredentialEncryption(key=Fernet.generate_key().decode())
        ciphertext = enc1.encrypt("secret")
        with pytest.raises(CredentialError):
            enc2.decrypt(ciphertext)


# ── CredentialVault.store ─────────────────────────────────────────────────────

class TestVaultStore:
    def test_store_returns_record_without_encrypted_data(self, vault):
        rec = vault.store(
            tenant_id="t1",
            name="My Key",
            credential_type=CredentialType.API_KEY,
            service_name="github",
            data={"api_key": "ghp_abc"},
        )
        assert rec.id
        assert rec.tenant_id == "t1"
        assert rec.name == "My Key"
        assert rec.encrypted_data == ""

    def test_store_persists_internally(self, vault):
        rec = vault.store(
            tenant_id="t1",
            name="Stored",
            credential_type=CredentialType.BEARER_TOKEN,
            service_name="stripe",
            data={"token": "sk_live_xxx"},
        )
        # Internal store has encrypted data
        internal = vault._store[rec.id]
        assert internal.encrypted_data != ""

    def test_store_non_dict_data_raises(self, vault):
        with pytest.raises(CredentialError):
            vault.store(
                tenant_id="t1",
                name="Bad",
                credential_type=CredentialType.API_KEY,
                service_name="x",
                data="not-a-dict",  # type: ignore[arg-type]
            )

    def test_store_with_scoped_personas(self, vault):
        rec = vault.store(
            tenant_id="t1",
            name="Scoped",
            credential_type=CredentialType.API_KEY,
            service_name="x",
            data={"api_key": "k"},
            scoped_personas=["researcher"],
        )
        internal = vault._store[rec.id]
        assert internal.scoped_personas == ["researcher"]

    def test_store_with_expiry(self, vault):
        expires = datetime.now(tz=timezone.utc) + timedelta(hours=1)
        rec = vault.store(
            tenant_id="t1",
            name="Expiring",
            credential_type=CredentialType.API_KEY,
            service_name="x",
            data={"api_key": "k"},
            expires_at=expires,
        )
        internal = vault._store[rec.id]
        assert internal.expires_at == expires

    def test_store_generates_unique_ids(self, vault):
        ids = set()
        for _ in range(5):
            rec = vault.store(
                tenant_id="t1", name="k", credential_type=CredentialType.API_KEY,
                service_name="x", data={"api_key": "v"},
            )
            ids.add(rec.id)
        assert len(ids) == 5


# ── CredentialVault.retrieve ──────────────────────────────────────────────────

class TestVaultRetrieve:
    def test_retrieve_decrypts_data(self, vault):
        rec = vault.store(
            tenant_id="t1", name="K", credential_type=CredentialType.API_KEY,
            service_name="svc", data={"api_key": "my-secret"},
        )
        data = vault.retrieve(rec.id, "t1")
        assert data == {"api_key": "my-secret"}

    def test_retrieve_wrong_tenant_raises_not_found(self, vault):
        cid = _store_api_key(vault, "t1")
        with pytest.raises(CredentialNotFound):
            vault.retrieve(cid, "t2")

    def test_retrieve_unknown_id_raises_not_found(self, vault):
        with pytest.raises(CredentialNotFound):
            vault.retrieve("nonexistent-id", "t1")

    def test_retrieve_expired_raises_credential_error(self, vault):
        expires = datetime.now(tz=timezone.utc) - timedelta(seconds=1)
        rec = vault.store(
            tenant_id="t1", name="Expired", credential_type=CredentialType.API_KEY,
            service_name="x", data={"api_key": "k"}, expires_at=expires,
        )
        with pytest.raises(CredentialError, match="expired"):
            vault.retrieve(rec.id, "t1")

    def test_retrieve_wrong_persona_raises_credential_error(self, vault):
        cid = _store_api_key(vault, "t1", persona="researcher")
        with pytest.raises(CredentialError, match="not permitted"):
            vault.retrieve(cid, "t1", persona_name="executor")

    def test_retrieve_correct_persona_succeeds(self, vault):
        cid = _store_api_key(vault, "t1", persona="researcher")
        data = vault.retrieve(cid, "t1", persona_name="researcher")
        assert data["api_key"] == "sk-secret-123"

    def test_retrieve_unscoped_allows_any_persona(self, vault):
        cid = _store_api_key(vault, "t1", persona="")  # no scoping
        data = vault.retrieve(cid, "t1", persona_name="any-persona")
        assert data["api_key"] == "sk-secret-123"


# ── CredentialVault.update / delete / list ────────────────────────────────────

class TestVaultCRUD:
    def test_update_replaces_data(self, vault):
        cid = _store_api_key(vault)
        vault.update(cid, "t1", {"api_key": "new-secret"})
        data = vault.retrieve(cid, "t1")
        assert data == {"api_key": "new-secret"}

    def test_update_returns_record_without_encrypted_data(self, vault):
        cid = _store_api_key(vault)
        rec = vault.update(cid, "t1", {"api_key": "x"})
        assert rec.encrypted_data == ""

    def test_update_wrong_tenant_raises(self, vault):
        cid = _store_api_key(vault, "t1")
        with pytest.raises(CredentialNotFound):
            vault.update(cid, "t2", {"api_key": "x"})

    def test_delete_removes_record(self, vault):
        cid = _store_api_key(vault)
        assert vault.delete(cid, "t1") is True
        with pytest.raises(CredentialNotFound):
            vault.retrieve(cid, "t1")

    def test_delete_wrong_tenant_raises(self, vault):
        cid = _store_api_key(vault, "t1")
        with pytest.raises(CredentialNotFound):
            vault.delete(cid, "t2")

    def test_list_returns_tenant_records_only(self, vault):
        vault.store(tenant_id="t1", name="A", credential_type=CredentialType.API_KEY, service_name="x", data={"api_key": "1"})
        vault.store(tenant_id="t1", name="B", credential_type=CredentialType.BEARER_TOKEN, service_name="y", data={"token": "2"})
        vault.store(tenant_id="t2", name="C", credential_type=CredentialType.API_KEY, service_name="z", data={"api_key": "3"})

        records = vault.list("t1")
        assert len(records) == 2
        assert all(r.tenant_id == "t1" for r in records)

    def test_list_returns_no_encrypted_data(self, vault):
        vault.store(tenant_id="t1", name="A", credential_type=CredentialType.API_KEY, service_name="x", data={"api_key": "1"})
        records = vault.list("t1")
        assert all(r.encrypted_data == "" for r in records)

    def test_list_empty_for_unknown_tenant(self, vault):
        assert vault.list("nobody") == []


# ── CredentialVault.inject_credentials ───────────────────────────────────────

class TestInjectCredentials:
    def _store(self, vault, ctype, data, tenant="t1"):
        rec = vault.store(
            tenant_id=tenant, name="x", credential_type=ctype,
            service_name="svc", data=data,
        )
        return rec.id

    def test_inject_api_key(self, vault):
        cid = self._store(vault, CredentialType.API_KEY, {"api_key": "sk-abc"})
        merged = vault.inject_credentials(cid, "t1", "researcher", {"query": "hello"})
        assert merged["api_key"] == "sk-abc"
        assert merged["query"] == "hello"

    def test_inject_oauth2(self, vault):
        cid = self._store(vault, CredentialType.OAUTH2, {"access_token": "tok-xyz"})
        merged = vault.inject_credentials(cid, "t1", "researcher", {})
        assert merged["access_token"] == "tok-xyz"

    def test_inject_basic_auth(self, vault):
        cid = self._store(vault, CredentialType.BASIC_AUTH, {"username": "user", "password": "pass"})
        merged = vault.inject_credentials(cid, "t1", "researcher", {})
        assert merged["username"] == "user"
        assert merged["password"] == "pass"

    def test_inject_bearer_token(self, vault):
        cid = self._store(vault, CredentialType.BEARER_TOKEN, {"token": "Bearer abc"})
        merged = vault.inject_credentials(cid, "t1", "researcher", {})
        assert merged["token"] == "Bearer abc"

    def test_inject_custom_merges_all(self, vault):
        cid = self._store(vault, CredentialType.CUSTOM, {"x_custom": "val", "y_extra": 42})
        merged = vault.inject_credentials(cid, "t1", "researcher", {"existing": True})
        assert merged["x_custom"] == "val"
        assert merged["y_extra"] == 42
        assert merged["existing"] is True

    def test_inject_does_not_mutate_original(self, vault):
        cid = self._store(vault, CredentialType.API_KEY, {"api_key": "k"})
        original = {"query": "q"}
        merged = vault.inject_credentials(cid, "t1", "researcher", original)
        assert "api_key" not in original
        assert "api_key" in merged

    def test_inject_wrong_tenant_raises(self, vault):
        cid = self._store(vault, CredentialType.API_KEY, {"api_key": "k"}, tenant="t1")
        with pytest.raises(CredentialNotFound):
            vault.inject_credentials(cid, "t2", "researcher", {})


# ── CredentialVault.refresh_oauth2 ────────────────────────────────────────────

class TestRefreshOAuth2:
    @pytest.mark.asyncio
    async def test_refresh_updates_access_token(self, vault):
        rec = vault.store(
            tenant_id="t1", name="OAuth", credential_type=CredentialType.OAUTH2,
            service_name="google",
            data={"access_token": "old-token", "refresh_token": "refresh-xyz"},
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "new-token", "expires_in": 3600}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            updated = await vault.refresh_oauth2(rec.id, "t1", "https://oauth.example.com/token")

        data = vault.retrieve(rec.id, "t1")
        assert data["access_token"] == "new-token"
        assert updated.encrypted_data == ""

    @pytest.mark.asyncio
    async def test_refresh_no_refresh_token_raises(self, vault):
        rec = vault.store(
            tenant_id="t1", name="OAuth", credential_type=CredentialType.OAUTH2,
            service_name="svc", data={"access_token": "tok"},  # no refresh_token
        )
        with pytest.raises(CredentialError, match="no refresh_token"):
            await vault.refresh_oauth2(rec.id, "t1", "https://example.com/token")

    @pytest.mark.asyncio
    async def test_refresh_http_error_raises_credential_error(self, vault):
        import httpx as _httpx
        rec = vault.store(
            tenant_id="t1", name="OAuth", credential_type=CredentialType.OAUTH2,
            service_name="svc", data={"access_token": "tok", "refresh_token": "rft"},
        )
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            request = _httpx.Request("POST", "https://example.com/token")
            response = _httpx.Response(401, request=request)
            mock_client.post = AsyncMock(side_effect=_httpx.HTTPStatusError("401", request=request, response=response))
            mock_client_cls.return_value = mock_client

            with pytest.raises(CredentialError, match="OAuth2 token refresh failed"):
                await vault.refresh_oauth2(rec.id, "t1", "https://example.com/token")


# ── sanitize_tool_params ──────────────────────────────────────────────────────

class TestSanitizeToolParams:
    def test_strips_sensitive_keys(self):
        params = {
            "query": "hello",
            "api_key": "sk-secret",
            "password": "hunter2",
            "token": "tok-xyz",
            "access_token": "at-abc",
            "Authorization": "Bearer xyz",
            "refresh_token": "rt-123",
            "secret": "shh",
            "credentials": "creds",
            "client_secret": "cs-999",
        }
        sanitized = sanitize_tool_params(params)
        assert sanitized["query"] == "hello"
        for key in ("api_key", "password", "token", "access_token", "Authorization",
                    "refresh_token", "secret", "credentials", "client_secret"):
            assert sanitized[key] == "***", f"Expected {key} to be redacted"

    def test_non_sensitive_keys_unchanged(self):
        params = {"url": "https://example.com", "method": "GET", "timeout": 30}
        sanitized = sanitize_tool_params(params)
        assert sanitized == params

    def test_returns_new_dict(self):
        params = {"api_key": "k"}
        sanitized = sanitize_tool_params(params)
        assert sanitized is not params
        assert params["api_key"] == "k"  # original unchanged

    def test_empty_params(self):
        assert sanitize_tool_params({}) == {}


# ── ToolExecutor vault integration ────────────────────────────────────────────

class TestToolExecutorVaultIntegration:
    """Smoke test: executor injects credentials when credential_id is present."""

    def _make_executor(self, vault=None):
        from nexus.tools.executor import ToolExecutor
        from nexus.tools.registry import ToolRegistry
        from nexus.tools.sandbox import Sandbox
        from nexus.core.verifier import IntentVerifier

        registry = MagicMock(spec=ToolRegistry)
        sandbox = MagicMock(spec=Sandbox)
        verifier = MagicMock(spec=IntentVerifier)

        # Tool lookup succeeds
        tool_def = MagicMock()
        tool_def.timeout_seconds = 10
        async def dummy_tool(**kwargs):
            return {"called_with": kwargs}
        registry.get.return_value = (tool_def, dummy_tool)

        # Verifier passes
        verifier.verify.return_value = None

        # Sandbox returns tool result
        sandbox.execute = AsyncMock(return_value={"ok": True})

        return ToolExecutor(registry=registry, sandbox=sandbox, verifier=verifier, vault=vault)

    def _make_intent(self, tool_params):
        from nexus.types import IntentDeclaration
        return IntentDeclaration(
            task_description="test task",
            planned_action="run tool",
            tool_name="dummy_tool",
            tool_params=tool_params,
            resource_targets=[],
            reasoning="test",
        )

    @pytest.mark.asyncio
    async def test_executor_injects_api_key_from_vault(self, vault):
        cid = _store_api_key(vault)
        executor = self._make_executor(vault=vault)
        # tenant_id/persona_name are now explicit execute() params, not in tool_params
        intent = self._make_intent({"query": "hello", "credential_id": cid})

        result, error = await executor.execute(intent, tenant_id="t1", persona_name="researcher")

        assert error is None
        # sandbox.execute was called — check params passed to it
        call_args = executor.sandbox.execute.call_args
        params_passed = call_args[0][1]  # positional arg[1] = params dict
        assert "api_key" in params_passed
        assert params_passed["api_key"] == "sk-secret-123"
        assert "credential_id" not in params_passed  # stripped

    @pytest.mark.asyncio
    async def test_executor_without_vault_skips_injection(self, vault):
        cid = _store_api_key(vault)
        executor = self._make_executor(vault=None)
        intent = self._make_intent({"query": "hello", "credential_id": cid})

        result, error = await executor.execute(intent, tenant_id="t1")

        # No vault → credential_id is still popped but no injection attempted
        assert error is None

    @pytest.mark.asyncio
    async def test_executor_no_credential_id_skips_vault(self, vault):
        executor = self._make_executor(vault=vault)
        intent = self._make_intent({"query": "hello"})
        result, error = await executor.execute(intent)
        assert error is None
        # vault was never touched

    @pytest.mark.asyncio
    async def test_executor_wrong_tenant_fails_gracefully(self, vault):
        """Credential injection with wrong tenant returns error, not exception."""
        cid = _store_api_key(vault, tenant="t1")
        executor = self._make_executor(vault=vault)
        intent = self._make_intent({"query": "hello", "credential_id": cid})

        result, error = await executor.execute(intent, tenant_id="t2")

        assert result is None
        assert error is not None
        assert "Credential injection failed" in error
