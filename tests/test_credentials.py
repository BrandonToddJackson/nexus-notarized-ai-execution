"""Phase 30 — Credential vault and encryption tests."""

import pytest

from nexus.credentials.encryption import CredentialEncryption
from nexus.credentials.vault import CredentialVault, sanitize_tool_params
from nexus.exceptions import CredentialError, CredentialNotFound
from nexus.types import CredentialType

TENANT_A = "tenant-alpha-001"
TENANT_B = "tenant-beta-002"


# ── Tests ────────────────────────────────────────────────────────────────────


def test_store_and_retrieve(vault):
    """Store an API_KEY credential and retrieve the decrypted data."""
    record = vault.store(
        tenant_id=TENANT_A,
        name="stripe",
        credential_type=CredentialType.API_KEY,
        service_name="stripe",
        data={"api_key": "sk_test_abc"},
    )
    assert record.encrypted_data == ""  # metadata only — no leak
    data = vault.retrieve(record.id, tenant_id=TENANT_A)
    assert data == {"api_key": "sk_test_abc"}


def test_tenant_isolation(vault):
    """Retrieve under a different tenant raises CredentialNotFound."""
    record = vault.store(
        tenant_id=TENANT_A,
        name="stripe",
        credential_type=CredentialType.API_KEY,
        service_name="stripe",
        data={"api_key": "sk_test_abc"},
    )
    with pytest.raises(CredentialNotFound):
        vault.retrieve(record.id, tenant_id=TENANT_B)


def test_encrypt_decrypt_roundtrip(encryption):
    """Encrypt then decrypt returns the original plaintext."""
    ciphertext = encryption.encrypt("hello")
    assert encryption.decrypt(ciphertext) == "hello"


def test_invalid_key_raises():
    """Constructing CredentialEncryption with a bad key raises CredentialError."""
    with pytest.raises(CredentialError):
        CredentialEncryption(key="bad-key")


def test_delete_removes_cred(vault):
    """Deleted credential cannot be retrieved."""
    record = vault.store(
        tenant_id=TENANT_A,
        name="temp",
        credential_type=CredentialType.API_KEY,
        service_name="temp",
        data={"api_key": "key123"},
    )
    vault.delete(record.id, tenant_id=TENANT_A)
    with pytest.raises(CredentialNotFound):
        vault.retrieve(record.id, tenant_id=TENANT_A)


def test_list_tenant_scope(vault):
    """list() returns only credentials for the requested tenant."""
    vault.store(tenant_id=TENANT_A, name="a1", credential_type=CredentialType.API_KEY, service_name="s1", data={"api_key": "k1"})
    vault.store(tenant_id=TENANT_A, name="a2", credential_type=CredentialType.API_KEY, service_name="s2", data={"api_key": "k2"})
    vault.store(tenant_id=TENANT_B, name="b1", credential_type=CredentialType.API_KEY, service_name="s3", data={"api_key": "k3"})

    result = vault.list(TENANT_A)
    assert len(result) == 2
    assert all(r.encrypted_data == "" for r in result)


def test_inject_api_key(vault):
    """inject_credentials merges the api_key into tool_params."""
    record = vault.store(
        tenant_id=TENANT_A,
        name="stripe",
        credential_type=CredentialType.API_KEY,
        service_name="stripe",
        data={"api_key": "sk_live_real"},
    )
    params = vault.inject_credentials(
        credential_id=record.id,
        tenant_id=TENANT_A,
        persona_name="researcher",
        tool_params={"amount": 100},
    )
    assert params["api_key"] == "sk_live_real"
    assert params["amount"] == 100


def test_sanitize_params():
    """sanitize_tool_params replaces sensitive keys with '***'."""
    sanitized = sanitize_tool_params({"api_key": "sk_real", "query": "hello"})
    assert sanitized["api_key"] == "***"
    assert sanitized["query"] == "hello"
