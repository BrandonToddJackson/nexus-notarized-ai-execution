"""Fernet-based symmetric encryption for credential data at rest."""

import logging

from cryptography.fernet import Fernet, InvalidToken

from nexus.exceptions import CredentialError

logger = logging.getLogger(__name__)


class CredentialEncryption:
    """Symmetric encryption wrapper using Fernet (AES-128-CBC + HMAC-SHA256).

    If no key is supplied the vault generates an ephemeral key and logs a
    WARNING — data encrypted with that key will be unrecoverable after restart.
    Pass ``NEXUS_CREDENTIAL_ENCRYPTION_KEY`` (a URL-safe base64 32-byte Fernet
    key) in production.
    """

    def __init__(self, key: str = "") -> None:
        if not key:
            generated: bytes = Fernet.generate_key()
            self._fernet = Fernet(generated)
            logger.warning(
                "CredentialEncryption: no encryption key provided — generated an ephemeral key. "
                "Credentials will be unrecoverable after process restart. "
                "Set NEXUS_CREDENTIAL_ENCRYPTION_KEY to a persistent Fernet key."
            )
        else:
            try:
                self._fernet = Fernet(key.encode())
            except Exception as exc:
                raise CredentialError(f"Invalid Fernet key: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: str) -> str:
        """Encrypt *plaintext* and return URL-safe base64 ciphertext string."""
        token: bytes = self._fernet.encrypt(plaintext.encode())
        return token.decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt *ciphertext* and return the original plaintext string.

        Raises:
            CredentialError: if the token is invalid or tampered with.
        """
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:
            raise CredentialError("Decryption failed: invalid or tampered token") from exc
