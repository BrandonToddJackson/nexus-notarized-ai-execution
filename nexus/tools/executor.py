"""Orchestrates: validate → credential injection → sandbox execute → capture result.

The last stop before a tool actually runs.
"""

import logging
import re
from typing import TYPE_CHECKING, Any, Optional, Tuple

from nexus.exceptions import ToolError
from nexus.types import IntentDeclaration
from nexus.tools.registry import ToolRegistry
from nexus.tools.sandbox import Sandbox
from nexus.core.verifier import IntentVerifier

if TYPE_CHECKING:
    from nexus.credentials.vault import CredentialVault
    from nexus.config import NexusConfig

logger = logging.getLogger(__name__)

_TEMPLATE_RE = re.compile(r"\{\{config\.(\w+)\}\}")


def _resolve_config_templates(params: dict, config: "NexusConfig") -> dict:
    """Replace {{config.field_name}} in string param values with NexusConfig values.

    This lets workflow tool_params reference secrets stored in .env without
    exposing them to the AI planner or the ledger (sanitize_tool_params strips
    them before sealing).

    Example:
        params = {"params": {"api_key": "{{config.instantly_api_key}}"}}
        → {"params": {"api_key": "abc123"}}  (if config.instantly_api_key = "abc123")
    """
    if config is None:
        return params

    def _resolve_value(v: Any) -> Any:
        if isinstance(v, str):
            def _sub(m: re.Match) -> str:
                field = m.group(1)
                return str(getattr(config, field, m.group(0)))
            return _TEMPLATE_RE.sub(_sub, v)
        if isinstance(v, dict):
            return {k: _resolve_value(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_resolve_value(item) for item in v]
        return v

    return {k: _resolve_value(v) for k, v in params.items()}


class ToolExecutor:
    """Orchestrates tool execution pipeline."""

    def __init__(
        self,
        registry: ToolRegistry,
        sandbox: Sandbox,
        verifier: IntentVerifier,
        vault: Optional["CredentialVault"] = None,
        config: Optional["NexusConfig"] = None,
    ):
        """
        Args:
            registry: Tool registry
            sandbox: Execution sandbox
            verifier: Intent cross-verifier
            vault: Optional credential vault; when present, injects credentials
                   into tool_params if ``credential_id`` is set on the intent.
            config: NexusConfig instance; when present, resolves {{config.X}}
                    template strings in tool_params before execution.
        """
        self.registry = registry
        self.sandbox = sandbox
        self.verifier = verifier
        self.vault = vault
        self.config = config

    async def execute(
        self,
        intent: IntentDeclaration,
        tenant_id: str = "",
        persona_name: Optional[str] = None,
    ) -> Tuple[Any, Optional[str]]:
        """Execute a tool call with full validation.

        Steps:
        1. Get tool from registry
        2. Verify intent matches tool call (IntentVerifier)
        3. Inject credentials from vault (if credential_id present in params)
        4. Execute in sandbox
        5. Return (result, error_string_or_None)

        Args:
            intent: Declared intent with tool name and parameters
            tenant_id: Calling tenant; required for vault credential lookup.
            persona_name: Active persona name; used for scoped-credential checks.

        Returns:
            Tuple of (result, error). error is None on success.
        """
        # Step 1: Resolve tool from registry
        try:
            definition, tool_fn = self.registry.get(intent.tool_name)
        except Exception as exc:
            logger.warning(f"[Executor] Tool lookup failed for '{intent.tool_name}': {exc}")
            return None, "Tool not found"

        # Step 2: Cross-verify declared intent vs actual call
        try:
            self.verifier.verify(intent, intent.tool_name, intent.tool_params)
        except Exception as exc:
            logger.warning(f"[Executor] Intent verification failed for '{intent.tool_name}': {exc}")
            return None, "Intent verification failed"

        # Step 3: Resolve {{config.X}} templates then inject vault credentials
        params = dict(intent.tool_params)
        params = _resolve_config_templates(params, self.config)
        credential_id = params.pop("credential_id", None)
        if credential_id and self.vault is not None:
            try:
                params = self.vault.inject_credentials(
                    credential_id=credential_id,
                    tenant_id=tenant_id,
                    persona_name=persona_name,
                    tool_params=params,
                )
                logger.debug("[Executor] Injected credentials from vault for credential '%s'", credential_id)
            except Exception as exc:
                logger.warning("[Executor] Credential injection failed: %s", exc)
                return None, f"Credential injection failed: {exc}"

        # Step 4: Execute inside sandbox with (possibly enriched) params
        try:
            result = await self.sandbox.execute(
                tool_fn, params, definition.timeout_seconds
            )
            return result, None
        except ToolError as te:
            # Surface the tool's own error message (e.g. "API key invalid", "url is required")
            logger.warning(f"[Executor] ToolError for '{intent.tool_name}': {te}")
            return None, str(te)
        except Exception as exc:
            logger.error(f"[Executor] Tool execution error for '{intent.tool_name}': {exc}", exc_info=True)
            return None, "Tool execution failed"
