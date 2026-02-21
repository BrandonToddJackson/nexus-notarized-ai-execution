"""Orchestrates: validate → sandbox execute → capture result.

The last stop before a tool actually runs.
"""

import logging
from typing import Any, Optional, Tuple

from nexus.types import IntentDeclaration
from nexus.tools.registry import ToolRegistry
from nexus.tools.sandbox import Sandbox
from nexus.core.verifier import IntentVerifier

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Orchestrates tool execution pipeline."""

    def __init__(self, registry: ToolRegistry, sandbox: Sandbox, verifier: IntentVerifier):
        """
        Args:
            registry: Tool registry
            sandbox: Execution sandbox
            verifier: Intent cross-verifier
        """
        self.registry = registry
        self.sandbox = sandbox
        self.verifier = verifier

    async def execute(self, intent: IntentDeclaration) -> Tuple[Any, Optional[str]]:
        """Execute a tool call with full validation.

        Steps:
        1. Get tool from registry
        2. Verify intent matches tool call (IntentVerifier)
        3. Execute in sandbox
        4. Return (result, error_string_or_None)

        Args:
            intent: Declared intent with tool name and parameters

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

        # Step 3: Execute inside sandbox
        try:
            result = await self.sandbox.execute(
                tool_fn, intent.tool_params, definition.timeout_seconds
            )
            return result, None
        except Exception as exc:
            logger.error(f"[Executor] Tool execution error for '{intent.tool_name}': {exc}", exc_info=True)
            return None, "Tool execution failed"
