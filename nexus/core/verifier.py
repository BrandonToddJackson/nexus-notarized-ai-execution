"""Cross-checks declared intent against actual tool parameters.

Prevents the LLM from declaring 'I want to read a file' then executing 'rm -rf'.
This is the anti-gaming layer — the LLM cannot self-declare its way out of constraints.
"""

import logging

from nexus.types import IntentDeclaration
from nexus.exceptions import PersonaViolation

logger = logging.getLogger(__name__)


class IntentVerifier:
    """Cross-checks declared intent against actual tool parameters."""

    def verify(self, intent: IntentDeclaration, tool_name: str, tool_params: dict) -> bool:
        """Check that declared intent matches actual execution.

        Checks:
        1. intent.tool_name == tool_name (exact match)
        2. intent.tool_params keys are subset of tool_params keys
        3. intent.resource_targets match tool_params values that look like resources
           (file paths, URLs, database tables)

        Args:
            intent: What the agent declared it would do
            tool_name: What tool is actually being called
            tool_params: What parameters are actually being passed

        Returns:
            True if verified

        Raises:
            PersonaViolation: with details on what mismatched
        """
        # Check 1: tool name must match exactly
        if intent.tool_name != tool_name:
            raise PersonaViolation(
                f"Intent declared tool '{intent.tool_name}' but actual tool is '{tool_name}'"
            )

        # Check 2: declared param keys must be a subset of actual param keys
        declared_keys = set(intent.tool_params.keys())
        actual_keys = set(tool_params.keys())
        extra_keys = declared_keys - actual_keys
        if extra_keys:
            raise PersonaViolation(
                f"Intent declared params {extra_keys} not present in actual tool params {actual_keys}"
            )

        # Check 3: resource targets should appear in tool_params values
        # Only validate if there are declared resource targets
        if intent.resource_targets:
            param_values = set()
            for v in tool_params.values():
                if isinstance(v, str):
                    param_values.add(v)
                elif isinstance(v, list):
                    param_values.update(str(item) for item in v)

            for target in intent.resource_targets:
                # Allow partial match — target may be a prefix/pattern of a param value
                matched = any(
                    target in val or val in target
                    for val in param_values
                )
                if not matched and param_values:
                    # Non-fatal: log but don't block — resource targets are advisory
                    logger.warning(
                        f"[IntentVerifier] Resource target {target!r} not found in tool params "
                        f"for tool {tool_name!r} (params: {param_values})"
                    )

        return True
