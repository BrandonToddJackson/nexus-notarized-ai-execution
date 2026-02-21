"""Validates tool output matches the declared intent.

Also performs basic safety checks: PII detection, coherence.
"""

import re
from typing import Any

from nexus.types import IntentDeclaration


class OutputValidator:
    """Validates tool output matches the declared intent."""

    # PII regex patterns
    SSN_PATTERN = re.compile(r"\d{3}-\d{2}-\d{4}")
    CC_PATTERN = re.compile(r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}")
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    async def validate(self, intent: IntentDeclaration, tool_result: Any) -> tuple[bool, str]:
        """Validate tool output.

        Checks:
        1. Result is not None/empty (unless intent was a delete)
        2. Basic PII scan: SSN, credit card, email patterns in unexpected places
        3. If result is text, basic coherence check (not just error messages)

        Args:
            intent: What was intended
            tool_result: What was actually returned

        Returns:
            Tuple of (is_valid, reason). reason is "" if valid.
        """
        # Check 1: Result not None/empty (unless it's a delete/write action)
        planned = intent.planned_action.lower()
        is_destructive = any(kw in planned for kw in ("delete", "remove", "clear", "write", "send"))

        if tool_result is None and not is_destructive:
            return False, "Tool returned None result for non-destructive action"

        if tool_result == "" and not is_destructive:
            return False, "Tool returned empty string for non-destructive action"

        if isinstance(tool_result, (list, dict)) and len(tool_result) == 0 and not is_destructive:
            return False, "Tool returned empty collection for non-destructive action"

        # Check 2: PII scan on string results
        if isinstance(tool_result, str):
            if self.SSN_PATTERN.search(tool_result):
                return False, "Output contains potential SSN (PII detected)"
            if self.CC_PATTERN.search(tool_result):
                return False, "Output contains potential credit card number (PII detected)"

            # Check 3: Coherence — result shouldn't look like a raw error trace
            error_indicators = [
                "Traceback (most recent call last)",
                "Error: command not found",
                "Permission denied",
                "FATAL ERROR",
            ]
            for indicator in error_indicators:
                if indicator in tool_result:
                    return False, f"Output looks like an error message: '{indicator}'"

        return True, ""
