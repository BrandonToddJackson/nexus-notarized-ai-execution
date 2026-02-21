"""Constrained execution environment for tool calls.

v1: Timeout + error wrapping. True filesystem/network isolation is v2.
The architecture slot exists but the implementation is lightweight.
"""

import asyncio
from typing import Any, Callable

from nexus.exceptions import ToolError


class Sandbox:
    """Constrained execution environment for tool calls."""

    async def execute(
        self,
        tool_fn: Callable[..., Any],
        params: dict,
        timeout: int = 30,
    ) -> Any:
        """Execute tool function with constraints.

        Constraints:
        - asyncio.wait_for with timeout
        - Catch and wrap all exceptions as ToolError
        - Log execution time

        Args:
            tool_fn: The async tool function to execute
            params: Parameters to pass to the function
            timeout: Timeout in seconds

        Returns:
            Tool result

        Raises:
            ToolError: On timeout or any execution error
        """
        import inspect

        try:
            call = tool_fn(**params)
            if inspect.iscoroutine(call):
                result = await asyncio.wait_for(call, timeout=timeout)
            else:
                # Sync function — run in thread executor to avoid blocking the loop
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: call),
                    timeout=timeout,
                )
            return result
        except asyncio.TimeoutError:
            raise ToolError(
                f"Tool timed out after {timeout}s",
                tool_name=getattr(tool_fn, "__name__", "unknown"),
            )
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                str(exc),
                tool_name=getattr(tool_fn, "__name__", "unknown"),
            ) from exc
