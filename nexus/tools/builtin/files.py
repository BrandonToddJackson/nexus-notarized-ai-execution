"""Built-in file tools: read and write."""

from nexus.tools.plugin import tool
from nexus.types import RiskLevel


@tool(name="file_read", description="Read a file\'s contents", risk_level=RiskLevel.LOW, resource_pattern="file:read:*")
async def file_read(path: str) -> str:
    """Read a file's contents.

    Args:
        path: File path to read

    Returns:
        File contents as text
    """
    # v1 stub — replace with actual file I/O with path validation
    return f"Contents of {path}: [File content would appear here]"


@tool(name="file_write", description="Write content to a file", risk_level=RiskLevel.MEDIUM, resource_pattern="file:write:*")
async def file_write(path: str, content: str) -> str:
    """Write content to a file.

    Args:
        path: File path to write to
        content: Content to write

    Returns:
        Confirmation message
    """
    # v1 stub — replace with actual file I/O with path validation
    return f"Written {len(content)} chars to {path}"
