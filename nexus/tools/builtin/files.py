"""Built-in file tools: read and write with path validation."""

import logging
from pathlib import Path

from nexus.tools.plugin import tool
from nexus.types import RiskLevel
from nexus.exceptions import ToolError

logger = logging.getLogger(__name__)

# Allowed base directories for file I/O. Only paths under these prefixes are permitted.
# This prevents path traversal attacks (e.g., reading /etc/passwd).
_ALLOWED_BASE_DIRS = [
    Path.home() / ".nexus" / "workspace",
    Path("/tmp/nexus"),
]


def _safe_path(path: str) -> Path:
    """Resolve path and validate it is within an allowed workspace directory.

    Args:
        path: File path string (absolute or relative)

    Returns:
        Resolved absolute Path

    Raises:
        ToolError: If the resolved path escapes the allowed workspace
    """
    p = Path(path).resolve()
    for base in _ALLOWED_BASE_DIRS:
        try:
            p.relative_to(base.resolve())
            return p
        except ValueError:
            continue
    allowed = ", ".join(str(b) for b in _ALLOWED_BASE_DIRS)
    raise ToolError(
        f"Path '{path}' is outside the allowed workspace. "
        f"Allowed directories: {allowed}"
    )


@tool(name="file_read", description="Read a file's contents", risk_level=RiskLevel.LOW, resource_pattern="file:read:*")
async def file_read(path: str) -> str:
    """Read a file's contents from the NEXUS workspace.

    Only files within ~/.nexus/workspace/ or /tmp/nexus/ can be read.

    Args:
        path: File path to read (must be within allowed workspace)

    Returns:
        File contents as text
    """
    safe = _safe_path(path)
    if not safe.exists():
        return f"File not found: {path}"
    if not safe.is_file():
        return f"Path is not a file: {path}"
    try:
        return safe.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return safe.read_text(encoding="latin-1")


@tool(name="file_write", description="Write content to a file", risk_level=RiskLevel.MEDIUM, resource_pattern="file:write:*")
async def file_write(path: str, content: str) -> str:
    """Write content to a file in the NEXUS workspace.

    Only files within ~/.nexus/workspace/ or /tmp/nexus/ can be written.
    Parent directories are created automatically.

    Args:
        path: File path to write to (must be within allowed workspace)
        content: Content to write

    Returns:
        Confirmation message with byte count
    """
    safe = _safe_path(path)
    safe.parent.mkdir(parents=True, exist_ok=True)
    safe.write_text(content, encoding="utf-8")
    return f"Written {len(content.encode('utf-8'))} bytes to {safe}"
