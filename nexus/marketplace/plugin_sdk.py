"""NEXUS Plugin SDK — contracts, decorators, and exceptions for plugin authors."""

from __future__ import annotations

import functools
import inspect
import re
import typing
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, field_validator

# ─── Exception Hierarchy ─────────────────────────────────────────────────────


class PluginError(Exception):
    """Base class for all plugin-related errors."""


class PluginManifestError(PluginError):
    """Manifest is missing required fields or has invalid values."""


class PluginInstallError(PluginError):
    """pip install failed or returned non-zero exit code."""

    def __init__(self, package_name: str, stderr: str):
        self.package_name = package_name
        self.stderr = stderr
        super().__init__(f"Failed to install '{package_name}': {stderr[:500]}")


class PluginImportError(PluginError):
    """Package installed but module cannot be imported."""

    def __init__(self, module_name: str, cause: Exception):
        self.module_name = module_name
        self.cause = cause
        super().__init__(f"Cannot import '{module_name}': {cause}")


class PluginCompatibilityError(PluginError):
    """Plugin requires a newer or incompatible NEXUS version."""

    def __init__(self, plugin_name: str, required: str, current: str):
        super().__init__(
            f"Plugin '{plugin_name}' requires nexus{required}, "
            f"but current version is {current}"
        )


class PluginNotFoundError(PluginError):
    """Named plugin is not installed."""

    def __init__(self, plugin_name: str):
        super().__init__(f"Plugin '{plugin_name}' is not installed")


# ─── Semver / version spec patterns ──────────────────────────────────────────

_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


# ─── PluginToolDefinition ────────────────────────────────────────────────────


class PluginToolDefinition(BaseModel):
    """Lightweight tool descriptor embedded in the manifest.

    The full tool function is imported only after install succeeds.
    """

    model_config = {"frozen": True}

    name: str = Field(
        ...,
        description="Function name as decorated with @nexus_plugin_tool.",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$",
    )
    description: str = Field(
        ...,
        description="One-sentence description shown in `nexus tools` and the frontend tool picker.",
        max_length=500,
    )
    risk_level: str = Field(
        default="medium",
        description="Risk level for Gate 1 scope enforcement: 'low', 'medium', or 'high'.",
        pattern=r"^(low|medium|high)$",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema object describing tool parameters.",
    )
    allowed_personas: list[str] = Field(
        default_factory=list,
        description="If non-empty, this tool is restricted to the listed persona names.",
    )


# ─── PluginManifest ──────────────────────────────────────────────────────────


class PluginManifest(BaseModel):
    """Complete plugin contract. Every NEXUS plugin must export an instance
    of this class as ``plugin_manifest`` from its top-level ``__init__.py``.

    Example::

        from nexus.marketplace import PluginManifest, PluginToolDefinition

        plugin_manifest = PluginManifest(
            name="weather",
            version="1.0.0",
            description="Fetch current weather via Open-Meteo.",
            author="Jane Doe",
            tools=[
                PluginToolDefinition(
                    name="get_current_weather",
                    description="Get current weather for a city.",
                    risk_level="low",
                    parameters={"city": {"type": "string", "description": "City name"}},
                )
            ],
            nexus_version=">=0.1.0",
        )
    """

    model_config = {"frozen": True}

    # ── Identity ──────────────────────────────────────────────────────────────
    name: str = Field(
        ...,
        description="Globally unique plugin name. Lowercase letters, digits, hyphens only.",
        pattern=r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$",
    )
    version: str = Field(..., description="Plugin version in semantic versioning (semver.org).")
    description: str = Field(..., min_length=10, max_length=1000)
    author: str = Field(..., min_length=1, max_length=200)
    author_email: str = Field(default="", max_length=254)

    # ── Tools & Personas ──────────────────────────────────────────────────────
    tools: list[PluginToolDefinition] = Field(..., min_length=1)
    personas: list[str] = Field(default_factory=list)

    # ── Dependencies ──────────────────────────────────────────────────────────
    dependencies: list[str] = Field(default_factory=list)
    nexus_version: str = Field(
        ...,
        description="Minimum NEXUS version specifier. Example: '>=0.1.0'.",
    )
    python_version: str = Field(default=">=3.11")

    # ── Metadata ──────────────────────────────────────────────────────────────
    homepage: str = Field(default="", max_length=2048)
    license: str = Field(default="MIT", max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=20)
    verified: bool = Field(default=False)
    checksum_sha256: str = Field(default="", max_length=64)
    installed_at: Optional[str] = Field(default=None)

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("version")
    @classmethod
    def version_must_be_semver(cls, v: str) -> str:
        if not _SEMVER_RE.match(v):
            raise ValueError(
                f"Version '{v}' is not valid semver. Use MAJOR.MINOR.PATCH, e.g. '1.0.0'."
            )
        return v

    @field_validator("nexus_version", "python_version")
    @classmethod
    def version_spec_must_be_valid(cls, v: str) -> str:
        try:
            from packaging.specifiers import SpecifierSet
            SpecifierSet(v)
        except Exception as e:
            raise ValueError(f"Invalid version specifier '{v}': {e}")
        return v

    @field_validator("tags")
    @classmethod
    def tags_must_be_lowercase(cls, v: list[str]) -> list[str]:
        cleaned = []
        for tag in v:
            if len(tag) > 50:
                raise ValueError(f"Tag '{tag[:20]}...' exceeds 50 chars")
            cleaned.append(tag.lower().strip())
        return cleaned

    @field_validator("tools")
    @classmethod
    def tool_names_must_be_unique(cls, v: list[PluginToolDefinition]) -> list[PluginToolDefinition]:
        names = [t.name for t in v]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate tool names in manifest: {list(set(dupes))}")
        return v


# ─── @nexus_plugin_tool decorator ────────────────────────────────────────────


def nexus_plugin_tool(
    name: Optional[str] = None,
    description: str = "",
    risk_level: str = "medium",
    parameters: Optional[dict] = None,
    allowed_personas: Optional[list[str]] = None,
) -> Callable:
    """Decorator that marks a function as a NEXUS plugin tool.

    The decorated function MUST be async, accept keyword arguments matching
    the ``parameters`` keys, and return a dict or JSON-serialisable value.

    Args:
        name: Tool name. Defaults to the function's ``__name__``.
        description: One-sentence description. Defaults to docstring first line.
        risk_level: ``'low'``, ``'medium'``, or ``'high'``.
        parameters: JSON Schema dict mapping param names to type descriptors.
                    Auto-inferred from type hints when *None*.
        allowed_personas: Restrict tool to these persona names (empty = unrestricted).

    Raises:
        ValueError: If the decorated function is not async, or *risk_level* is invalid.
    """
    valid_risk_levels = {"low", "medium", "high"}

    def decorator(func: Callable) -> Callable:
        if not inspect.iscoroutinefunction(func):
            raise ValueError(
                f"@nexus_plugin_tool: '{func.__name__}' must be an async function. "
                f"Change 'def {func.__name__}' to 'async def {func.__name__}'."
            )
        if risk_level not in valid_risk_levels:
            raise ValueError(
                f"@nexus_plugin_tool: risk_level='{risk_level}' is invalid. "
                f"Must be one of: {valid_risk_levels}"
            )

        tool_name = name or func.__name__

        # Auto-extract description from docstring first non-empty line
        tool_description = description
        if not tool_description and func.__doc__:
            for line in func.__doc__.strip().splitlines():
                stripped = line.strip()
                if stripped:
                    tool_description = stripped
                    break

        # Auto-infer parameters from type hints when not provided
        tool_parameters = parameters if parameters is not None else _infer_parameters_from_signature(func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Guard against oversized parameters (prompt injection / payload delivery)
            for key, val in kwargs.items():
                if isinstance(val, str) and len(val) > 50_000:
                    raise ValueError(
                        f"Plugin tool '{tool_name}': parameter '{key}' exceeds 50,000 char limit. "
                        f"Possible prompt injection or payload delivery attempt."
                    )
            return await func(*args, **kwargs)

        # Attach metadata for PluginRegistry introspection
        wrapper.__nexus_plugin_tool__ = True
        wrapper.__nexus_tool_name__ = tool_name
        wrapper.__nexus_tool_description__ = tool_description
        wrapper.__nexus_tool_risk_level__ = risk_level
        wrapper.__nexus_tool_parameters__ = tool_parameters
        wrapper.__nexus_tool_allowed_personas__ = allowed_personas or []

        return wrapper

    return decorator


def _infer_parameters_from_signature(func: Callable) -> dict:
    """Auto-generate a JSON Schema parameters dict from Python type hints.

    Handles: ``str``, ``int``, ``float``, ``bool``, ``list``, ``dict``,
    ``Optional[X]``. Falls back to ``{"type": "string"}`` for unknown types.
    """
    _TYPE_MAP = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    sig = inspect.signature(func)
    params: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            params[param_name] = {"type": "string", "description": ""}
            continue

        # Handle Optional[X] → Union[X, None]
        origin = getattr(annotation, "__origin__", None)
        args_hint = getattr(annotation, "__args__", ())

        if origin is typing.Union and type(None) in args_hint:
            inner = next((a for a in args_hint if a is not type(None)), str)
            json_type = _TYPE_MAP.get(inner, "string")
            entry: dict[str, Any] = {"type": json_type, "description": ""}
            if param.default is not inspect.Parameter.empty:
                entry["default"] = param.default
            params[param_name] = entry
            continue

        json_type = _TYPE_MAP.get(annotation, "string")
        entry = {"type": json_type, "description": ""}
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        params[param_name] = entry

    return params
