"""Load and validate personas.yaml / tools.yaml into Python objects.

Resolution order for persona files:
  1. Path passed explicitly by caller
  2. ./personas.yaml in current working directory
  3. Built-in defaults (nexus/config/defaults/personas.yaml)

Same pattern for tools.yaml.
"""

from pathlib import Path
from typing import Optional

import yaml

from nexus.config.schema import PersonasConfig, ToolsConfig, PersonaYAML, ToolYAML
from nexus.types import PersonaContract, ToolDefinition, RiskLevel

# Paths to bundled defaults
_DEFAULTS_DIR = Path(__file__).parent / "defaults"
_DEFAULT_PERSONAS = _DEFAULTS_DIR / "personas.yaml"
_DEFAULT_TOOLS = _DEFAULTS_DIR / "tools.yaml"


def _find_file(name: str, explicit: Optional[Path]) -> Path:
    """Locate config file: explicit > cwd > defaults."""
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p

    cwd_path = Path.cwd() / name
    if cwd_path.exists():
        return cwd_path

    defaults_path = _DEFAULTS_DIR / name
    if defaults_path.exists():
        return defaults_path

    raise FileNotFoundError(
        f"No {name} found. Create one in your project directory "
        f"or use load_personas_yaml(path=...)."
    )


def load_personas_yaml(path: Optional[Path] = None) -> list[PersonaContract]:
    """Load personas.yaml → list of PersonaContract objects.

    Args:
        path: Explicit path to personas.yaml. If None, searches cwd then defaults.

    Returns:
        List of validated PersonaContract instances.
    """
    resolved = _find_file("personas.yaml", path)
    raw = yaml.safe_load(resolved.read_text())
    config = PersonasConfig.model_validate(raw or {"personas": []})

    contracts = []
    for entry in config.personas:
        contracts.append(PersonaContract(
            name=entry.name,
            description=entry.description,
            allowed_tools=entry.allowed_tools,
            resource_scopes=entry.resource_scopes,
            intent_patterns=entry.intent_patterns,
            risk_tolerance=entry.risk_tolerance,
            max_ttl_seconds=entry.max_ttl_seconds,
            trust_tier=entry.trust_tier,
        ))
    return contracts


def load_tools_yaml(path: Optional[Path] = None) -> list[ToolDefinition]:
    """Load tools.yaml → list of ToolDefinition objects.

    Args:
        path: Explicit path to tools.yaml. If None, searches cwd then defaults.

    Returns:
        List of validated ToolDefinition instances (no implementations attached).
    """
    resolved = _find_file("tools.yaml", path)
    raw = yaml.safe_load(resolved.read_text())
    config = ToolsConfig.model_validate(raw or {"tools": []})

    definitions = []
    for entry in config.tools:
        definitions.append(ToolDefinition(
            name=entry.name,
            description=entry.description,
            parameters={},           # populated by @tool decorator at runtime
            risk_level=entry.risk_level,
            resource_pattern=entry.resource_pattern,
            timeout_seconds=entry.timeout_seconds,
            requires_approval=entry.requires_approval,
        ))
    return definitions
