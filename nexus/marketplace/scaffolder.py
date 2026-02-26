"""Plugin scaffold generator for ``nexus plugin new``."""

from __future__ import annotations

import re
from pathlib import Path

_VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")


def scaffold_plugin(name: str, output_dir: Path) -> Path:
    """Generate a complete plugin package scaffold.

    Args:
        name: Plugin name (e.g. ``'my-weather-tool'``). Must be lowercase
              letters, digits, and hyphens, at least 2 chars.
        output_dir: Directory to create the package in.

    Returns:
        Path to the created package root (``output_dir / f"nexus-plugin-{name}"``).

    Raises:
        ValueError: If *name* is invalid.
    """
    if not _VALID_NAME_RE.match(name):
        raise ValueError(
            f"Plugin name '{name}' is invalid. "
            f"Use lowercase letters, digits, and hyphens only (min 2 chars). "
            f"Example: 'my-tool'"
        )

    module_name = f"nexus_plugin_{name.replace('-', '_')}"
    tool_fn = f"{name.replace('-', '_')}_action"
    package_dir = output_dir / f"nexus-plugin-{name}"
    module_dir = package_dir / module_name

    package_dir.mkdir(parents=True, exist_ok=True)
    module_dir.mkdir(exist_ok=True)

    (module_dir / "__init__.py").write_text(
        _INIT_TEMPLATE.format(name=name, module_name=module_name, tool_fn=tool_fn)
    )
    (module_dir / "tools.py").write_text(_TOOLS_TEMPLATE.format(tool_fn=tool_fn))
    (module_dir / "personas.yaml").write_text(_PERSONAS_TEMPLATE)
    (package_dir / "pyproject.toml").write_text(
        _PYPROJECT_TEMPLATE.format(name=name, module_name=module_name)
    )
    (package_dir / "README.md").write_text(_README_TEMPLATE.format(name=name))

    return package_dir


# ─── Templates ────────────────────────────────────────────────────────────────

_INIT_TEMPLATE = '''\
"""nexus-plugin-{name} — A NEXUS marketplace plugin."""

from nexus.marketplace import PluginManifest, PluginToolDefinition
from {module_name}.tools import {tool_fn}

plugin_manifest = PluginManifest(
    name="{name}",
    version="0.1.0",
    description="A NEXUS plugin for {name}.",
    author="Your Name",
    author_email="you@example.com",
    tools=[
        PluginToolDefinition(
            name="{tool_fn}",
            description="TODO: describe what this tool does.",
            risk_level="medium",
            parameters={{
                "input": {{"type": "string", "description": "Input string."}},
            }},
        )
    ],
    personas=[],
    dependencies=[],
    nexus_version=">=0.1.0",
    homepage="",
    license="MIT",
    tags=["{name}"],
)
'''

_TOOLS_TEMPLATE = '''\
"""Tool implementations for this plugin."""

from nexus.marketplace import nexus_plugin_tool


@nexus_plugin_tool(
    description="TODO: describe what this tool does.",
    risk_level="medium",
    parameters={{
        "input": {{"type": "string", "description": "Input string."}},
    }},
)
async def {tool_fn}(input: str) -> dict:
    """Process the input and return a result.

    Args:
        input: Input string to process.

    Returns:
        dict with 'result' key.
    """
    # TODO: implement
    return {{"result": f"Processed: {{input}}"}}
'''

_PERSONAS_TEMPLATE = """\
# Optional: define personas bundled with this plugin.
# Delete this file if your plugin does not need custom personas.
# Each entry follows the same schema as nexus/seed/personas.yaml.
#
# Example:
# - name: my_plugin_analyst
#   description: "Scoped persona for my-plugin tools."
#   allowed_tools: [my_action]
#   scope_limits:
#     max_ttl_seconds: 300
#   intent_patterns: ["analyze", "process"]
#   trust_tier: 1
"""

_PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "nexus-plugin-{name}"
version = "0.1.0"
description = "A NEXUS marketplace plugin."
readme = "README.md"
requires-python = ">=3.11"
license = {{text = "MIT"}}
keywords = ["nexus", "nexus-plugin", "{name}"]
classifiers = [
    "Framework :: NEXUS",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "nexus>=0.1.0",
]

[project.urls]
Homepage = "https://github.com/yourname/nexus-plugin-{name}"

[tool.hatch.build.targets.wheel]
packages = ["{module_name}"]
"""

_README_TEMPLATE = """\
# nexus-plugin-{name}

A [NEXUS](https://github.com/BrandonToddJackson/nexus-notarized-ai-execution)
marketplace plugin.

## Installation

```bash
nexus plugin install {name}
```

## Tools

| Tool | Description | Risk Level |
|------|-------------|------------|
| TODO | TODO | medium |

## Usage

After installing, the tool is automatically available to your NEXUS agents.
Every call passes through the 4-gate accountability pipeline and is sealed
in the immutable ledger.

## License

MIT
"""
