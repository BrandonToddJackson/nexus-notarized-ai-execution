"""PluginValidator — manifest schema checks and security heuristics."""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexus.marketplace.plugin_sdk import PluginManifest


class PluginValidator:
    """Validates a PluginManifest before installation proceeds.

    All validation is synchronous and has no side effects.
    Called by PluginRegistry.install() before any pip or importlib calls.
    """

    # Tool names reserved by NEXUS core — plugins cannot override these
    _PROTECTED_TOOL_NAMES: frozenset[str] = frozenset({
        "web_search",
        "web_fetch",
        "code_execute",
        "file_read",
        "file_write",
        "shell_execute",
    })

    # Dependencies that plugins must never declare (security / stability)
    _BANNED_DEPENDENCIES: frozenset[str] = frozenset({
        "nexus",       # circular install
        "nexus-core",  # circular install
        "pwntools",    # exploitation framework
        "scapy",       # network packet crafting
    })

    def validate(self, manifest: "PluginManifest") -> list[str]:
        """Run all validation checks.

        Returns:
            List of error strings. Empty list means the manifest is valid.

        Checks (in order):
            1. NEXUS version compatibility
            2. Python version compatibility
            3. Tool name conflicts with protected names
            4. Dependency safety (banned packages)
            5. Tool count ≤ 50
            6. Persona name format
        """
        errors: list[str] = []

        # 1. NEXUS version compatibility
        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version
            import nexus
            nexus_version = Version(nexus.__version__)
            spec = SpecifierSet(manifest.nexus_version)
            if nexus_version not in spec:
                errors.append(
                    f"NEXUS version incompatibility: plugin requires nexus{manifest.nexus_version} "
                    f"but installed version is {nexus.__version__}. "
                    f"Upgrade NEXUS or install an older plugin version."
                )
        except Exception as e:
            errors.append(f"Could not check NEXUS version compatibility: {e}")

        # 2. Python version compatibility
        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version
            current_python = Version(
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )
            py_spec = SpecifierSet(manifest.python_version)
            if current_python not in py_spec:
                errors.append(
                    f"Python version incompatibility: plugin requires python{manifest.python_version} "
                    f"but runtime is Python {current_python}."
                )
        except Exception as e:
            errors.append(f"Could not check Python version compatibility: {e}")

        # 3. Protected tool name conflicts
        for tool_def in manifest.tools:
            if tool_def.name in self._PROTECTED_TOOL_NAMES:
                errors.append(
                    f"Tool name '{tool_def.name}' is reserved by NEXUS core and cannot "
                    f"be overridden by a plugin. Rename your tool."
                )

        # 4. Banned dependencies
        for dep in manifest.dependencies:
            # Extract package name before any version specifier
            pkg_name = re.split(r"[>=<!~\[\s]", dep)[0].lower().strip()
            if pkg_name in self._BANNED_DEPENDENCIES:
                errors.append(
                    f"Dependency '{dep}' is banned for security or stability reasons. "
                    f"Remove it from manifest.dependencies."
                )

        # 5. Tool count limit
        if len(manifest.tools) > 50:
            errors.append(
                f"Plugin declares {len(manifest.tools)} tools but the limit is 50. "
                f"Split into multiple plugins."
            )

        # 6. Persona name format
        persona_name_re = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,63}$")
        for persona_name in manifest.personas:
            if not persona_name_re.match(persona_name):
                errors.append(
                    f"Persona name '{persona_name}' is invalid. "
                    r"Must match ^[a-zA-Z_][a-zA-Z0-9_-]{0,63}$"
                )

        return errors

    @staticmethod
    def check_setuptools_version() -> None:
        """Raise RuntimeError if setuptools < 78.1.1.

        CVE-2025-47273 (path traversal → arbitrary file write) and
        CVE-2024-6345 (RCE via package_index URL handler) both require
        setuptools < 78.1.1. Running the plugin registry on a vulnerable
        setuptools is a supply-chain attack vector.

        Raises:
            RuntimeError: If setuptools version is below the safe minimum.
        """
        try:
            import setuptools
            from packaging.version import Version
            installed = Version(setuptools.__version__)
            minimum = Version("78.1.1")
            if installed < minimum:
                raise RuntimeError(
                    f"SECURITY: setuptools {setuptools.__version__} is installed. "
                    f"CVE-2025-47273 and CVE-2024-6345 require setuptools>=78.1.1. "
                    f"Run: pip install --upgrade 'setuptools>=78.1.1'"
                )
        except ImportError:
            pass  # setuptools not installed at module level — pip will manage it
