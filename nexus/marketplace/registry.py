"""PluginRegistry — install, uninstall, upgrade, list, and search plugins."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml

from nexus.marketplace.plugin_sdk import (
    PluginManifest,
    PluginInstallError,
    PluginImportError,
    PluginNotFoundError,
    PluginManifestError,
)
from nexus.marketplace.validator import PluginValidator

logger = logging.getLogger(__name__)

# PyPI JSON API base
_PYPI_JSON_API = "https://pypi.org/pypi"
_PYPI_SEARCH_URL = "https://pypi.org/search/"

# All NEXUS marketplace plugins must be named nexus-plugin-<name> on PyPI
_PACKAGE_PREFIX = "nexus-plugin-"


class PluginRegistry:
    """Manages the lifecycle of NEXUS marketplace plugins.

    Args:
        tool_registry: ToolRegistry instance to register/deregister tools.
                       May be ``None`` in headless CLI mode — install still
                       persists state but skips live tool registration.
        config: NexusConfig for pip timeouts and security flags.
        persona_manager: Optional PersonaManager. When present, plugin
                         ``personas.yaml`` definitions are loaded on install.
    """

    def __init__(
        self,
        tool_registry: Any,  # Optional[ToolRegistry] — avoid circular import
        config: Any,          # NexusConfig
        persona_manager: Any = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._persona_manager = persona_manager
        self._config = config
        self._validator = PluginValidator()

        # plugin_name → PluginManifest
        self._installed: dict[str, PluginManifest] = {}

        # plugin_name → list[tool_name]
        self._plugin_tools: dict[str, list[str]] = {}

        # plugin_name → list[persona_name]
        self._plugin_personas: dict[str, list[str]] = {}

        # tool_name → list[persona_name] (empty = unrestricted)
        self._tool_allowed_personas: dict[str, list[str]] = {}

        # Serializes all install/uninstall operations
        self._lock = asyncio.Lock()

        # Runtime CVE guard: check setuptools version once at startup
        # (CVE-2025-47273 path traversal + CVE-2024-6345 RCE both require setuptools < 78.1.1)
        self._validator.check_setuptools_version()

        # Persisted state location
        self._state_file = Path.home() / ".nexus" / "plugins" / "installed.json"
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

    # ─── Public API ───────────────────────────────────────────────────────────

    async def install(
        self,
        package_name: str,
        version: Optional[str] = None,
        force: bool = False,
    ) -> PluginManifest:
        """Install a plugin from PyPI.

        Sequence:
            1. Validate package name format (typosquatting defence)
            2. Normalize to ``nexus-plugin-<name>[==version]``
            3. Skip if already installed and ``force=False``
            4. Dry-run dep existence check (if ``config.plugin_dry_run_check``)
            5. pip install subprocess
            6. Static source scan before import (CVE-2025-14009 mitigation)
            7. ``importlib.import_module``
            8. Retrieve and verify ``plugin_manifest`` attribute
            9. ``PluginValidator.validate()``
            10. Register tools into ToolRegistry (if available)
            11. Load ``personas.yaml`` (if persona_manager available)
            12. Compute SHA-256 checksum
            13. Build final manifest with metadata
            14. Persist state to disk

        Args:
            package_name: PyPI package name with optional ``nexus-plugin-`` prefix
                          and optional ``==version`` specifier.
            version: Explicit version. Ignored if *package_name* already has a specifier.
            force: Reinstall even if already installed at the same version.

        Returns:
            :class:`PluginManifest` with ``checksum_sha256`` and ``installed_at`` set.

        Raises:
            PluginInstallError: pip returned non-zero exit code.
            PluginImportError: Package installed but cannot be imported.
            PluginManifestError: Module lacks ``plugin_manifest``, manifest is invalid,
                                  or manifest fails validation.
        """
        async with self._lock:
            # 1. Validate package name format
            self._validate_package_name_format(package_name)

            # 2. Normalize
            pip_specifier = _normalize_package_name(package_name, version)
            module_name = _pip_specifier_to_module_name(pip_specifier)
            plugin_name = module_name.removeprefix("nexus_plugin_").replace("_", "-")

            # 3. Skip if already installed
            if plugin_name in self._installed and not force:
                logger.info(
                    "Plugin '%s' already installed at v%s. Use force=True to reinstall.",
                    plugin_name,
                    self._installed[plugin_name].version,
                )
                return self._installed[plugin_name]

            # 4. Dry-run check
            if getattr(self._config, "plugin_dry_run_check", True):
                await self._pip_dry_run(pip_specifier)

            # 5. pip install
            logger.info("Installing plugin package: %s", pip_specifier)
            await self._pip_install(pip_specifier)

            # 6. Static scan before import
            scan_warnings = await self._static_scan_module(module_name)
            if scan_warnings:
                if getattr(self._config, "plugin_strict_scan", False):
                    await self._pip_uninstall(_base_name(pip_specifier))
                    raise PluginManifestError(
                        f"Plugin '{module_name}' failed static security scan and was NOT imported.\n"
                        "Suspicious patterns:\n"
                        + "\n".join(f"  - {w}" for w in scan_warnings)
                        + "\nSet NEXUS_PLUGIN_STRICT_SCAN=False to bypass (not recommended)."
                    )
                else:
                    for w in scan_warnings:
                        logger.warning("SECURITY WARNING [%s]: %s", module_name, w)

            # 7. Import
            importlib.invalidate_caches()
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                raise PluginImportError(module_name, exc) from exc

            # 8. Get manifest
            if not hasattr(module, "plugin_manifest"):
                raise PluginManifestError(
                    f"Plugin module '{module_name}' does not export 'plugin_manifest'. "
                    f"Every NEXUS plugin must define: plugin_manifest = PluginManifest(...) "
                    f"in its __init__.py."
                )

            manifest: PluginManifest = module.plugin_manifest

            if not isinstance(manifest, PluginManifest):
                raise PluginManifestError(
                    f"'plugin_manifest' in '{module_name}' is not a PluginManifest instance. "
                    f"Got: {type(manifest).__name__}"
                )

            # 9. Validate
            errors = self._validator.validate(manifest)
            if errors:
                await self._pip_uninstall(_base_name(pip_specifier))
                raise PluginManifestError(
                    f"Plugin '{manifest.name}' failed validation with {len(errors)} error(s):\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )

            # 10. Register tools
            registered_tools: list[str] = []
            if self._tool_registry is not None:
                from nexus.types import ToolDefinition, RiskLevel
                _risk_map = {
                    "low": RiskLevel.LOW,
                    "medium": RiskLevel.MEDIUM,
                    "high": RiskLevel.HIGH,
                }
                for tool_def in manifest.tools:
                    tool_func = _resolve_tool_function(module, tool_def.name)
                    definition = ToolDefinition(
                        name=tool_def.name,
                        description=tool_def.description,
                        parameters={
                            "type": "object",
                            "properties": tool_def.parameters,
                            "required": list(tool_def.parameters.keys()),
                        },
                        risk_level=_risk_map.get(tool_def.risk_level, RiskLevel.MEDIUM),
                    )
                    self._tool_registry.register(definition, tool_func, source="marketplace")
                    registered_tools.append(tool_def.name)
                    if tool_def.allowed_personas:
                        self._tool_allowed_personas[tool_def.name] = list(tool_def.allowed_personas)
                    logger.debug(
                        "Registered tool '%s' from plugin '%s'", tool_def.name, manifest.name
                    )

            self._plugin_tools[manifest.name] = registered_tools

            # 11. Load personas
            loaded_personas: list[str] = []
            if self._persona_manager is not None and manifest.personas:
                loaded_personas = await self._load_plugin_personas(module, manifest)
            self._plugin_personas[manifest.name] = loaded_personas

            # 12. Checksum
            checksum = await self._compute_package_checksum(module_name)

            # 13. Final manifest (frozen → model_copy)
            final_manifest = manifest.model_copy(
                update={
                    "verified": False,  # never trust self-declared verified=True
                    "checksum_sha256": checksum,
                    "installed_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            # 14. Persist
            self._installed[manifest.name] = final_manifest
            await self._persist_state()

            logger.info(
                "Plugin '%s' v%s installed (%d tools, %d personas).",
                manifest.name,
                manifest.version,
                len(registered_tools),
                len(loaded_personas),
            )
            return final_manifest

    async def uninstall(self, plugin_name: str) -> bool:
        """Uninstall a plugin and remove its tools and personas.

        Args:
            plugin_name: ``manifest.name`` (e.g. ``'weather'``).

        Returns:
            ``True`` if successfully uninstalled.

        Raises:
            PluginNotFoundError: Plugin is not installed.
        """
        async with self._lock:
            if plugin_name not in self._installed:
                raise PluginNotFoundError(plugin_name)

            # Deregister tools
            if self._tool_registry is not None:
                for tool_name in self._plugin_tools.get(plugin_name, []):
                    try:
                        self._tool_registry.unregister(tool_name)
                        self._tool_allowed_personas.pop(tool_name, None)
                        logger.debug("Deregistered tool '%s'", tool_name)
                    except Exception as exc:
                        logger.warning("Failed to deregister tool '%s': %s", tool_name, exc)

            # Unload personas
            if self._persona_manager is not None:
                for persona_name in self._plugin_personas.get(plugin_name, []):
                    try:
                        await self._persona_manager.delete(persona_name)
                        logger.debug("Unloaded persona '%s'", persona_name)
                    except Exception as exc:
                        logger.warning("Failed to unload persona '%s': %s", persona_name, exc)

            # pip uninstall
            pip_package_name = f"{_PACKAGE_PREFIX}{plugin_name}"
            await self._pip_uninstall(pip_package_name)

            # Clean up state
            del self._installed[plugin_name]
            self._plugin_tools.pop(plugin_name, None)
            self._plugin_personas.pop(plugin_name, None)

            await self._persist_state()
            logger.info("Plugin '%s' uninstalled.", plugin_name)
            return True

    async def upgrade(
        self,
        plugin_name: str,
        target_version: Optional[str] = None,
    ) -> PluginManifest:
        """Upgrade an installed plugin to a newer version.

        Equivalent to: ``uninstall`` then ``install(force=True)``.

        Args:
            plugin_name: ``manifest.name`` of the installed plugin.
            target_version: Specific version. ``None`` = latest.

        Raises:
            PluginNotFoundError: Plugin is not installed.
        """
        if plugin_name not in self._installed:
            raise PluginNotFoundError(plugin_name)
        await self.uninstall(plugin_name)
        # Pass the full pip package name so _validate_package_name_format passes
        return await self.install(
            f"{_PACKAGE_PREFIX}{plugin_name}", version=target_version, force=True
        )

    def list_installed(self) -> list[PluginManifest]:
        """Return all installed plugin manifests, sorted by name."""
        return sorted(self._installed.values(), key=lambda m: m.name)

    def get(self, plugin_name: str) -> PluginManifest:
        """Return the manifest for a specific installed plugin.

        Raises:
            PluginNotFoundError: Plugin is not installed.
        """
        if plugin_name not in self._installed:
            raise PluginNotFoundError(plugin_name)
        return self._installed[plugin_name]

    def is_persona_allowed(self, tool_name: str, persona_name: str) -> bool:
        """Check whether a persona is allowed to use a plugin tool.

        Returns:
            ``True`` if the tool has no persona restriction, or the persona
            is explicitly listed in ``allowed_personas``.  Always returns
            ``True`` for tools not registered by this registry.
        """
        restriction = self._tool_allowed_personas.get(tool_name)
        if not restriction:
            return True
        return persona_name in restriction

    async def search(
        self,
        query: str,
        limit: int = 20,
        timeout: float = 10.0,
    ) -> list[dict]:
        """Search PyPI for NEXUS plugins matching a query.

        Returns ``[]`` on timeout or network error — never raises.

        Args:
            query: Free-text search term (e.g. ``'slack'``, ``'github'``).
            limit: Maximum results (1–100).
            timeout: HTTP timeout in seconds.

        Returns:
            List of dicts with keys: ``name``, ``plugin_name``, ``version``,
            ``description``, ``author``, ``homepage``,
            ``downloads_last_month``, ``installed``, ``installed_version``,
            ``verified``.
        """
        limit = max(1, min(100, limit))
        results: list[dict] = []

        search_url = (
            f"{_PYPI_SEARCH_URL}?q=nexus-plugin-{query}&c=Framework+%3A%3A+NEXUS"
        )

        try:
            import nexus as _nexus_pkg
            ua = f"nexus/{_nexus_pkg.__version__} plugin-registry"
        except Exception:
            ua = "nexus/unknown plugin-registry"

        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                headers={"User-Agent": ua},
                follow_redirects=True,
            ) as client:
                response = await client.get(search_url)
                response.raise_for_status()

                package_names = _parse_pypi_search_html(response.text)
                package_names = [
                    p for p in package_names if p.startswith(_PACKAGE_PREFIX)
                ][:limit]

                metadata_tasks = [
                    self._fetch_pypi_metadata(client, pkg_name)
                    for pkg_name in package_names
                ]
                metadata_results = await asyncio.gather(
                    *metadata_tasks, return_exceptions=True
                )

                for pkg_name, meta in zip(package_names, metadata_results):
                    if isinstance(meta, Exception):
                        logger.debug(
                            "Failed to fetch metadata for '%s': %s", pkg_name, meta
                        )
                        continue
                    short_name = pkg_name.removeprefix(_PACKAGE_PREFIX)
                    installed = short_name in self._installed
                    results.append({
                        "name": pkg_name,
                        "plugin_name": short_name,
                        "version": meta.get("version", ""),
                        "description": meta.get("description", ""),
                        "author": meta.get("author", ""),
                        "homepage": meta.get("homepage", ""),
                        "downloads_last_month": meta.get("downloads_last_month", 0),
                        "installed": installed,
                        "installed_version": (
                            self._installed[short_name].version if installed else ""
                        ),
                        "verified": False,
                    })

        except httpx.TimeoutException:
            logger.warning("PyPI search timed out after %.1fs for '%s'", timeout, query)
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "PyPI search HTTP error %s for '%s'", exc.response.status_code, query
            )
        except Exception as exc:
            logger.warning("PyPI search failed for '%s': %s", query, exc)

        results.sort(key=lambda r: r["downloads_last_month"], reverse=True)
        return results

    async def load_state(self) -> None:
        """Restore previously installed plugins from disk.

        Called during application startup. Re-imports all persisted plugins
        and re-registers their tools. Skips any plugin that cannot be imported
        (e.g. manually pip-uninstalled), logging a warning instead of raising.
        """
        if not self._state_file.exists():
            return

        try:
            state: dict = json.loads(self._state_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not load plugin state from %s: %s", self._state_file, exc
            )
            return

        for plugin_name, manifest_data in state.items():
            try:
                manifest = PluginManifest.model_validate(manifest_data)
                module_name = f"nexus_plugin_{plugin_name.replace('-', '_')}"

                importlib.invalidate_caches()
                module = importlib.import_module(module_name)

                registered_tools: list[str] = []
                if self._tool_registry is not None:
                    from nexus.types import ToolDefinition, RiskLevel
                    _risk_map = {
                        "low": RiskLevel.LOW,
                        "medium": RiskLevel.MEDIUM,
                        "high": RiskLevel.HIGH,
                    }
                    for tool_def in manifest.tools:
                        tool_func = _resolve_tool_function(module, tool_def.name)
                        definition = ToolDefinition(
                            name=tool_def.name,
                            description=tool_def.description,
                            parameters={
                                "type": "object",
                                "properties": tool_def.parameters,
                                "required": list(tool_def.parameters.keys()),
                            },
                            risk_level=_risk_map.get(tool_def.risk_level, RiskLevel.MEDIUM),
                        )
                        self._tool_registry.register(
                            definition, tool_func, source="marketplace"
                        )
                        registered_tools.append(tool_def.name)
                        if tool_def.allowed_personas:
                            self._tool_allowed_personas[tool_def.name] = list(
                                tool_def.allowed_personas
                            )

                self._installed[plugin_name] = manifest
                self._plugin_tools[plugin_name] = registered_tools
                logger.info(
                    "Restored plugin '%s' v%s (%d tools).",
                    plugin_name,
                    manifest.version,
                    len(registered_tools),
                )

            except ImportError:
                logger.warning(
                    "Plugin '%s' is in state file but cannot be imported. "
                    "It may have been uninstalled manually. Skipping.",
                    plugin_name,
                )
            except Exception as exc:
                logger.warning("Failed to restore plugin '%s': %s", plugin_name, exc)

    # ─── Security helpers ─────────────────────────────────────────────────────

    def _validate_package_name_format(self, package_name: str) -> None:
        """Enforce strict naming and detect typosquatting attempts.

        Raises:
            PluginManifestError: Package name is invalid.
        """
        # Strip version specifier for name validation
        base_name = re.split(r"[>=<!~\[]", package_name)[0].strip()

        # Must start with nexus-plugin-
        if not base_name.startswith(_PACKAGE_PREFIX):
            raise PluginManifestError(
                f"Package '{base_name}' does not start with '{_PACKAGE_PREFIX}'. "
                f"All NEXUS marketplace plugins must follow this naming convention. "
                f"Did you mean '{_PACKAGE_PREFIX}{base_name}'?"
            )

        # No consecutive hyphens
        if "--" in base_name:
            raise PluginManifestError(
                f"Package '{base_name}' contains consecutive hyphens, which is invalid."
            )

        # Validate plugin name portion
        plugin_name_part = base_name.removeprefix(_PACKAGE_PREFIX)
        if not re.match(r"^[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$", plugin_name_part):
            # Single char names are also ok
            if not re.match(r"^[a-z0-9]$", plugin_name_part):
                raise PluginManifestError(
                    f"Plugin name portion '{plugin_name_part}' contains invalid characters. "
                    f"Use lowercase letters, digits, and single hyphens only."
                )

        # Levenshtein proximity warning (typosquatting detection)
        for installed_name in self._installed:
            distance = _levenshtein(plugin_name_part, installed_name)
            if 0 < distance <= 2:
                logger.warning(
                    "SECURITY WARNING: '%s%s' is very similar to installed plugin "
                    "'%s%s' (edit distance=%d). Possible typosquatting — verify carefully.",
                    _PACKAGE_PREFIX,
                    plugin_name_part,
                    _PACKAGE_PREFIX,
                    installed_name,
                    distance,
                )

    async def _static_scan_module(self, module_name: str) -> list[str]:
        """Scan installed plugin source for dangerous patterns BEFORE importing.

        Best-effort defence against CVE-2025-14009 (malicious ``__init__.py``
        executing at import time). Returns a list of warning strings; empty
        means the scan found nothing suspicious.
        """
        warnings_found: list[str] = []

        spec = importlib.util.find_spec(module_name)
        if not spec or not spec.submodule_search_locations:
            return warnings_found

        pkg_dir = Path(list(spec.submodule_search_locations)[0])

        files_to_scan = [
            pkg_dir / "__init__.py",
            pkg_dir / "tools.py",
        ]

        _DANGEROUS_PATTERNS = [
            (r"\bos\.system\s*\(", "os.system() call"),
            (r"\bos\.popen\s*\(", "os.popen() call"),
            (r"\bsubprocess\.", "subprocess usage"),
            (r"\bsocket\.socket\s*\(", "raw socket creation"),
            (r"__import__\s*\(\s*['\"]os['\"]", "__import__('os') obfuscation"),
            (r"\beval\s*\(", "eval() usage"),
            (r"\bexec\s*\(", "exec() usage"),
            (r"base64\.b64decode", "Base64 decode (possible payload obfuscation)"),
            (r"urllib\.request\.urlopen", "HTTP request at import time"),
        ]

        for file_path in files_to_scan:
            if not file_path.exists():
                continue
            source = file_path.read_text(errors="replace")
            for pattern, description in _DANGEROUS_PATTERNS:
                if re.search(pattern, source):
                    warnings_found.append(f"{file_path.name}: {description}")

        return warnings_found

    # ─── pip helpers ─────────────────────────────────────────────────────────

    async def _pip_install(self, specifier: str) -> None:
        """Run ``pip install`` in a subprocess."""
        cmd = [
            sys.executable, "-m", "pip", "install",
            specifier, "--quiet", "--no-input",
        ]
        timeout = getattr(self._config, "plugin_install_timeout", 120.0)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise PluginInstallError(
                specifier, f"pip install timed out after {timeout}s"
            )

        if proc.returncode != 0:
            raise PluginInstallError(
                specifier, stderr.decode("utf-8", errors="replace")
            )

    async def _pip_uninstall(self, package_name: str) -> None:
        """Run ``pip uninstall -y``. Logs warning on failure, never raises."""
        cmd = [
            sys.executable, "-m", "pip", "uninstall",
            package_name, "-y", "--quiet",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(
                "pip uninstall '%s' failed (exit %s): %s",
                package_name,
                proc.returncode,
                stderr.decode("utf-8", errors="replace")[:300],
            )

    async def _pip_dry_run(self, specifier: str) -> None:
        """Run ``pip install --dry-run`` to detect dependency issues.

        Logs warnings on problems but never blocks installation — this is
        a best-effort check for CVE-2025-27607 class dep hijacking.
        """
        cmd = [
            sys.executable, "-m", "pip", "install",
            specifier, "--dry-run", "--quiet",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            if proc.returncode != 0:
                logger.warning(
                    "pip dry-run for '%s' returned non-zero: %s",
                    specifier,
                    stderr.decode("utf-8", errors="replace")[:300],
                )
        except asyncio.TimeoutError:
            proc.kill()
            logger.warning("pip dry-run timed out for '%s' — skipping check", specifier)
        except Exception as exc:
            logger.warning("pip dry-run failed for '%s': %s", specifier, exc)

    # ─── Misc helpers ─────────────────────────────────────────────────────────

    async def _load_plugin_personas(self, module: Any, manifest: PluginManifest) -> list[str]:
        """Load ``personas.yaml`` from plugin module directory."""
        module_dir = Path(module.__file__).parent
        personas_file = module_dir / "personas.yaml"

        if not personas_file.exists():
            logger.debug("No personas.yaml found in plugin '%s'.", manifest.name)
            return []

        try:
            persona_data = yaml.safe_load(personas_file.read_text())
        except yaml.YAMLError as exc:
            logger.warning(
                "Failed to parse personas.yaml for plugin '%s': %s", manifest.name, exc
            )
            return []

        if not isinstance(persona_data, list):
            logger.warning("personas.yaml for '%s' must be a YAML list.", manifest.name)
            return []

        loaded: list[str] = []
        for persona_def in persona_data:
            if not isinstance(persona_def, dict) or "name" not in persona_def:
                continue
            try:
                await self._persona_manager.create_from_dict(persona_def, source="plugin")
                loaded.append(persona_def["name"])
                logger.debug(
                    "Loaded persona '%s' from plugin '%s'",
                    persona_def["name"],
                    manifest.name,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load persona '%s' from plugin '%s': %s",
                    persona_def.get("name", "?"),
                    manifest.name,
                    exc,
                )
        return loaded

    async def _compute_package_checksum(self, module_name: str) -> str:
        """Compute SHA-256 of the installed ``__init__.py`` as a tamper fingerprint."""
        try:
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                content = Path(spec.origin).read_bytes()
                return hashlib.sha256(content).hexdigest()
        except Exception as exc:
            logger.debug("Could not compute checksum for '%s': %s", module_name, exc)
        return ""

    async def _fetch_pypi_metadata(self, client: httpx.AsyncClient, package_name: str) -> dict:
        """Fetch package metadata from the PyPI JSON API."""
        url = f"{_PYPI_JSON_API}/{package_name}/json"
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        info = data.get("info", {})

        homepage = ""
        project_urls: dict = info.get("project_urls") or {}
        for key in ("Homepage", "Repository", "Source", "Documentation"):
            if key in project_urls:
                homepage = project_urls[key]
                break

        return {
            "version": info.get("version", ""),
            "description": info.get("summary", ""),
            "author": info.get("author", ""),
            "homepage": homepage or info.get("home_page", ""),
            "downloads_last_month": 0,  # requires pypistats.org — not implemented in v1
        }

    async def _persist_state(self) -> None:
        """Write current installed state to ``~/.nexus/plugins/installed.json``."""
        state = {
            name: manifest.model_dump()
            for name, manifest in self._installed.items()
        }
        try:
            self._state_file.write_text(json.dumps(state, indent=2, default=str))
        except OSError as exc:
            logger.error(
                "Could not persist plugin state to %s: %s", self._state_file, exc
            )


# ─── Module-level helpers ─────────────────────────────────────────────────────


def _normalize_package_name(package_name: str, version: Optional[str]) -> str:
    """Ensure package has ``nexus-plugin-`` prefix and append version specifier.

    Examples::

        _normalize_package_name("weather", "1.0.0")           → "nexus-plugin-weather==1.0.0"
        _normalize_package_name("nexus-plugin-slack", None)   → "nexus-plugin-slack"
        _normalize_package_name("slack==2.0.0", None)         → "nexus-plugin-slack==2.0.0"
    """
    specifier_match = re.search(r"([>=<!~]{1,2}[\d.].*)$", package_name)
    inline_specifier = specifier_match.group(1) if specifier_match else ""
    base_name = (
        package_name[: specifier_match.start()].strip()
        if specifier_match
        else package_name.strip()
    )

    if not base_name.startswith(_PACKAGE_PREFIX):
        base_name = f"{_PACKAGE_PREFIX}{base_name}"

    specifier = inline_specifier or (f"=={version}" if version else "")
    return f"{base_name}{specifier}"


def _pip_specifier_to_module_name(pip_specifier: str) -> str:
    """Convert a pip specifier to an importable module name.

    Examples::

        "nexus-plugin-weather==1.0.0"  → "nexus_plugin_weather"
        "nexus-plugin-github-tools"    → "nexus_plugin_github_tools"
    """
    base = re.split(r"[>=<!~\[]", pip_specifier)[0].strip()
    return base.replace("-", "_")


def _base_name(pip_specifier: str) -> str:
    """Extract bare package name from a pip specifier (strips version)."""
    return re.split(r"[>=<!~\[]", pip_specifier)[0].strip()


def _resolve_tool_function(module: Any, tool_name: str) -> Any:
    """Find a tool function in a plugin module by name.

    Search order:
        1. ``module.tools.<tool_name>``  (conventional layout)
        2. ``module.<tool_name>``        (flat layout)

    Raises:
        PluginManifestError: Function not found in either location.
    """
    # 1. Conventional: tools submodule
    tools_submodule = getattr(module, "tools", None)
    if tools_submodule is not None:
        func = getattr(tools_submodule, tool_name, None)
        if func is not None and callable(func):
            return func

    # 2. Flat layout
    func = getattr(module, tool_name, None)
    if func is not None and callable(func):
        return func

    raise PluginManifestError(
        f"Tool function '{tool_name}' not found in plugin module '{module.__name__}'. "
        f"It should be defined in {module.__name__}.tools or {module.__name__} directly, "
        f"decorated with @nexus_plugin_tool."
    )


def _parse_pypi_search_html(html: str) -> list[str]:
    """Extract package names from PyPI search result HTML.

    Parses ``<span class="package-snippet__name">...</span>`` elements.
    """
    pattern = re.compile(
        r'<span[^>]+class="package-snippet__name"[^>]*>\s*([a-zA-Z0-9_\-\.]+)\s*</span>'
    )
    return pattern.findall(html)


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]
