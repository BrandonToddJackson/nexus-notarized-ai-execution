"""Code Sandbox v2 — subprocess-based Python and JavaScript execution.

Runs user code in isolated temp dirs with import validation, memory limits,
optional pip/npm install, structured I/O, and output format control.
"""

import ast
import asyncio
import json
import shutil
from asyncio.subprocess import PIPE
from pathlib import Path
from uuid import uuid4

from nexus.config import config as _default_config
from nexus.config import NexusConfig
from nexus.exceptions import SandboxError
from nexus.types import ToolDefinition, RiskLevel
from nexus.tools.plugin import _registered_tools


class CodeSandbox:
    """Subprocess-based code execution sandbox for Python and JavaScript."""

    def __init__(self, config: NexusConfig) -> None:
        self._config = config

    # ──────────────────────────────────────────────────────────────────────
    # Python
    # ──────────────────────────────────────────────────────────────────────

    async def execute_python(
        self,
        code: str,
        stdin: str = "",
        timeout: int | None = None,
        max_memory_mb: int | None = None,
        input_data: object = None,
        packages: list[str] | None = None,
        environment_variables: dict[str, str] | None = None,
        allow_network: bool = False,
        language_version: str | None = None,
        output_format: str = "auto",
        max_output_kb: int = 1024,
    ) -> dict:
        config = self._config

        # 1. Import validation
        allowed = set(config.sandbox_allowed_imports)
        # Extend with pip package names if pip install is enabled
        if packages and config.sandbox_allow_pip_install:
            for pkg in packages:
                norm = pkg.split("==")[0].replace("-", "_")
                allowed.add(norm)

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise SandboxError(f"Syntax error in code: {exc}") from exc

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in allowed:
                        raise SandboxError(
                            f"Forbidden import: '{alias.name}' Allowed: {sorted(allowed)}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top not in allowed:
                        raise SandboxError(
                            f"Forbidden import: '{node.module}' Allowed: {sorted(allowed)}"
                        )

        # 2. Create temp dir
        sandbox_dir = Path(f"/tmp/nexus_sandbox_{uuid4().hex}/")
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        python_bin: Path | None = None

        try:
            # 3. Optional pip install
            if packages and config.sandbox_allow_pip_install:
                venv_dir = sandbox_dir / "venv"
                venv_proc = await asyncio.create_subprocess_exec(
                    "python3", "-m", "venv", str(venv_dir),
                    stdout=PIPE, stderr=PIPE,
                )
                try:
                    await asyncio.wait_for(venv_proc.communicate(), timeout=30)
                except asyncio.TimeoutError:
                    venv_proc.kill()
                    raise SandboxError("venv creation timed out")

                pip_bin = venv_dir / "bin" / "pip3"
                for package in packages:
                    pip_proc = await asyncio.create_subprocess_exec(
                        str(pip_bin), "install", "--quiet", "--no-input", package,
                        stdout=PIPE, stderr=PIPE,
                    )
                    try:
                        _, err = await asyncio.wait_for(
                            pip_proc.communicate(),
                            timeout=config.sandbox_pip_install_timeout,
                        )
                    except asyncio.TimeoutError:
                        pip_proc.kill()
                        raise SandboxError(
                            f"pip install failed for '{package}': timed out"
                        )
                    if pip_proc.returncode != 0:
                        raise SandboxError(
                            f"pip install failed for '{package}': {err.decode(errors='replace').strip()}"
                        )
                python_bin = venv_dir / "bin" / "python3"

            # 4. Write files
            script_path = sandbox_dir / "script.py"
            wrapper_path = sandbox_dir / "_wrapper.py"

            script_path.write_text(code)

            memory_mb = max_memory_mb or config.sandbox_max_memory_mb
            wrapper_src = (
                "import resource, runpy, os, json as _json\n"
                f"_mb = {memory_mb}\n"
                "try:\n"
                "    resource.setrlimit(resource.RLIMIT_AS, (_mb * 1024 * 1024,) * 2)\n"
                "except (ValueError, resource.error):\n"
                "    pass\n"
                "_raw = os.environ.get('NEXUS_INPUT')\n"
                "__nexus_input__ = _json.loads(_raw) if _raw else None\n"
                f"runpy.run_path({str(script_path)!r}, init_globals={{'__nexus_input__': __nexus_input__}})\n"
            )
            wrapper_path.write_text(wrapper_src)

            # 5. Build env dict
            import os
            env: dict[str, str] = {
                "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
                "HOME": "/tmp",
                "TMPDIR": str(sandbox_dir),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            }
            if input_data is not None:
                env["NEXUS_INPUT"] = json.dumps(input_data)
            if not allow_network:
                proxy = "http://127.0.0.1:0"
                env["http_proxy"] = proxy
                env["https_proxy"] = proxy
                env["HTTP_PROXY"] = proxy
                env["HTTPS_PROXY"] = proxy
            if environment_variables:
                for k, v in environment_variables.items():
                    if not k.startswith("NEXUS_"):
                        env[k] = v

            # 6. Select Python binary
            if python_bin is not None:
                python_exec = str(python_bin)
            else:
                python_exec = f"python{language_version or '3'}"
                try:
                    if shutil.which(python_exec) is None:
                        python_exec = "python3"
                except Exception:
                    python_exec = "python3"

            # 7. Execute
            proc = await asyncio.create_subprocess_exec(
                python_exec, str(wrapper_path),
                stdin=PIPE, stdout=PIPE, stderr=PIPE,
                env=env, cwd=str(sandbox_dir),
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(input=stdin.encode() if stdin else b""),
                    timeout=timeout or config.sandbox_max_execution_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.communicate()
                except Exception:
                    pass
                raise SandboxError(
                    f"Execution timed out after {timeout or config.sandbox_max_execution_seconds}s"
                )

        finally:
            # 8. Cleanup
            shutil.rmtree(sandbox_dir, ignore_errors=True)

        # 9. Truncate output
        max_bytes = (max_output_kb or config.sandbox_max_output_kb) * 1024
        truncated = len(stdout_bytes) > max_bytes
        stdout_bytes = stdout_bytes[:max_bytes]
        stdout_str = stdout_bytes.decode(errors="replace")
        stderr_str = stderr_bytes.decode(errors="replace")
        exit_code = proc.returncode if proc.returncode is not None else -1

        # 10. Format output
        return _format_output(stdout_str, stderr_str, exit_code, truncated, output_format)

    # ──────────────────────────────────────────────────────────────────────
    # JavaScript
    # ──────────────────────────────────────────────────────────────────────

    async def execute_javascript(
        self,
        code: str,
        stdin: str = "",
        timeout: int | None = None,
        input_data: object = None,
        packages: list[str] | None = None,
        environment_variables: dict[str, str] | None = None,
        allow_network: bool = False,
        module_system: str = "commonjs",
        node_version: str | None = None,
        output_format: str = "auto",
        max_output_kb: int = 1024,
    ) -> dict:
        config = self._config

        # 1. No static import validation for JS.

        # 2. Create temp dir
        sandbox_dir = Path(f"/tmp/nexus_sandbox_{uuid4().hex}/")
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 3. Optional npm install
            if packages and config.sandbox_allow_npm_install:
                pkg_json = sandbox_dir / "package.json"
                pkg_json.write_text(json.dumps({"type": "commonjs"}))

                npm_proc = await asyncio.create_subprocess_exec(
                    "npm", "install",
                    "--prefix", str(sandbox_dir),
                    "--quiet", "--no-audit", "--no-fund",
                    *packages,
                    stdout=PIPE, stderr=PIPE,
                )
                try:
                    _, err = await asyncio.wait_for(
                        npm_proc.communicate(),
                        timeout=config.sandbox_npm_install_timeout,
                    )
                except asyncio.TimeoutError:
                    npm_proc.kill()
                    raise SandboxError("npm install failed: timed out")
                if npm_proc.returncode != 0:
                    raise SandboxError(
                        f"npm install failed: {err.decode(errors='replace').strip()}"
                    )

            # 4. Write code file
            is_esm = module_system == "esm"
            ext = ".mjs" if is_esm else ".cjs"
            preamble = "" if is_esm else "const __nexus_require = require; "
            code_file = sandbox_dir / f"script{ext}"
            code_file.write_text(preamble + code)

            # 5. Build env dict
            import os
            env: dict[str, str] = {
                "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
                "HOME": "/tmp",
                "NODE_PATH": str(sandbox_dir / "node_modules"),
            }
            if input_data is not None:
                env["NEXUS_INPUT"] = json.dumps(input_data)
            if not allow_network:
                proxy = "http://127.0.0.1:0"
                env["http_proxy"] = proxy
                env["https_proxy"] = proxy
                env["HTTP_PROXY"] = proxy
                env["HTTPS_PROXY"] = proxy
            if environment_variables:
                for k, v in environment_variables.items():
                    if not k.startswith("NEXUS_"):
                        env[k] = v

            # 6. Node args
            # Note: --input-type=module is for stdin piping only.
            # For file execution, .mjs extension is sufficient for ESM.
            node_args = ["--max-old-space-size=256"]

            # 7. Select node binary
            if node_version:
                node_exec = f"node{node_version}"
                if shutil.which(node_exec) is None:
                    node_exec = "node"
            else:
                node_exec = "node"
            try:
                if shutil.which(node_exec) is None:
                    node_exec = "node"
            except Exception:
                pass

            # Execute
            proc = await asyncio.create_subprocess_exec(
                node_exec, *node_args, str(code_file),
                stdin=PIPE, stdout=PIPE, stderr=PIPE,
                env=env, cwd=str(sandbox_dir),
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(input=stdin.encode() if stdin else b""),
                    timeout=timeout or config.sandbox_max_execution_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.communicate()
                except Exception:
                    pass
                raise SandboxError(
                    f"Execution timed out after {timeout or config.sandbox_max_execution_seconds}s"
                )

        finally:
            # Cleanup
            shutil.rmtree(sandbox_dir, ignore_errors=True)

        # 9. Truncate output
        max_bytes = (max_output_kb or config.sandbox_max_output_kb) * 1024
        truncated = len(stdout_bytes) > max_bytes
        stdout_bytes = stdout_bytes[:max_bytes]
        stdout_str = stdout_bytes.decode(errors="replace")
        stderr_str = stderr_bytes.decode(errors="replace")
        exit_code = proc.returncode if proc.returncode is not None else -1

        # 10. Format output
        return _format_output(stdout_str, stderr_str, exit_code, truncated, output_format)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_output(
    stdout_str: str,
    stderr_str: str,
    exit_code: int,
    truncated: bool,
    output_format: str,
) -> dict:
    if output_format == "text":
        return {
            "stdout": stdout_str,
            "stderr": stderr_str,
            "exit_code": exit_code,
            "truncated": truncated,
        }

    # Try to parse JSON
    def _try_json():
        stripped = stdout_str.strip()
        if not stripped:
            return None, False
        try:
            return json.loads(stripped), True
        except json.JSONDecodeError:
            return None, False

    parsed, ok = _try_json()

    if output_format == "json":
        if ok:
            return {
                "result": parsed,
                "stderr": stderr_str,
                "exit_code": exit_code,
                "truncated": truncated,
            }
        else:
            return {
                "error": "Output is not valid JSON",
                "stdout": stdout_str,
                "stderr": stderr_str,
                "exit_code": exit_code,
                "truncated": truncated,
            }

    # output_format == "auto"
    if ok:
        return {
            "result": parsed,
            "stderr": stderr_str,
            "exit_code": exit_code,
            "truncated": truncated,
        }
    return {
        "stdout": stdout_str,
        "stderr": stderr_str,
        "exit_code": exit_code,
        "truncated": truncated,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Module-level registration
# ──────────────────────────────────────────────────────────────────────────────

sandbox = CodeSandbox(_default_config)

code_execute_python = sandbox.execute_python
code_execute_javascript = sandbox.execute_javascript

_PYTHON_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python 3 source code to execute"},
        "stdin": {"type": "string", "description": "Text to pass as stdin"},
        "timeout": {"type": "integer", "description": "Execution timeout in seconds"},
        "max_memory_mb": {"type": "integer", "description": "Memory limit in MB"},
        "input_data": {"description": "JSON-serialisable data injected as NEXUS_INPUT env var"},
        "packages": {
            "type": "array",
            "items": {"type": "string"},
            "description": "pip packages to install (requires sandbox_allow_pip_install=true)",
        },
        "environment_variables": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Extra environment variables (NEXUS_* keys are ignored)",
        },
        "allow_network": {"type": "boolean", "description": "Allow outbound network access"},
        "language_version": {"type": "string", "description": "Python version suffix (e.g. '3.11')"},
        "output_format": {
            "type": "string",
            "enum": ["auto", "json", "text"],
            "description": "How to parse stdout: auto tries JSON, text returns raw",
        },
        "max_output_kb": {"type": "integer", "description": "Max stdout bytes (KB) before truncation"},
    },
    "required": ["code"],
}

_JS_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "JavaScript source code to execute"},
        "stdin": {"type": "string", "description": "Text to pass as stdin"},
        "timeout": {"type": "integer", "description": "Execution timeout in seconds"},
        "input_data": {"description": "JSON-serialisable data injected as NEXUS_INPUT env var"},
        "packages": {
            "type": "array",
            "items": {"type": "string"},
            "description": "npm packages to install (requires sandbox_allow_npm_install=true)",
        },
        "environment_variables": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Extra environment variables (NEXUS_* keys are ignored)",
        },
        "allow_network": {"type": "boolean", "description": "Allow outbound network access"},
        "module_system": {
            "type": "string",
            "enum": ["commonjs", "esm"],
            "description": "Module system: commonjs (require) or esm (import)",
        },
        "node_version": {"type": "string", "description": "Node version suffix"},
        "output_format": {
            "type": "string",
            "enum": ["auto", "json", "text"],
            "description": "How to parse stdout: auto tries JSON, text returns raw",
        },
        "max_output_kb": {"type": "integer", "description": "Max stdout bytes (KB) before truncation"},
    },
    "required": ["code"],
}

_registered_tools["code_execute_python"] = (
    ToolDefinition(
        name="code_execute_python",
        description=(
            "Execute Python 3 code in an isolated sandbox. "
            "Supports stdin, input_data (NEXUS_INPUT env), memory limits, optional pip install, "
            "and structured output (auto/json/text)."
        ),
        parameters=_PYTHON_SCHEMA,
        risk_level=RiskLevel.MEDIUM,
        resource_pattern="code:*",
        timeout_seconds=120,
    ),
    lambda **p: sandbox.execute_python(**p),
)

_registered_tools["code_execute_javascript"] = (
    ToolDefinition(
        name="code_execute_javascript",
        description=(
            "Execute Node.js JavaScript code in an isolated sandbox. "
            "Supports stdin, input_data (NEXUS_INPUT env), optional npm install, "
            "commonjs/esm module systems, and structured output (auto/json/text)."
        ),
        parameters=_JS_SCHEMA,
        risk_level=RiskLevel.MEDIUM,
        resource_pattern="code:*",
        timeout_seconds=120,
    ),
    lambda **p: sandbox.execute_javascript(**p),
)
