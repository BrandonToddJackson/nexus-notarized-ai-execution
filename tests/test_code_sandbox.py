"""Phase 30 — Code sandbox tests: Python execution, forbidden imports, timeout, JS, output capture."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from nexus.tools.sandbox_v2 import CodeSandbox
from nexus.exceptions import SandboxError


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_process(stdout=b"", stderr=b"", returncode=0):
    """Create a mock async subprocess."""
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    return proc


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_python_success(test_config):
    """Simple print code executes and captures stdout."""
    sandbox = CodeSandbox(test_config)
    mock_proc = _mock_process(stdout=b"hello world\n", stderr=b"", returncode=0)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        result = await sandbox.execute_python("print('hello world')")

    assert result["exit_code"] == 0
    assert "hello world" in result["stdout"]


@pytest.mark.asyncio
async def test_execute_python_forbidden_import(test_config):
    """Code with forbidden import raises SandboxError before subprocess."""
    sandbox = CodeSandbox(test_config)

    with pytest.raises(SandboxError, match="Forbidden import"):
        await sandbox.execute_python("import subprocess")


@pytest.mark.asyncio
async def test_execute_python_timeout(test_config):
    """Timeout raises SandboxError."""
    sandbox = CodeSandbox(test_config)
    mock_proc = MagicMock()
    mock_proc.kill = MagicMock()
    mock_proc.returncode = None

    call_count = 0

    async def _communicate(*a, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise asyncio.TimeoutError()
        return (b"", b"")

    mock_proc.communicate = _communicate

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        with pytest.raises(SandboxError, match="timed out"):
            # Use only allowed imports so AST validation passes
            await sandbox.execute_python("x = 1", timeout=1)


@pytest.mark.asyncio
async def test_execute_javascript_success(test_config):
    """JavaScript execution captures stdout."""
    sandbox = CodeSandbox(test_config)
    mock_proc = _mock_process(stdout=b"hi\n", stderr=b"", returncode=0)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        with patch("shutil.which", return_value="/usr/local/bin/node"):
            result = await sandbox.execute_javascript("console.log('hi')")

    assert result["exit_code"] == 0
    assert "hi" in result["stdout"]


@pytest.mark.asyncio
async def test_sandbox_output_captured(test_config):
    """Both stdout and stderr are captured from subprocess."""
    sandbox = CodeSandbox(test_config)
    mock_proc = _mock_process(
        stdout=b"output line\n",
        stderr=b"warning line\n",
        returncode=1,
    )

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        result = await sandbox.execute_python("print('output line')\nimport sys; sys.stderr.write('warning line\\n')")

    assert "output line" in result["stdout"]
    assert "warning line" in result["stderr"]
    assert result["exit_code"] == 1
