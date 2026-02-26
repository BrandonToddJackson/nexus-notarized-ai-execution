"""Phase 30 — HTTP request tool tests: GET, POST, JMESPath, errors, retry."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from nexus.tools.builtin.http_request import http_request
from nexus.exceptions import ToolError


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_response(status_code=200, body=None, headers=None, url="https://api.example.com/data"):
    """Create a mock httpx Response-like object + content bytes."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {"content-type": "application/json"}
    resp.url = url
    resp.reason_phrase = "OK" if status_code < 400 else "Error"

    content = json.dumps(body).encode() if body is not None else b""
    return resp, content


def _patch_execute_single(responses):
    """Patch _execute_single to return a sequence of (response, content, elapsed_ms)."""
    call_count = 0

    async def _fake_execute_single(method, url, query_params, client_kwargs, request_kwargs, size_limit_bytes):
        nonlocal call_count
        resp, content = responses[call_count]
        call_count += 1
        return resp, content, 42  # 42ms elapsed

    return patch(
        "nexus.tools.builtin.http_request._execute_single",
        side_effect=_fake_execute_single,
    ), lambda: call_count


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_request_success():
    """GET 200 returns status_code, body, url, elapsed_ms."""
    resp, content = _mock_response(200, body={"key": "value"})
    patcher, _ = _patch_execute_single([(resp, content)])

    with patcher:
        result = await http_request(url="https://api.example.com/data")

    assert result["status_code"] == 200
    assert result["body"] == {"key": "value"}
    assert "elapsed_ms" in result


@pytest.mark.asyncio
async def test_post_request_with_json_body():
    """POST with JSON body returns created resource."""
    resp, content = _mock_response(
        201,
        body={"id": 1, "name": "test"},
        url="https://api.example.com/items",
    )
    patcher, _ = _patch_execute_single([(resp, content)])

    with patcher:
        result = await http_request(
            url="https://api.example.com/items",
            method="POST",
            body={"name": "test"},
        )

    assert result["status_code"] == 201
    assert result["body"]["id"] == 1


@pytest.mark.asyncio
async def test_jmespath_extraction():
    """response_path extracts nested value via JMESPath."""
    resp, content = _mock_response(200, body={"data": {"name": "alice"}})
    patcher, _ = _patch_execute_single([(resp, content)])

    with patcher:
        result = await http_request(
            url="https://api.example.com/users",
            response_path="data.name",
        )

    assert result["body"] == "alice"


@pytest.mark.asyncio
async def test_error_response_returns_error_dict():
    """GET 404 returns dict with error=True."""
    resp, content = _mock_response(404, body={"detail": "not found"})
    patcher, _ = _patch_execute_single([(resp, content)])

    with patcher:
        result = await http_request(url="https://api.example.com/missing")

    assert result["error"] is True
    assert result["status_code"] == 404


@pytest.mark.asyncio
async def test_retry_on_server_error():
    """First request 500, second 200 — retries and succeeds."""
    resp_500, content_500 = _mock_response(500, body={"error": "server"})
    resp_200, content_200 = _mock_response(200, body={"ok": True})
    patcher, get_count = _patch_execute_single([
        (resp_500, content_500),
        (resp_200, content_200),
    ])

    with patcher:
        result = await http_request(
            url="https://api.example.com/flaky",
            max_retries=1,
        )

    assert result["status_code"] == 200
    assert result["body"] == {"ok": True}
    assert get_count() == 2
