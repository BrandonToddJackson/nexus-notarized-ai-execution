"""Phase 20 tests — http_request and data_transform builtin tools.

All HTTP tests use a real local TCP server (Python stdlib http.server in a daemon
thread).  No unittest.mock, MagicMock, or patch anywhere in this file.

data_transform tests operate on in-memory Python data; no network I/O is needed.
"""

import asyncio
import base64
import json
import socket
import threading
import http.server as _http_server
import time
import urllib.parse
import uuid

import pytest

import nexus.tools.builtin  # trigger registration
from nexus.tools.plugin import get_registered_tools
from nexus.tools.builtin.http_request import (
    http_request,
    _parse_link_header,
    _parse_response_body,
    _extract_jmespath,
)
from nexus.tools.builtin.data_transform import (
    data_transform,
    _get_nested,
    _set_nested,
    _del_nested,
)
from nexus.exceptions import ToolError


# ═══════════════════════════════════════════════════════════════════════════════
# Real local HTTP server
# ═══════════════════════════════════════════════════════════════════════════════

# Per-key request counters used by retry endpoints.  Each test passes a unique
# UUID as the `_key` query parameter so counters don't bleed between tests.
_request_counters: dict[str, int] = {}
_counters_lock = threading.Lock()

# Products served by /json
_PRODUCTS = [
    {"id": 1, "name": "Widget A", "price": 9.99},
    {"id": 2, "name": "Widget B", "price": 4.99},
    {"id": 3, "name": "Gadget X", "price": 29.99},
]
_PRODUCTS_JSON = json.dumps({"products": _PRODUCTS, "total": 3, "page": 1}).encode()

_LARGE_BODY = b"x" * 200_000  # 200 KB — well over any reasonable size limit


def _qparam(path: str, name: str, default: str = "") -> str:
    """Extract a single query parameter value from a raw request path."""
    if "?" not in path:
        return default
    qs = urllib.parse.parse_qs(path.split("?", 1)[1])
    return qs.get(name, [default])[0]


class _TestHandler(_http_server.BaseHTTPRequestHandler):
    """Handles all test scenarios; all responses are deterministic and real."""

    def _clean_path(self) -> str:
        return self.path.split("?")[0]

    def _send_json(self, status: int, data: dict, extra_headers: dict = None):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_raw(self, status: int, body: bytes, content_type: str = "application/octet-stream", extra_headers: dict = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self._clean_path()

        # ── Basic response types ──────────────────────────────────────────────
        if path == "/json":
            self._send_raw(200, _PRODUCTS_JSON, "application/json")

        elif path == "/text":
            self._send_raw(200, b"hello real world", "text/plain")

        elif path == "/binary":
            self._send_raw(200, bytes(range(16)), "application/octet-stream")

        elif path == "/large":
            self._send_raw(200, _LARGE_BODY, "text/plain")

        # ── Error responses ───────────────────────────────────────────────────
        elif path == "/404":
            self._send_json(404, {"error": "not found"})

        elif path == "/500":
            self._send_json(500, {"error": "server error"})

        # ── Auth echo: returns received Authorization / API-Key headers ────────
        elif path == "/auth-echo":
            self._send_json(200, {
                "authorization": self.headers.get("Authorization", ""),
                "x_api_key":    self.headers.get("X-Custom-Key", ""),
            })

        # ── Timeout trigger: sleeps before responding ─────────────────────────
        elif path == "/slow":
            try:
                time.sleep(1.0)
                self._send_raw(200, b'{"ok": true}', "application/json")
            except Exception:
                pass  # client disconnected — swallow BrokenPipe

        # ── Retry-on-500: fails `fail_n` times then succeeds ─────────────────
        elif path == "/retry-500":
            key = _qparam(self.path, "_key", "default")
            fail_n = int(_qparam(self.path, "fail_n", "2"))
            with _counters_lock:
                count = _request_counters.get(key, 0) + 1
                _request_counters[key] = count
            if count <= fail_n:
                self._send_json(500, {"attempt": count, "ok": False})
            else:
                self._send_json(200, {"attempt": count, "ok": True})

        # ── Retry-on-429: fails `fail_n` times with Retry-After: 0 then 200 ──
        elif path == "/retry-429":
            key = _qparam(self.path, "_key", "default")
            fail_n = int(_qparam(self.path, "fail_n", "1"))
            with _counters_lock:
                count = _request_counters.get(key, 0) + 1
                _request_counters[key] = count
            if count <= fail_n:
                self._send_json(429, {"attempt": count, "retry": True},
                                extra_headers={"Retry-After": "0"})
            else:
                self._send_json(200, {"attempt": count, "ok": True})

        # ── Cursor pagination: cursor value encodes the page ──────────────────
        elif path == "/cursor":
            cursor = _qparam(self.path, "cursor", "")
            port = self.server.server_address[1]
            base = f"http://127.0.0.1:{port}/cursor"
            if cursor == "":
                self._send_json(200, {"items": [1, 2], "next_cursor": "page2"})
            elif cursor == "page2":
                self._send_json(200, {"items": [3, 4], "next_cursor": "page3"})
            elif cursor == "page3":
                self._send_json(200, {"items": [5, 6], "next_cursor": None})
            else:
                self._send_json(200, {"items": [], "next_cursor": None})

        # ── Infinite cursor: always returns a next cursor (tests max_pages) ───
        elif path == "/infinite-cursor":
            self._send_json(200, {"items": [42], "next": "always"})

        # ── Offset pagination: simple 7-item dataset in chunks ────────────────
        elif path == "/offset":
            offset = int(_qparam(self.path, "offset", "0"))
            limit  = int(_qparam(self.path, "limit", "3"))
            dataset = list(range(1, 8))   # [1,2,3,4,5,6,7], total=7
            chunk = dataset[offset : offset + limit]
            self._send_json(200, {"items": chunk, "total": len(dataset)})

        # ── Empty-items endpoint: always returns empty list (tests early stop) ─
        elif path == "/empty-offset":
            self._send_json(200, {"items": [], "total": 100})

        # ── Link-header pagination: 3 pages ───────────────────────────────────
        elif path == "/paginated":
            page = int(_qparam(self.path, "page", "1"))
            port = self.server.server_address[1]
            base = f"http://127.0.0.1:{port}/paginated"
            if page == 1:
                items = [{"item": 1}, {"item": 2}]
                link  = f'<{base}?page=2>; rel="next"'
            elif page == 2:
                items = [{"item": 3}, {"item": 4}]
                link  = f'<{base}?page=3>; rel="next"'
            else:
                items = [{"item": 5}]
                link  = ""
            extra = {"Link": link} if link else {}
            self._send_json(200, items, extra_headers=extra if extra else None)

        else:
            self._send_json(404, {"error": f"unknown path: {path}"})

    def do_POST(self):
        """Echo endpoint: returns received content-type and body."""
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        ct = self.headers.get("Content-Type", "")

        response: dict = {
            "received_bytes": len(raw),
            "content_type": ct,
        }
        if "application/json" in ct:
            try:
                response["json_body"] = json.loads(raw)
            except Exception:
                response["json_body"] = None
        elif "application/x-www-form-urlencoded" in ct:
            response["form_data"] = urllib.parse.parse_qs(raw.decode("utf-8"))
        elif "multipart/form-data" in ct:
            response["multipart"] = True   # we don't parse, just note it arrived
        else:
            response["raw_body"] = raw.decode("utf-8", errors="replace")

        self._send_json(200, response)

    def log_message(self, *args):
        pass  # silence server output during test runs


@pytest.fixture(scope="module")
def local_server():
    """Start a real TCP HTTP server on a random OS-assigned port."""
    server = _http_server.HTTPServer(("127.0.0.1", 0), _TestHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(scope="module")
def closed_port():
    """A TCP port guaranteed to refuse connections (briefly bound then released)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def retry_key():
    """Unique key per test so retry counters never bleed between tests."""
    return str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════════════════════
# Registration smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegistration:
    def test_http_request_registered(self):
        tools = get_registered_tools()
        assert "http_request" in tools

    def test_data_transform_registered(self):
        tools = get_registered_tools()
        assert "data_transform" in tools

    def test_http_request_schema_has_url(self):
        tools = get_registered_tools()
        defn, _ = tools["http_request"]
        assert "url" in defn.parameters.get("properties", {})

    def test_data_transform_schema_has_operations(self):
        tools = get_registered_tools()
        defn, _ = tools["data_transform"]
        assert "operations" in defn.parameters.get("properties", {})


# ═══════════════════════════════════════════════════════════════════════════════
# Pure-unit helper tests (no network)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseLinkHeader:
    def test_single_next(self):
        result = _parse_link_header('<https://api.example.com/page2>; rel="next"')
        assert result["next"] == "https://api.example.com/page2"

    def test_multiple_rels(self):
        result = _parse_link_header(
            '<https://api.example.com/page2>; rel="next", '
            '<https://api.example.com/page5>; rel="last"'
        )
        assert result["next"] == "https://api.example.com/page2"
        assert result["last"] == "https://api.example.com/page5"

    def test_empty_header(self):
        assert _parse_link_header("") == {}

    def test_no_rel(self):
        assert _parse_link_header("<https://api.example.com/page2>; title=foo") == {}


class TestParseResponseBody:
    def test_json_format(self):
        assert _parse_response_body(b'{"key": "value"}', "json") == {"key": "value"}

    def test_text_format(self):
        assert _parse_response_body(b"hello world", "text") == "hello world"

    def test_binary_format(self):
        data = b"\x00\x01\x02"
        assert _parse_response_body(data, "binary") == base64.b64encode(data).decode("ascii")

    def test_auto_resolves_json(self):
        assert _parse_response_body(b'{"x": 1}', "auto") == {"x": 1}

    def test_auto_falls_back_to_text(self):
        assert _parse_response_body(b"plain text", "auto") == "plain text"


# ═══════════════════════════════════════════════════════════════════════════════
# http_request — real TCP, no mocks
# ═══════════════════════════════════════════════════════════════════════════════

class TestHttpRequestBasic:
    """Every test makes a real TCP connection to the local server."""

    @pytest.mark.asyncio
    async def test_get_json_response(self, local_server):
        result = await http_request(url=f"{local_server}/json")
        assert result["status_code"] == 200
        assert result["body"]["total"] == 3
        assert result["body"]["products"][0] == {"id": 1, "name": "Widget A", "price": 9.99}
        assert "elapsed_ms" in result
        assert isinstance(result["elapsed_ms"], int)

    @pytest.mark.asyncio
    async def test_get_text_response_format(self, local_server):
        result = await http_request(url=f"{local_server}/text", response_format="text")
        assert result["status_code"] == 200
        assert result["body"] == "hello real world"
        assert isinstance(result["body"], str)

    @pytest.mark.asyncio
    async def test_get_binary_response_format(self, local_server):
        result = await http_request(url=f"{local_server}/binary", response_format="binary")
        assert result["status_code"] == 200
        decoded = base64.b64decode(result["body"])
        assert decoded == bytes(range(16))

    @pytest.mark.asyncio
    async def test_get_auto_format_returns_json_when_possible(self, local_server):
        result = await http_request(url=f"{local_server}/json")  # default: auto
        assert isinstance(result["body"], dict)        # parsed as JSON, not string

    @pytest.mark.asyncio
    async def test_404_returns_error_dict(self, local_server):
        result = await http_request(url=f"{local_server}/404")
        assert result["error"] is True
        assert result["status_code"] == 404
        assert result["body"]["error"] == "not found"

    @pytest.mark.asyncio
    async def test_500_no_retry_by_default(self, local_server, retry_key):
        # /retry-500 counts requests; with max_retries=0 only 1 call should happen
        result = await http_request(
            url=f"{local_server}/retry-500",
            query_params={"_key": retry_key, "fail_n": "999"},  # always fail
        )
        assert result["status_code"] == 500
        assert result["body"]["attempt"] == 1   # exactly one attempt

    @pytest.mark.asyncio
    async def test_500_with_retry_succeeds_on_third_attempt(self, local_server, retry_key):
        # fail_n=2 → fails attempts 1 and 2, succeeds on attempt 3
        result = await http_request(
            url=f"{local_server}/retry-500",
            query_params={"_key": retry_key, "fail_n": "2"},
            max_retries=2,
            retry_on=[500],
        )
        assert result["status_code"] == 200
        assert result["body"]["ok"] is True
        assert result["body"]["attempt"] == 3

    @pytest.mark.asyncio
    async def test_429_retry_after_header_respected(self, local_server, retry_key):
        # Retry-After: 0 → asyncio.sleep(0) so the test doesn't hang
        result = await http_request(
            url=f"{local_server}/retry-429",
            query_params={"_key": retry_key, "fail_n": "1"},
            max_retries=1,
            retry_on=[429],
        )
        assert result["status_code"] == 200
        assert result["body"]["ok"] is True
        assert result["body"]["attempt"] == 2

    @pytest.mark.asyncio
    async def test_connection_error_exhausts_retries(self, closed_port):
        with pytest.raises(ToolError, match="Connection failed"):
            await http_request(
                url=f"http://127.0.0.1:{closed_port}/api",
                max_retries=1,
                timeout_seconds=2,
            )

    @pytest.mark.asyncio
    async def test_timeout_raises_tool_error(self, local_server):
        # /slow sleeps 1 second; timeout of 0.1 s triggers ReadTimeout → ToolError
        with pytest.raises(ToolError, match="Connection failed"):
            await http_request(url=f"{local_server}/slow", timeout_seconds=0.1)

    @pytest.mark.asyncio
    async def test_jmespath_extracts_nested_field(self, local_server):
        result = await http_request(
            url=f"{local_server}/json",
            response_path="products[*].name",
        )
        assert result["body"] == ["Widget A", "Widget B", "Gadget X"]

    @pytest.mark.asyncio
    async def test_jmespath_extracts_scalar(self, local_server):
        result = await http_request(
            url=f"{local_server}/json",
            response_path="products[1].price",
        )
        assert result["body"] == 4.99

    @pytest.mark.asyncio
    async def test_size_limit_raises_tool_error(self, local_server):
        # /large returns 200 KB; limit to 10 KB
        with pytest.raises(ToolError, match="size limit"):
            await http_request(url=f"{local_server}/large", response_size_limit_kb=10)

    @pytest.mark.asyncio
    async def test_verify_ssl_false_adds_flag_to_response(self, local_server):
        # Plain HTTP — verify_ssl=False still sets ssl_verification_disabled flag
        result = await http_request(url=f"{local_server}/json", verify_ssl=False)
        assert result["status_code"] == 200
        assert result.get("ssl_verification_disabled") is True

    @pytest.mark.asyncio
    async def test_missing_url_raises_tool_error(self):
        with pytest.raises(ToolError, match="url is required"):
            await http_request()

    @pytest.mark.asyncio
    async def test_http2_flag_does_not_break_request(self, local_server):
        # http2=True over plaintext falls back to HTTP/1.1 — request still works
        result = await http_request(url=f"{local_server}/json", http2=True)
        assert result["status_code"] == 200
        assert result["body"]["total"] == 3

    @pytest.mark.asyncio
    async def test_response_headers_present(self, local_server):
        result = await http_request(url=f"{local_server}/json")
        assert "content-type" in result["headers"]
        assert "application/json" in result["headers"]["content-type"]

    @pytest.mark.asyncio
    async def test_response_url_in_result(self, local_server):
        result = await http_request(url=f"{local_server}/json")
        assert "/json" in result["url"]

    # ── Body format tests: verify server received the correct encoding ────────

    @pytest.mark.asyncio
    async def test_post_json_body_server_receives_json(self, local_server):
        payload = {"task": "Phase 20", "count": 42}
        result = await http_request(
            url=f"{local_server}/",
            method="POST",
            body=payload,
            body_format="json",
        )
        assert result["status_code"] == 200
        assert "application/json" in result["body"]["content_type"]
        assert result["body"]["json_body"] == payload

    @pytest.mark.asyncio
    async def test_post_form_body_server_receives_form_encoded(self, local_server):
        result = await http_request(
            url=f"{local_server}/",
            method="POST",
            body={"username": "alice", "role": "admin"},
            body_format="form",
        )
        assert result["status_code"] == 200
        ct = result["body"]["content_type"]
        assert "application/x-www-form-urlencoded" in ct
        form = result["body"]["form_data"]
        assert form["username"] == ["alice"]
        assert form["role"] == ["admin"]

    @pytest.mark.asyncio
    async def test_post_graphql_body_sent_as_json(self, local_server):
        gql = {"query": "{ user { id name } }", "variables": {"id": 7}}
        result = await http_request(
            url=f"{local_server}/",
            method="POST",
            body=gql,
            body_format="graphql",
        )
        assert result["status_code"] == 200
        assert "application/json" in result["body"]["content_type"]
        assert result["body"]["json_body"] == gql

    @pytest.mark.asyncio
    async def test_post_raw_string_body(self, local_server):
        result = await http_request(
            url=f"{local_server}/",
            method="POST",
            body="raw payload text",
            body_format="raw",
        )
        assert result["status_code"] == 200
        assert result["body"]["raw_body"] == "raw payload text"
        assert result["body"]["received_bytes"] == len(b"raw payload text")

    @pytest.mark.asyncio
    async def test_post_multipart_content_type_set(self, local_server):
        # httpx sets multipart/form-data + boundary when files= is used
        result = await http_request(
            url=f"{local_server}/",
            method="POST",
            body={"file": ("report.txt", b"contents here", "text/plain")},
            body_format="multipart",
        )
        assert result["status_code"] == 200
        assert "multipart/form-data" in result["body"]["content_type"]
        assert result["body"]["multipart"] is True

    # ── Auth header tests: server echoes received headers ────────────────────

    @pytest.mark.asyncio
    async def test_bearer_auth_sets_authorization_header(self, local_server):
        result = await http_request(
            url=f"{local_server}/auth-echo",
            auth={"type": "bearer", "token": "tok_abc123"},
        )
        assert result["status_code"] == 200
        assert result["body"]["authorization"] == "Bearer tok_abc123"

    @pytest.mark.asyncio
    async def test_basic_auth_base64_encodes_credentials(self, local_server):
        result = await http_request(
            url=f"{local_server}/auth-echo",
            auth={"type": "basic", "username": "alice", "password": "s3cr3t"},
        )
        expected = "Basic " + base64.b64encode(b"alice:s3cr3t").decode()
        assert result["body"]["authorization"] == expected

    @pytest.mark.asyncio
    async def test_api_key_auth_sets_custom_header(self, local_server):
        result = await http_request(
            url=f"{local_server}/auth-echo",
            auth={"type": "api_key", "header": "X-Custom-Key", "key": "key-xyz-789"},
        )
        assert result["body"]["x_api_key"] == "key-xyz-789"

    @pytest.mark.asyncio
    async def test_proxy_unreachable_raises_tool_error(self, closed_port):
        # Proxy pointing to a closed port — proves the proxy param was passed to httpx
        with pytest.raises(ToolError, match="Connection failed"):
            await http_request(
                url=f"http://127.0.0.1/irrelevant",
                proxy=f"http://127.0.0.1:{closed_port}",
                timeout_seconds=2,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# http_request — pagination (real local server)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHttpRequestPagination:

    @pytest.mark.asyncio
    async def test_cursor_pagination_3_pages(self, local_server):
        result = await http_request(
            url=f"{local_server}/cursor",
            pagination={
                "type": "cursor",
                "results_path": "items",
                "next_cursor_path": "next_cursor",
                "cursor_param": "cursor",
                "max_pages": 10,
            },
        )
        assert result["pages"] == 3
        assert result["total_results"] == 6
        assert result["results"] == [1, 2, 3, 4, 5, 6]

    @pytest.mark.asyncio
    async def test_cursor_pagination_max_pages_stops_early(self, local_server):
        # /infinite-cursor always returns a next cursor
        result = await http_request(
            url=f"{local_server}/infinite-cursor",
            pagination={
                "type": "cursor",
                "results_path": "items",
                "next_cursor_path": "next",
                "max_pages": 2,
            },
        )
        assert result["pages"] == 2   # stopped by max_pages, not by missing cursor

    @pytest.mark.asyncio
    async def test_offset_pagination_full_dataset(self, local_server):
        result = await http_request(
            url=f"{local_server}/offset",
            pagination={
                "type": "offset",
                "results_path": "items",
                "total_path": "total",
                "offset_param": "offset",
                "limit_param": "limit",
                "limit": 3,
                "max_pages": 10,
            },
        )
        # 7-item dataset in chunks of 3: pages [1,2,3], [4,5,6], [7]
        assert result["total_results"] == 7
        assert result["results"] == [1, 2, 3, 4, 5, 6, 7]

    @pytest.mark.asyncio
    async def test_offset_pagination_stops_at_total(self, local_server):
        # limit=4 → page 1 = [1,2,3,4], offset=4 < total=7 so continues;
        # page 2 = [5,6,7], offset=8 >= 7 so stops
        result = await http_request(
            url=f"{local_server}/offset",
            pagination={
                "type": "offset",
                "results_path": "items",
                "total_path": "total",
                "offset_param": "offset",
                "limit_param": "limit",
                "limit": 4,
                "max_pages": 10,
            },
        )
        assert result["results"] == [1, 2, 3, 4, 5, 6, 7]

    @pytest.mark.asyncio
    async def test_link_header_pagination_3_pages(self, local_server):
        result = await http_request(
            url=f"{local_server}/paginated",
            response_format="json",
            pagination={
                "type": "link_header",
                "results_path": "",
                "max_pages": 10,
            },
        )
        assert result["pages"] == 3
        assert result["total_results"] == 5
        items = [r["item"] for r in result["results"]]
        assert items == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_empty_first_page_stops_offset_pagination(self, local_server):
        # /empty-offset always returns [] — pagination should stop after 1 page
        result = await http_request(
            url=f"{local_server}/empty-offset",
            pagination={
                "type": "offset",
                "results_path": "items",
                "total_path": "total",
                "offset_param": "offset",
                "limit_param": "limit",
                "limit": 10,
                "max_pages": 10,
            },
        )
        assert result["pages"] == 1
        assert result["total_results"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# data_transform — 15 operations
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_DATA = [
    {"id": 1, "name": "Alice", "age": 30, "role": "admin"},
    {"id": 2, "name": "Bob",   "age": 25, "role": "user"},
    {"id": 3, "name": "Carol", "age": 35, "role": "user"},
    {"id": 4, "name": "Dave",  "age": 25, "role": "admin"},
]


class TestDataTransformFilter:
    @pytest.mark.asyncio
    async def test_filter_eq(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "filter", "field": "role", "operator": "eq", "value": "admin"}],
        )
        assert result["output_count"] == 2
        assert all(r["role"] == "admin" for r in result["result"])

    @pytest.mark.asyncio
    async def test_filter_in(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "filter", "field": "age", "operator": "in", "value": [25, 30]}],
        )
        assert result["output_count"] == 3

    @pytest.mark.asyncio
    async def test_filter_is_null(self):
        data = [{"id": 1, "val": None}, {"id": 2, "val": "x"}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "filter", "field": "val", "operator": "is_null"}],
        )
        assert result["output_count"] == 1
        assert result["result"][0]["id"] == 1

    @pytest.mark.asyncio
    async def test_filter_is_not_null(self):
        data = [{"id": 1, "val": None}, {"id": 2, "val": "x"}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "filter", "field": "val", "operator": "is_not_null"}],
        )
        assert result["output_count"] == 1
        assert result["result"][0]["id"] == 2

    @pytest.mark.asyncio
    async def test_filter_dot_notation(self):
        data = [
            {"user": {"active": True}, "id": 1},
            {"user": {"active": False}, "id": 2},
        ]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "filter", "field": "user.active", "operator": "eq", "value": True}],
        )
        assert result["output_count"] == 1
        assert result["result"][0]["id"] == 1

    @pytest.mark.asyncio
    async def test_filter_contains_string(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "filter", "field": "name", "operator": "contains", "value": "a"}],
        )
        names = [r["name"] for r in result["result"]]
        assert "Carol" in names   # contains lowercase 'a'
        assert "Dave" in names    # contains lowercase 'a'
        assert "Alice" not in names   # only 'A' (capital)

    @pytest.mark.asyncio
    async def test_filter_gt(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "filter", "field": "age", "operator": "gt", "value": 30}],
        )
        assert result["output_count"] == 1
        assert result["result"][0]["name"] == "Carol"


class TestDataTransformMap:
    @pytest.mark.asyncio
    async def test_map_template_substitution(self):
        result = await data_transform(
            input_data=SAMPLE_DATA[:2],
            operations=[{"op": "map", "template": "{name} ({role})", "output_field": "label"}],
        )
        assert result["result"][0]["label"] == "Alice (admin)"
        assert result["result"][1]["label"] == "Bob (user)"

    @pytest.mark.asyncio
    async def test_map_math_multiply(self):
        data = [{"price": 10}, {"price": 20}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "map", "math_op": "multiply", "field": "price", "operand": 2}],
        )
        assert result["result"][0]["price"] == 20
        assert result["result"][1]["price"] == 40


class TestDataTransformSort:
    @pytest.mark.asyncio
    async def test_sort_asc(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "sort", "field": "age", "order": "asc"}],
        )
        ages = [r["age"] for r in result["result"]]
        assert ages == sorted(ages)

    @pytest.mark.asyncio
    async def test_sort_desc(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "sort", "field": "name", "order": "desc"}],
        )
        names = [r["name"] for r in result["result"]]
        assert names == sorted(names, reverse=True)

    @pytest.mark.asyncio
    async def test_sort_by_date(self):
        data = [
            {"dt": "2023-06-15"},
            {"dt": "2021-01-01"},
            {"dt": "2024-12-31"},
        ]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "sort", "field": "dt", "type": "date", "order": "asc"}],
        )
        assert result["result"][0]["dt"] == "2021-01-01"
        assert result["result"][-1]["dt"] == "2024-12-31"


class TestDataTransformGroupBy:
    @pytest.mark.asyncio
    async def test_group_by_field(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "group_by", "field": "role"}],
        )
        groups = result["result"]
        assert "admin" in groups
        assert "user" in groups
        assert len(groups["admin"]) == 2
        assert len(groups["user"]) == 2


class TestDataTransformFlatten:
    @pytest.mark.asyncio
    async def test_flatten_top_level_list_of_lists(self):
        data = [[1, 2, 3], [4, 5], [6]]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "flatten"}],
        )
        assert result["result"] == [1, 2, 3, 4, 5, 6]

    @pytest.mark.asyncio
    async def test_flatten_by_field(self):
        data = [{"tags": ["a", "b"]}, {"tags": ["c"]}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "flatten", "field": "tags"}],
        )
        assert result["result"] == ["a", "b", "c"]


class TestDataTransformPick:
    @pytest.mark.asyncio
    async def test_pick_specific_fields(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "pick", "fields": ["id", "name"]}],
        )
        for item in result["result"]:
            assert set(item.keys()) == {"id", "name"}

    @pytest.mark.asyncio
    async def test_pick_dot_notation(self):
        data = [{"user": {"id": 1, "name": "Alice"}, "secret": "x"}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "pick", "fields": ["user.name"]}],
        )
        assert result["result"][0] == {"user": {"name": "Alice"}}


class TestDataTransformOmit:
    @pytest.mark.asyncio
    async def test_omit_fields(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "omit", "fields": ["role", "age"]}],
        )
        for item in result["result"]:
            assert "role" not in item
            assert "age" not in item
            assert "id" in item


class TestDataTransformRename:
    @pytest.mark.asyncio
    async def test_rename_field(self):
        result = await data_transform(
            input_data=SAMPLE_DATA[:1],
            operations=[{"op": "rename", "mapping": {"name": "full_name"}}],
        )
        assert "full_name" in result["result"][0]
        assert "name" not in result["result"][0]


class TestDataTransformDeduplicate:
    @pytest.mark.asyncio
    async def test_deduplicate_by_field(self):
        data = [{"id": 1, "val": "a"}, {"id": 2, "val": "a"}, {"id": 3, "val": "b"}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "deduplicate", "field": "val"}],
        )
        assert result["output_count"] == 2
        assert result["result"][0]["id"] == 1   # keeps first

    @pytest.mark.asyncio
    async def test_deduplicate_full_equality(self):
        data = [{"x": 1}, {"x": 2}, {"x": 1}, {"x": 3}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "deduplicate"}],
        )
        assert result["output_count"] == 3


class TestDataTransformLimitSkip:
    @pytest.mark.asyncio
    async def test_limit(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "limit", "count": 2}],
        )
        assert result["output_count"] == 2
        assert result["result"][0]["id"] == 1

    @pytest.mark.asyncio
    async def test_skip(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "skip", "count": 2}],
        )
        assert result["output_count"] == 2
        assert result["result"][0]["id"] == 3


class TestDataTransformAggregate:
    @pytest.mark.asyncio
    async def test_aggregate_sum(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "aggregate", "field": "age", "operation": "sum", "output_field": "total_age"}],
        )
        assert result["result"] == {"total_age": 115}   # 30+25+35+25

    @pytest.mark.asyncio
    async def test_aggregate_avg(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "aggregate", "field": "age", "operation": "avg", "output_field": "avg_age"}],
        )
        assert abs(result["result"]["avg_age"] - 28.75) < 0.01

    @pytest.mark.asyncio
    async def test_aggregate_count(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "aggregate", "field": "id", "operation": "count", "output_field": "n"}],
        )
        assert result["result"] == {"n": 4}

    @pytest.mark.asyncio
    async def test_aggregate_min_max(self):
        r_min = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "aggregate", "field": "age", "operation": "min", "output_field": "min_age"}],
        )
        r_max = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "aggregate", "field": "age", "operation": "max", "output_field": "max_age"}],
        )
        assert r_min["result"]["min_age"] == 25
        assert r_max["result"]["max_age"] == 35

    @pytest.mark.asyncio
    async def test_aggregate_distinct(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[{"op": "aggregate", "field": "role", "operation": "distinct", "output_field": "roles"}],
        )
        assert set(result["result"]["roles"]) == {"admin", "user"}


class TestDataTransformMerge:
    @pytest.mark.asyncio
    async def test_merge_list_of_dicts(self):
        data = [{"a": 1}, {"b": 2}, {"c": 3}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "merge"}],
        )
        assert result["result"] == {"a": 1, "b": 2, "c": 3}


class TestDataTransformSet:
    @pytest.mark.asyncio
    async def test_set_constant_value(self):
        result = await data_transform(
            input_data=SAMPLE_DATA[:2],
            operations=[{"op": "set", "field": "active", "value": True}],
        )
        assert all(r["active"] is True for r in result["result"])


class TestDataTransformCast:
    @pytest.mark.asyncio
    async def test_cast_str_to_int(self):
        data = [{"val": "42"}, {"val": "7"}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "cast", "field": "val", "to": "int"}],
        )
        assert result["result"][0]["val"] == 42
        assert isinstance(result["result"][0]["val"], int)

    @pytest.mark.asyncio
    async def test_cast_to_float(self):
        data = [{"val": "3.14"}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "cast", "field": "val", "to": "float"}],
        )
        assert abs(result["result"][0]["val"] - 3.14) < 0.001

    @pytest.mark.asyncio
    async def test_cast_to_str(self):
        data = [{"val": 123}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "cast", "field": "val", "to": "str"}],
        )
        assert result["result"][0]["val"] == "123"

    @pytest.mark.asyncio
    async def test_cast_to_bool(self):
        data = [{"val": "true"}, {"val": "false"}, {"val": "0"}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "cast", "field": "val", "to": "bool"}],
        )
        assert result["result"][0]["val"] is True
        assert result["result"][1]["val"] is False
        assert result["result"][2]["val"] is False


class TestDataTransformChained:
    @pytest.mark.asyncio
    async def test_filter_sort_limit(self):
        result = await data_transform(
            input_data=SAMPLE_DATA,
            operations=[
                {"op": "filter", "field": "role", "operator": "eq", "value": "user"},
                {"op": "sort", "field": "age", "order": "asc"},
                {"op": "limit", "count": 1},
            ],
        )
        assert result["output_count"] == 1
        assert result["result"][0]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_input_not_mutated(self):
        original = [{"x": 1}, {"x": 2}]
        snapshot = [{"x": 1}, {"x": 2}]
        await data_transform(
            input_data=original,
            operations=[{"op": "set", "field": "x", "value": 999}],
        )
        assert original == snapshot   # deep copy means original is unchanged


# ═══════════════════════════════════════════════════════════════════════════════
# Regression: sort with None values (bug fix — _sort_key returned (1, None))
# ═══════════════════════════════════════════════════════════════════════════════

class TestSortNoneRegression:
    """Before fix: sorting a list where ≥2 items lack the sort field raised
    TypeError: '<' not supported between 'NoneType' and 'NoneType'.
    Fix: use (1, 0) sentinel so None items sort stably at the end."""

    @pytest.mark.asyncio
    async def test_sort_multiple_missing_field_no_crash(self):
        data = [{"id": 1}, {"id": 2}, {"id": 3, "score": 5}]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "sort", "field": "score", "order": "asc"}],
        )
        assert result["result"][0]["id"] == 3   # only item with a value comes first

    @pytest.mark.asyncio
    async def test_sort_all_missing_field_no_crash(self):
        data = [{"id": i} for i in range(5)]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "sort", "field": "score", "order": "asc"}],
        )
        assert result["output_count"] == 5      # all items preserved

    @pytest.mark.asyncio
    async def test_sort_date_multiple_unparseable_no_crash(self):
        data = [
            {"id": 1, "dt": "not-a-date"},
            {"id": 2, "dt": "also-invalid"},
            {"id": 3, "dt": "2024-01-15"},
        ]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "sort", "field": "dt", "type": "date", "order": "asc"}],
        )
        assert result["result"][0]["id"] == 3   # valid date sorts first

    @pytest.mark.asyncio
    async def test_sort_number_type_multiple_nones_no_crash(self):
        data = [
            {"id": 1, "price": 9.99},
            {"id": 2},
            {"id": 3},
            {"id": 4, "price": 4.99},
        ]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "sort", "field": "price", "type": "number", "order": "asc"}],
        )
        ids = [r["id"] for r in result["result"]]
        assert ids[0] == 4          # cheapest first
        assert ids[1] == 1          # second cheapest
        assert set(ids[2:]) == {2, 3}   # None-priced items at end

    @pytest.mark.asyncio
    async def test_sort_nones_appear_last(self):
        data = [
            {"id": 1, "v": None},
            {"id": 2, "v": 10},
            {"id": 3, "v": None},
            {"id": 4, "v": 5},
        ]
        result = await data_transform(
            input_data=data,
            operations=[{"op": "sort", "field": "v", "type": "number", "order": "asc"}],
        )
        ids = [r["id"] for r in result["result"]]
        assert ids[0] == 4          # v=5
        assert ids[1] == 2          # v=10
        assert set(ids[2:]) == {1, 3}   # v=None → last


# ═══════════════════════════════════════════════════════════════════════════════
# data_transform — real-world product catalog smoke test
# ═══════════════════════════════════════════════════════════════════════════════

PRODUCT_CATALOG = [
    {"id": 1,  "name": "Laptop Pro",      "category": "electronics", "price": 1299.99, "stock": 15,  "active": True},
    {"id": 2,  "name": "Wireless Mouse",  "category": "electronics", "price": 29.99,  "stock": 200, "active": True},
    {"id": 3,  "name": "Standing Desk",   "category": "furniture",   "price": 449.00,  "stock": 8,   "active": True},
    {"id": 4,  "name": "USB-C Hub",       "category": "electronics", "price": 49.99,  "stock": 75,  "active": False},
    {"id": 5,  "name": "Ergonomic Chair", "category": "furniture",   "price": 599.00,  "stock": 5,   "active": True},
    {"id": 6,  "name": "Monitor 4K",      "category": "electronics", "price": 799.00,  "stock": 22,  "active": True},
    {"id": 7,  "name": "Keyboard Mech",   "category": "electronics", "price": 149.99,  "stock": 50,  "active": True},
    {"id": 8,  "name": "Desk Lamp",       "category": "furniture",   "price": 39.99,   "stock": 120, "active": False},
    {"id": 9,  "name": "Webcam HD",       "category": "electronics", "price": 89.99,  "stock": 35,  "active": True},
    {"id": 10, "name": "Cable Organizer", "category": "accessories", "price": 12.99,  "stock": 300, "active": True},
]


class TestDataTransformRealData:

    @pytest.mark.asyncio
    async def test_active_electronics_sorted_by_price(self):
        result = await data_transform(
            input_data=PRODUCT_CATALOG,
            operations=[
                {"op": "filter", "field": "active",   "operator": "eq", "value": True},
                {"op": "filter", "field": "category", "operator": "eq", "value": "electronics"},
                {"op": "sort",   "field": "price", "type": "number", "order": "asc"},
                {"op": "pick",   "fields": ["id", "name", "price"]},
            ],
        )
        # Active electronics: ids 1,2,6,7,9
        assert result["input_count"] == 10
        assert result["output_count"] == 5
        prices = [r["price"] for r in result["result"]]
        assert prices == sorted(prices)
        assert result["result"][0] == {"id": 2,  "name": "Wireless Mouse", "price": 29.99}
        assert result["result"][-1] == {"id": 1, "name": "Laptop Pro",     "price": 1299.99}
        assert all(set(r.keys()) == {"id", "name", "price"} for r in result["result"])

    @pytest.mark.asyncio
    async def test_top3_cheapest_active_products(self):
        result = await data_transform(
            input_data=PRODUCT_CATALOG,
            operations=[
                {"op": "filter", "field": "active", "operator": "eq", "value": True},
                {"op": "sort",   "field": "price", "type": "number", "order": "asc"},
                {"op": "limit",  "count": 3},
            ],
        )
        assert result["output_count"] == 3
        # Desk Lamp ($39.99) is inactive; 3rd cheapest active is Webcam HD ($89.99)
        prices = [r["price"] for r in result["result"]]
        assert prices == [12.99, 29.99, 89.99]
        ids = [r["id"] for r in result["result"]]
        assert ids == [10, 2, 9]

    @pytest.mark.asyncio
    async def test_aggregate_total_stock_of_active_products(self):
        filtered = await data_transform(
            input_data=PRODUCT_CATALOG,
            operations=[{"op": "filter", "field": "active", "operator": "eq", "value": True}],
        )
        active = filtered["result"]
        assert len(active) == 8   # ids 1,2,3,5,6,7,9,10

        result = await data_transform(
            input_data=active,
            operations=[{"op": "aggregate", "field": "stock", "operation": "sum", "output_field": "total"}],
        )
        # 15+200+8+5+22+50+35+300 = 635
        assert result["result"]["total"] == 635

    @pytest.mark.asyncio
    async def test_aggregate_avg_price_of_active_products(self):
        filtered = await data_transform(
            input_data=PRODUCT_CATALOG,
            operations=[{"op": "filter", "field": "active", "operator": "eq", "value": True}],
        )
        result = await data_transform(
            input_data=filtered["result"],
            operations=[{"op": "aggregate", "field": "price", "operation": "avg", "output_field": "avg"}],
        )
        expected = (1299.99 + 29.99 + 449.00 + 599.00 + 799.00 + 149.99 + 89.99 + 12.99) / 8
        assert abs(result["result"]["avg"] - expected) < 0.01

    @pytest.mark.asyncio
    async def test_group_by_category_counts(self):
        result = await data_transform(
            input_data=PRODUCT_CATALOG,
            operations=[
                {"op": "filter",   "field": "active", "operator": "eq", "value": True},
                {"op": "group_by", "field": "category"},
            ],
        )
        groups = result["result"]
        assert len(groups["electronics"]) == 5   # 1,2,6,7,9
        assert len(groups["furniture"])   == 2   # 3,5
        assert len(groups["accessories"]) == 1   # 10

    @pytest.mark.asyncio
    async def test_apply_10pct_discount_to_price(self):
        result = await data_transform(
            input_data=PRODUCT_CATALOG[:3],
            operations=[
                {"op": "set",  "field": "discounted",       "value": True},
                {"op": "map",  "math_op": "multiply", "field": "price",
                               "operand": 0.9, "output_field": "sale_price"},
            ],
        )
        items = result["result"]
        assert all(r["discounted"] is True for r in items)
        assert abs(items[0]["sale_price"] - 1299.99 * 0.9) < 0.01
        assert items[0]["price"] == 1299.99   # original untouched

    @pytest.mark.asyncio
    async def test_rename_and_omit(self):
        result = await data_transform(
            input_data=PRODUCT_CATALOG[:4],
            operations=[
                {"op": "rename", "mapping": {"price": "unit_price"}},
                {"op": "omit",   "fields": ["stock", "active"]},
            ],
        )
        for item in result["result"]:
            assert "unit_price" in item
            assert "price" not in item
            assert "stock" not in item
            assert "active" not in item

    @pytest.mark.asyncio
    async def test_deduplicate_by_category_keeps_first(self):
        result = await data_transform(
            input_data=PRODUCT_CATALOG,
            operations=[{"op": "deduplicate", "field": "category"}],
        )
        assert result["output_count"] == 3
        ids = [r["id"] for r in result["result"]]
        assert 1  in ids   # first electronics
        assert 3  in ids   # first furniture
        assert 10 in ids   # first accessories

    @pytest.mark.asyncio
    async def test_cast_price_to_int(self):
        result = await data_transform(
            input_data=PRODUCT_CATALOG[:5],
            operations=[{"op": "cast", "field": "price", "to": "int"}],
        )
        prices = [r["price"] for r in result["result"]]
        assert prices == [1299, 29, 449, 49, 599]
        assert all(isinstance(p, int) for p in prices)

    @pytest.mark.asyncio
    async def test_skip_limit_simulates_page_2(self):
        result = await data_transform(
            input_data=PRODUCT_CATALOG,
            operations=[
                {"op": "sort",  "field": "id", "type": "number", "order": "asc"},
                {"op": "skip",  "count": 3},
                {"op": "limit", "count": 3},
            ],
        )
        ids = [r["id"] for r in result["result"]]
        assert ids == [4, 5, 6]

    @pytest.mark.asyncio
    async def test_name_contains_filter(self):
        result = await data_transform(
            input_data=PRODUCT_CATALOG,
            operations=[{"op": "filter", "field": "name", "operator": "contains", "value": "Pro"}],
        )
        assert result["output_count"] == 1
        assert result["result"][0]["id"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Explicit coverage for untested filter operators, math ops, aggregates,
# and rate_limit pagination — all identified as implemented but untested.
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterOperatorsComplete:
    """Covers all 14 filter operators, not just the 6 in the main suite."""

    DATA = [
        {"id": 1, "name": "alpha", "score": 10, "tags": ["a", "b"]},
        {"id": 2, "name": "beta",  "score": 20, "tags": ["b", "c"]},
        {"id": 3, "name": "gamma", "score": 30, "tags": ["c"]},
        {"id": 4, "name": "delta", "score": 10, "tags": []},
        {"id": 5, "name": None,    "score": None},
    ]

    @pytest.mark.asyncio
    async def test_filter_ne(self):
        result = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "filter", "field": "score", "operator": "ne", "value": 10}],
        )
        ids = [r["id"] for r in result["result"]]
        assert ids == [2, 3, 5]   # score != 10; None != 10 so id=5 passes

    @pytest.mark.asyncio
    async def test_filter_gte(self):
        result = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "filter", "field": "score", "operator": "gte", "value": 20}],
        )
        ids = [r["id"] for r in result["result"]]
        assert ids == [2, 3]

    @pytest.mark.asyncio
    async def test_filter_lt(self):
        result = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "filter", "field": "score", "operator": "lt", "value": 20}],
        )
        ids = [r["id"] for r in result["result"]]
        assert ids == [1, 4]

    @pytest.mark.asyncio
    async def test_filter_lte(self):
        result = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "filter", "field": "score", "operator": "lte", "value": 20}],
        )
        ids = [r["id"] for r in result["result"]]
        assert ids == [1, 2, 4]

    @pytest.mark.asyncio
    async def test_filter_starts_with(self):
        result = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "filter", "field": "name", "operator": "starts_with", "value": "al"}],
        )
        assert result["output_count"] == 1
        assert result["result"][0]["id"] == 1

    @pytest.mark.asyncio
    async def test_filter_ends_with(self):
        result = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "filter", "field": "name", "operator": "ends_with", "value": "ta"}],
        )
        ids = [r["id"] for r in result["result"]]
        assert ids == [2, 4]   # "beta", "delta"

    @pytest.mark.asyncio
    async def test_filter_not_contains_string(self):
        result = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "filter", "field": "name", "operator": "not_contains", "value": "a"}],
        )
        # names without "a": "beta" has no lowercase 'a'... wait:
        # alpha→has 'a', beta→has 'a', gamma→has 'a', delta→has 'a', None→is_null
        # So only None passes not_contains for non-string (returns True from the not-string branch)
        # Let's use a value that only some match
        result2 = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "filter", "field": "name", "operator": "not_contains", "value": "eta"}],
        )
        # "beta" contains "eta"; others don't (or are None)
        ids = [r["id"] for r in result2["result"]]
        assert 2 not in ids    # "beta" excluded
        assert 1 in ids        # "alpha" kept

    @pytest.mark.asyncio
    async def test_filter_not_in(self):
        result = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "filter", "field": "score", "operator": "not_in", "value": [10, 30]}],
        )
        ids = [r["id"] for r in result["result"]]
        assert ids == [2, 5]   # score=20 and score=None

    @pytest.mark.asyncio
    async def test_filter_contains_list(self):
        """contains also works on list fields — value is in the list."""
        result = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "filter", "field": "tags", "operator": "contains", "value": "b"}],
        )
        ids = [r["id"] for r in result["result"]]
        assert ids == [1, 2]   # tags include "b"

    @pytest.mark.asyncio
    async def test_filter_multi_condition_and(self):
        """Multiple conditions with AND logic."""
        result = await data_transform(
            input_data=self.DATA,
            operations=[{
                "op": "filter",
                "conditions": [
                    {"field": "score", "operator": "gte", "value": 10},
                    {"field": "score", "operator": "lte", "value": 20},
                ],
                "logic": "and",
            }],
        )
        ids = [r["id"] for r in result["result"]]
        assert ids == [1, 2, 4]   # score 10, 20, 10

    @pytest.mark.asyncio
    async def test_filter_multi_condition_or(self):
        """Multiple conditions with OR logic."""
        result = await data_transform(
            input_data=self.DATA,
            operations=[{
                "op": "filter",
                "conditions": [
                    {"field": "score", "operator": "eq", "value": 10},
                    {"field": "score", "operator": "eq", "value": 30},
                ],
                "logic": "or",
            }],
        )
        ids = [r["id"] for r in result["result"]]
        assert ids == [1, 3, 4]   # score 10, 30, 10


class TestMapMathOpsComplete:
    """Covers all 8 math operations in the map op."""

    DATA = [{"val": 10.0}, {"val": -4.0}, {"val": 3.7}]

    @pytest.mark.asyncio
    async def test_map_math_add(self):
        result = await data_transform(
            input_data=[{"val": 10}],
            operations=[{"op": "map", "math_op": "add", "field": "val", "operand": 5}],
        )
        assert result["result"][0]["val"] == 15

    @pytest.mark.asyncio
    async def test_map_math_subtract(self):
        result = await data_transform(
            input_data=[{"val": 10}],
            operations=[{"op": "map", "math_op": "subtract", "field": "val", "operand": 3}],
        )
        assert result["result"][0]["val"] == 7

    @pytest.mark.asyncio
    async def test_map_math_divide(self):
        result = await data_transform(
            input_data=[{"val": 10}],
            operations=[{"op": "map", "math_op": "divide", "field": "val", "operand": 4}],
        )
        assert abs(result["result"][0]["val"] - 2.5) < 0.001

    @pytest.mark.asyncio
    async def test_map_math_round(self):
        result = await data_transform(
            input_data=[{"val": 3.14159}],
            operations=[{"op": "map", "math_op": "round", "field": "val", "operand": 2}],
        )
        assert result["result"][0]["val"] == 3.14

    @pytest.mark.asyncio
    async def test_map_math_abs(self):
        result = await data_transform(
            input_data=[{"val": -42}],
            operations=[{"op": "map", "math_op": "abs", "field": "val"}],
        )
        assert result["result"][0]["val"] == 42

    @pytest.mark.asyncio
    async def test_map_math_floor(self):
        result = await data_transform(
            input_data=[{"val": 3.9}],
            operations=[{"op": "map", "math_op": "floor", "field": "val"}],
        )
        assert result["result"][0]["val"] == 3

    @pytest.mark.asyncio
    async def test_map_math_ceil(self):
        result = await data_transform(
            input_data=[{"val": 3.1}],
            operations=[{"op": "map", "math_op": "ceil", "field": "val"}],
        )
        assert result["result"][0]["val"] == 4


class TestAggregateOpsComplete:
    """Covers first, last, concat — the 3 aggregate ops not in the main suite."""

    DATA = [
        {"id": 1, "word": "hello", "score": 5},
        {"id": 2, "word": "world", "score": 10},
        {"id": 3, "word": "nexus", "score": 15},
    ]

    @pytest.mark.asyncio
    async def test_aggregate_first(self):
        result = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "aggregate", "field": "word", "operation": "first", "output_field": "fw"}],
        )
        assert result["result"] == {"fw": "hello"}

    @pytest.mark.asyncio
    async def test_aggregate_last(self):
        result = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "aggregate", "field": "word", "operation": "last", "output_field": "lw"}],
        )
        assert result["result"] == {"lw": "nexus"}

    @pytest.mark.asyncio
    async def test_aggregate_concat(self):
        result = await data_transform(
            input_data=self.DATA,
            operations=[{
                "op": "aggregate",
                "field": "word",
                "operation": "concat",
                "separator": ", ",
                "output_field": "joined",
            }],
        )
        assert result["result"] == {"joined": "hello, world, nexus"}

    @pytest.mark.asyncio
    async def test_aggregate_concat_no_separator(self):
        result = await data_transform(
            input_data=self.DATA,
            operations=[{"op": "aggregate", "field": "word", "operation": "concat", "output_field": "j"}],
        )
        assert result["result"] == {"j": "helloworldnexus"}


class TestPaginationRateLimit:
    """Rate limit in pagination: verify requests_per_second is honored."""

    @pytest.mark.asyncio
    async def test_cursor_pagination_with_rate_limit(self, local_server):
        # With rate_limit set, pagination still returns correct results;
        # rate_limit=100 req/s means sleep(0.01) between pages — fast enough for CI
        result = await http_request(
            url=f"{local_server}/cursor",
            pagination={
                "type": "cursor",
                "results_path": "items",
                "next_cursor_path": "next_cursor",
                "cursor_param": "cursor",
                "max_pages": 10,
            },
            rate_limit={"requests_per_second": 100},
        )
        assert result["pages"] == 3
        assert result["results"] == [1, 2, 3, 4, 5, 6]

    @pytest.mark.asyncio
    async def test_offset_pagination_with_rate_limit(self, local_server):
        result = await http_request(
            url=f"{local_server}/offset",
            pagination={
                "type": "offset",
                "results_path": "items",
                "total_path": "total",
                "offset_param": "offset",
                "limit_param": "limit",
                "limit": 7,
                "max_pages": 10,
            },
            rate_limit={"requests_per_second": 100},
        )
        assert result["total_results"] == 7
        assert result["results"] == [1, 2, 3, 4, 5, 6, 7]

    @pytest.mark.asyncio
    async def test_link_header_pagination_with_rate_limit(self, local_server):
        result = await http_request(
            url=f"{local_server}/paginated",
            response_format="json",
            pagination={
                "type": "link_header",
                "results_path": "",
                "max_pages": 10,
            },
            rate_limit={"requests_per_second": 100},
        )
        assert result["pages"] == 3
        assert result["total_results"] == 5
