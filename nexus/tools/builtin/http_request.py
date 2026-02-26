"""HTTP request tool with retry, pagination, JMESPath extraction, and auth support."""

import asyncio
import base64
import json
import re
import time
from typing import Any

import httpx
import jmespath

from nexus.exceptions import ToolError
from nexus.types import ToolDefinition, RiskLevel
from nexus.tools.plugin import _registered_tools


def _parse_link_header(header: str) -> dict[str, str]:
    """Parse RFC 5988 Link headers into a dict mapping rel -> url."""
    result = {}
    for part in header.split(","):
        part = part.strip()
        m = re.match(r'<([^>]+)>(.*)$', part)
        if not m:
            continue
        url, attrs = m.group(1), m.group(2)
        rel_m = re.search(r'rel=["\']?(\w+)["\']?', attrs)
        if rel_m:
            result[rel_m.group(1)] = url
    return result


def _extract_jmespath(data: Any, path: str) -> Any:
    """Extract data using a JMESPath expression."""
    if not path:
        return data
    return jmespath.search(path, data)


def _parse_response_body(content: bytes, response_format: str, encoding: str = "utf-8") -> Any:
    """Parse response body according to the requested format."""
    if response_format == "binary":
        return base64.b64encode(content).decode("ascii")
    elif response_format == "text":
        return content.decode(encoding, errors="replace")
    elif response_format == "json":
        return json.loads(content)
    else:  # "auto"
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return content.decode(encoding, errors="replace")


async def _execute_single(
    method: str,
    url: str,
    query_params: dict,
    client_kwargs: dict,
    request_kwargs: dict,
    size_limit_bytes: int,
) -> tuple:
    """Execute a single HTTP request with streaming size limit enforcement."""
    start = time.monotonic()
    async with httpx.AsyncClient(**client_kwargs) as client:
        # Only pass params when non-empty — httpx strips existing URL query
        # params when params={} is passed explicitly (e.g. link_header next URLs)
        extra = {"params": query_params} if query_params else {}
        async with client.stream(method.upper(), url, **extra, **request_kwargs) as response:
            content = b""
            async for chunk in response.aiter_bytes(8192):
                content += chunk
                if len(content) > size_limit_bytes:
                    raise ToolError(f"Response exceeds size limit of {size_limit_bytes} bytes")
    elapsed_ms = int((time.monotonic() - start) * 1000)
    return response, content, elapsed_ms


async def http_request(**kwargs) -> dict:
    """Make an HTTP request with retry, pagination, JMESPath extraction, and auth."""
    method = kwargs.get("method", "GET")
    url = kwargs.get("url")
    if not url:
        raise ToolError("url is required")

    headers = dict(kwargs.get("headers") or {})
    query_params = dict(kwargs.get("query_params") or {})
    body = kwargs.get("body")
    body_format = kwargs.get("body_format", "json")
    response_format = kwargs.get("response_format", "auto")
    response_encoding = kwargs.get("response_encoding", "utf-8")
    response_path = kwargs.get("response_path", "")
    timeout_seconds = kwargs.get("timeout_seconds", 30)
    max_retries = kwargs.get("max_retries", 0)
    retry_on = kwargs.get("retry_on") or [500, 502, 503, 504]
    verify_ssl = kwargs.get("verify_ssl", True)
    http2 = kwargs.get("http2", False)
    proxy = kwargs.get("proxy")
    size_limit_bytes = (kwargs.get("response_size_limit_kb") or 10240) * 1024
    auth_config = kwargs.get("auth")
    pagination = kwargs.get("pagination")
    rate_limit = kwargs.get("rate_limit")

    # Build auth headers
    if auth_config:
        auth_type = auth_config.get("type", "")
        if auth_type == "basic":
            creds = f"{auth_config.get('username', '')}:{auth_config.get('password', '')}".encode()
            headers["Authorization"] = f"Basic {base64.b64encode(creds).decode()}"
        elif auth_type == "bearer":
            headers["Authorization"] = f"Bearer {auth_config.get('token', '')}"
        elif auth_type == "api_key":
            key_header = auth_config.get("header", "X-API-Key")
            headers[key_header] = auth_config.get("key", "")

    # Build request kwargs
    request_kwargs: dict = {"headers": headers}

    # Body serialization
    if body is not None:
        if body_format == "json":
            request_kwargs["json"] = body
        elif body_format == "form":
            request_kwargs["data"] = body
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif body_format == "multipart":
            request_kwargs["files"] = body
        elif body_format == "graphql":
            request_kwargs["json"] = body
        elif body_format == "raw":
            if isinstance(body, str):
                request_kwargs["content"] = body.encode()
            else:
                request_kwargs["content"] = body

    # Client kwargs
    client_kwargs: dict = {
        "timeout": timeout_seconds,
        "verify": verify_ssl,
        "http2": http2,
    }
    if proxy:
        client_kwargs["proxy"] = proxy

    max_tries = max_retries + 1

    async def _make_request(req_url: str, req_params: dict) -> tuple:
        last_error = None
        for attempt in range(max_tries):
            try:
                response, content, elapsed_ms = await _execute_single(
                    method, req_url, req_params, client_kwargs, request_kwargs, size_limit_bytes
                )
                if response.status_code in retry_on and attempt < max_tries - 1:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            sleep_s = float(retry_after)
                        except ValueError:
                            sleep_s = 2 ** attempt
                    else:
                        sleep_s = 2 ** attempt
                    await asyncio.sleep(sleep_s)
                    continue
                return response, content, elapsed_ms
            except ToolError:
                raise
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                last_error = e
                if attempt == max_tries - 1:
                    raise ToolError(f"Connection failed after {max_tries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)
        raise ToolError(f"Request failed after {max_tries} attempts: {last_error}")

    def _build_response(response, content, elapsed_ms, resp_format, resp_encoding, resp_path, ssl_verify) -> dict:
        parsed_body = _parse_response_body(content, resp_format, resp_encoding)
        if resp_path:
            parsed_body = _extract_jmespath(parsed_body, resp_path)

        resp_headers = dict(response.headers)

        if response.status_code >= 400:
            return {
                "error": True,
                "status_code": response.status_code,
                "reason": response.reason_phrase if hasattr(response, 'reason_phrase') else str(response.status_code),
                "body": parsed_body,
                "headers": resp_headers,
                "url": str(response.url),
            }

        result = {
            "status_code": response.status_code,
            "headers": resp_headers,
            "body": parsed_body,
            "url": str(response.url),
            "elapsed_ms": elapsed_ms,
        }
        if not ssl_verify:
            result["ssl_verification_disabled"] = True
        return result

    # Pagination
    if pagination:
        pag_type = pagination.get("type", "")
        results_path = pagination.get("results_path", "")
        max_pages = pagination.get("max_pages", 10)
        accumulated = []
        pages = 0

        if pag_type == "cursor":
            next_cursor_path = pagination.get("next_cursor_path", "")
            cursor_param = pagination.get("cursor_param", "cursor")
            current_params = dict(query_params)

            while pages < max_pages:
                response, content, elapsed_ms = await _make_request(url, current_params)
                parsed_body = _parse_response_body(content, response_format, response_encoding)
                results = jmespath.search(results_path, parsed_body) or [] if results_path else parsed_body
                if isinstance(results, list):
                    accumulated.extend(results)
                else:
                    accumulated.append(results)
                next_cursor = jmespath.search(next_cursor_path, parsed_body) if next_cursor_path else None
                pages += 1
                if not next_cursor:
                    break
                current_params = dict(current_params)
                current_params[cursor_param] = next_cursor
                if rate_limit:
                    await asyncio.sleep(1.0 / rate_limit.get("requests_per_second", 1))

        elif pag_type == "offset":
            offset_param = pagination.get("offset_param", "offset")
            limit_param = pagination.get("limit_param", "limit")
            limit = pagination.get("limit", 100)
            total_path = pagination.get("total_path", "")
            offset = 0
            current_params = dict(query_params)
            current_params[limit_param] = limit

            while pages < max_pages:
                current_params[offset_param] = offset
                response, content, elapsed_ms = await _make_request(url, current_params)
                parsed_body = _parse_response_body(content, response_format, response_encoding)
                results = jmespath.search(results_path, parsed_body) or [] if results_path else parsed_body
                if not isinstance(results, list):
                    results = [results] if results else []
                accumulated.extend(results)
                total = jmespath.search(total_path, parsed_body) if total_path else None
                pages += 1
                offset += limit
                if not results or (total is not None and offset >= total):
                    break
                if rate_limit:
                    await asyncio.sleep(1.0 / rate_limit.get("requests_per_second", 1))

        elif pag_type == "link_header":
            current_url = url
            current_params = dict(query_params)

            while pages < max_pages:
                response, content, elapsed_ms = await _make_request(current_url, current_params)
                parsed_body = _parse_response_body(content, response_format, response_encoding)
                results = jmespath.search(results_path, parsed_body) or [] if results_path else parsed_body
                if isinstance(results, list):
                    accumulated.extend(results)
                else:
                    accumulated.append(results)
                links = _parse_link_header(response.headers.get("Link", ""))
                next_url = links.get("next")
                pages += 1
                if not next_url:
                    break
                current_url = next_url
                current_params = {}  # URL already has params
                if rate_limit:
                    await asyncio.sleep(1.0 / rate_limit.get("requests_per_second", 1))

        return {
            "pages": pages,
            "total_results": len(accumulated),
            "results": accumulated,
            "status_code": 200,
        }

    # Single request
    response, content, elapsed_ms = await _make_request(url, query_params)
    return _build_response(response, content, elapsed_ms, response_format, response_encoding, response_path, verify_ssl)


_HTTP_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"], "description": "HTTP method"},
        "url": {"type": "string", "description": "URL to request"},
        "headers": {"type": "object", "description": "HTTP headers", "additionalProperties": {"type": "string"}},
        "query_params": {"type": "object", "description": "URL query parameters"},
        "body": {"description": "Request body"},
        "body_format": {"type": "string", "enum": ["json", "form", "multipart", "graphql", "raw"], "default": "json"},
        "response_format": {"type": "string", "enum": ["auto", "json", "text", "binary"], "default": "auto"},
        "response_encoding": {"type": "string", "default": "utf-8"},
        "response_path": {"type": "string", "description": "JMESPath to extract from response body"},
        "timeout_seconds": {"type": "integer", "default": 30},
        "max_retries": {"type": "integer", "default": 0},
        "retry_on": {"type": "array", "items": {"type": "integer"}, "description": "Status codes to retry on"},
        "verify_ssl": {"type": "boolean", "default": True},
        "http2": {"type": "boolean", "default": False},
        "proxy": {"type": "string", "description": "Proxy URL"},
        "response_size_limit_kb": {"type": "integer", "default": 10240},
        "auth": {
            "type": "object",
            "description": "Authentication config",
            "properties": {
                "type": {"type": "string", "enum": ["basic", "bearer", "api_key"]},
                "username": {"type": "string"},
                "password": {"type": "string"},
                "token": {"type": "string"},
                "key": {"type": "string"},
                "header": {"type": "string"},
            },
        },
        "pagination": {
            "type": "object",
            "description": "Pagination config",
            "properties": {
                "type": {"type": "string", "enum": ["cursor", "offset", "link_header"]},
                "results_path": {"type": "string"},
                "next_cursor_path": {"type": "string"},
                "cursor_param": {"type": "string"},
                "offset_param": {"type": "string"},
                "limit_param": {"type": "string"},
                "limit": {"type": "integer"},
                "total_path": {"type": "string"},
                "max_pages": {"type": "integer", "default": 10},
            },
        },
        "rate_limit": {
            "type": "object",
            "properties": {
                "requests_per_second": {"type": "number"},
            },
        },
    },
    "required": ["url"],
}

_registered_tools["http_request"] = (
    ToolDefinition(
        name="http_request",
        description="Make HTTP requests with retry, pagination, JMESPath extraction, and credential injection",
        parameters=_HTTP_REQUEST_SCHEMA,
        risk_level=RiskLevel.MEDIUM,
        resource_pattern="web:*",
        timeout_seconds=120,
        requires_approval=False,
    ),
    http_request,
)
