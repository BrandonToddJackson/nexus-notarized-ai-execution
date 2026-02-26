#!/usr/bin/env python
"""Wrapper that starts mcp-server-fetch with certifi SSL certs configured.

mcp-server-fetch is the official fetch server from modelcontextprotocol/servers.
On macOS with Homebrew Python the system SSL cert bundle is not available to
httpx by default.  Setting SSL_CERT_FILE before the server imports httpx
ensures certificate verification succeeds for HTTPS URLs.

Run standalone (for manual testing):
    python tests/fixtures/mcp_fetch_server.py
"""

import os
import sys

# Set the CA bundle BEFORE importing anything that might create an SSL context.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass  # certifi not installed; rely on system certs

# Patch httpx to use the correct SSL bundle if truststore is unavailable.
# httpx >=0.24 picks up SSL_CERT_FILE via its default ssl_context factory.
try:
    import ssl
    _ctx = ssl.create_default_context()  # forces env-var lookup
    os.environ["HTTPX_CAFILE"] = os.environ.get("SSL_CERT_FILE", "")
except Exception:
    pass

# ── run the real server ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Parse --ignore-robots-txt flag if present
    ignore_robots = "--ignore-robots-txt" in sys.argv
    import asyncio
    from mcp_server_fetch.server import serve
    asyncio.run(serve(ignore_robots_txt=ignore_robots))
