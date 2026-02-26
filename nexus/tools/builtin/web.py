"""Built-in web tools: search and fetch."""

import logging

from nexus.tools.plugin import tool
from nexus.types import RiskLevel

logger = logging.getLogger(__name__)


@tool(name="web_search", description="Search the web for information", risk_level=RiskLevel.LOW, resource_pattern="web:*")
async def web_search(query: str) -> str:
    """Search the web for information.

    Uses DuckDuckGo Instant Answer API (no key required).
    Falls back to a descriptive stub if the request fails.

    Args:
        query: Search query string

    Returns:
        Search results as text
    """
    try:
        import httpx
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        lines = []
        abstract = data.get("AbstractText", "")
        if abstract:
            lines.append(f"Summary: {abstract}")
            source = data.get("AbstractSource", "")
            if source:
                lines.append(f"Source: {source}")

        related = data.get("RelatedTopics", [])[:5]
        for item in related:
            if isinstance(item, dict) and item.get("Text"):
                lines.append(f"- {item['Text']}")

        if not lines:
            lines.append(f"No instant results found for: {query}")
            lines.append("Try a more specific query or use web_fetch with a direct URL.")

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("web_search failed for %r: %s", query, exc)
        return f"Search results for: {query}\n\n(Search temporarily unavailable: {exc})"


@tool(name="web_fetch", description="Fetch content from a URL", risk_level=RiskLevel.LOW, resource_pattern="web:*")
async def web_fetch(url: str) -> str:
    """Fetch webpage content.

    Args:
        url: URL to fetch

    Returns:
        Page content as text (truncated to 5000 chars)
    """
    try:
        import httpx
        headers = {"User-Agent": "NEXUS-Agent/1.0 (AI assistant; educational use)"}
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            content = resp.text

        # Strip HTML tags minimally for readability
        try:
            from html.parser import HTMLParser

            class _Stripper(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._chunks = []
                    self._skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style", "noscript"):
                        self._skip = True

                def handle_endtag(self, tag):
                    if tag in ("script", "style", "noscript"):
                        self._skip = False

                def handle_data(self, data):
                    if not self._skip:
                        stripped = data.strip()
                        if stripped:
                            self._chunks.append(stripped)

                def get_text(self):
                    return " ".join(self._chunks)

            stripper = _Stripper()
            stripper.feed(content)
            text = stripper.get_text()
        except Exception:
            text = content  # fall back to raw HTML if parser fails

        return text[:5000]

    except Exception as exc:
        logger.warning("web_fetch failed for %r: %s", url, exc)
        return f"Failed to fetch {url}: {exc}"
