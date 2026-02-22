"""Built-in web tools: search and fetch.

v1: Stubs that return formatted strings. Production: integrate with Serper/Tavily.
"""

from nexus.tools.plugin import tool
from nexus.types import RiskLevel


@tool(name="web_search", description="Search the web for information", risk_level=RiskLevel.LOW, resource_pattern="web:*")
async def web_search(query: str) -> str:
    """Search the web for information.

    Args:
        query: Search query string

    Returns:
        Search results as text
    """
    # v1 stub — replace with real search API integration
    return f"Search results for: {query}\n\n1. Result about {query} from example.com\n2. More info on {query} from docs.example.com"


@tool(name="web_fetch", description="Fetch content from a URL", risk_level=RiskLevel.LOW, resource_pattern="web:*")
async def web_fetch(url: str) -> str:
    """Fetch webpage content.

    Args:
        url: URL to fetch

    Returns:
        Page content as text (truncated to 5000 chars)
    """
    # v1 stub — replace with httpx.get(url).text[:5000]
    return f"Fetched content from {url}: [Page content would appear here]"
