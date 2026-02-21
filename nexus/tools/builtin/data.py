"""Built-in data tools: statistics, knowledge search."""

from typing import Any

from nexus.tools.plugin import tool
from nexus.types import RiskLevel


@tool(name="compute_stats", description="Compute statistics on data", risk_level=RiskLevel.LOW)
async def compute_stats(data: list, metrics: list = None) -> dict:
    """Compute statistics on data.

    Args:
        data: List of numeric values
        metrics: List of metric names to compute (e.g., ["mean", "median", "std"])

    Returns:
        Dict of computed statistics
    """
    # v1 stub — replace with numpy/pandas stats
    return {"count": len(data) if isinstance(data, list) else 0, "metrics": metrics or ["mean"]}


@tool(name="knowledge_search", description="Search the knowledge base", risk_level=RiskLevel.LOW, resource_pattern="kb:*")
async def knowledge_search(query: str, namespace: str = "default") -> str:
    """Search the knowledge base.

    This tool is special — it calls KnowledgeStore.query() internally.
    Wired up in Engine (Phase 4).

    Args:
        query: Search query
        namespace: Knowledge namespace to search

    Returns:
        Relevant knowledge as text
    """
    # v1 stub — wired to KnowledgeStore in engine.py
    return f"Knowledge results for \'{query}\' in namespace \'{namespace}\': [Results would appear here]"
