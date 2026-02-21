"""GET /v1/ledger — Audit trail queries."""

from fastapi import APIRouter, Request, Query
from typing import Optional

router = APIRouter(tags=["ledger"])


@router.get("/ledger")
async def list_seals(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Paginated seal history for the authenticated tenant.

    Args:
        limit: Max results per page
        offset: Pagination offset
    """
    # TODO: Get tenant_id from request.state, query ledger
    return {"seals": [], "total": 0, "limit": limit, "offset": offset}


@router.get("/ledger/{chain_id}")
async def get_chain_seals(request: Request, chain_id: str):
    """Get all seals for a specific chain.

    Args:
        chain_id: Chain to retrieve
    """
    # TODO: Get chain seals from ledger, verify tenant ownership
    return {"chain_id": chain_id, "seals": []}
