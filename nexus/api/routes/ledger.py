"""GET /v1/ledger — Audit trail queries."""

from fastapi import APIRouter, Request, Query

router = APIRouter(tags=["ledger"])


def _seal_to_dict(seal) -> dict:
    gates = []
    if seal.anomaly_result:
        for g in seal.anomaly_result.gates:
            gates.append({
                "name": g.gate_name,
                "verdict": g.verdict.value if hasattr(g.verdict, "value") else g.verdict,
                "score": g.score,
                "threshold": g.threshold,
            })
    return {
        "id": seal.id,
        "chain_id": seal.chain_id,
        "step_index": seal.step_index,
        "persona": seal.persona_id,
        "tool": seal.tool_name,
        "status": seal.status.value if hasattr(seal.status, "value") else seal.status,
        "gates": gates,
        "reasoning": seal.cot_trace or [],
        "result": seal.tool_result,
        "created_at": seal.created_at.isoformat() if seal.created_at else None,
        "error": seal.error,
    }


@router.get("/ledger")
async def list_seals(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Paginated seal history for the authenticated tenant."""
    tenant_id = getattr(request.state, "tenant_id", "demo")
    ledger = request.app.state.ledger
    seals = await ledger.get_by_tenant(tenant_id, limit=limit, offset=offset)
    return {
        "seals": [_seal_to_dict(s) for s in seals],
        "total": len(seals),
        "limit": limit,
        "offset": offset,
    }


@router.get("/ledger/{chain_id}")
async def get_chain_seals(request: Request, chain_id: str):
    """Get all seals for a specific chain."""
    ledger = request.app.state.ledger
    seals = await ledger.get_chain(chain_id)
    return {
        "chain_id": chain_id,
        "seals": [_seal_to_dict(s) for s in seals],
    }
