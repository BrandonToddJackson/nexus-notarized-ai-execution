"""POST /v1/execute — Primary NEXUS endpoint."""

import time
import logging
from fastapi import APIRouter, Request, HTTPException

from nexus.api.schemas import (
    ExecuteRequest, ExecuteResponse, SealResponse, GateResponse, CostSummary,
)
from nexus.exceptions import AnomalyDetected, ChainAborted, EscalationRequired

logger = logging.getLogger(__name__)
router = APIRouter(tags=["execute"])


def _seal_to_response(seal) -> SealResponse:
    """Convert a Seal domain object to a SealResponse."""
    gates = []
    if seal.anomaly_result:
        for g in seal.anomaly_result.gates:
            gates.append(GateResponse(
                name=g.gate_name,
                verdict=g.verdict.value if hasattr(g.verdict, "value") else g.verdict,
                score=g.score,
                threshold=g.threshold,
            ))
    return SealResponse(
        id=seal.id,
        step_index=seal.step_index,
        persona=seal.persona_id,
        tool=seal.tool_name,
        status=seal.status.value if hasattr(seal.status, "value") else seal.status,
        gates=gates,
        reasoning=seal.cot_trace or [],
        result=seal.tool_result,
        created_at=seal.created_at.isoformat(),
        error=seal.error,
    )


@router.post("/execute", response_model=ExecuteResponse)
async def execute_task(request: Request, body: ExecuteRequest):
    """Execute a task through the NEXUS pipeline."""
    start = time.time()

    engine = request.app.state.engine
    tenant_id = getattr(request.state, "tenant_id", "demo")

    chain = None
    blocked_chain_id: str | None = None
    try:
        chain = await engine.run(
            task=body.task,
            tenant_id=tenant_id,
            persona_name=body.persona,
        )
    except AnomalyDetected as exc:
        # Chain was blocked — engine already sealed the blocked action and failed the chain.
        # engine.run() re-raises before returning chain, so we retrieve seals via chain_id.
        blocked_chain_id = exc.chain_id
    except (ChainAborted, EscalationRequired) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"[execute] Unhandled error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    duration_ms = int((time.time() - start) * 1000)

    # Collect seals from ledger. Works for both completed and blocked chains.
    ledger = request.app.state.ledger

    if chain is None and not blocked_chain_id:
        # No chain was created (decomposition failed before chain creation)
        raise HTTPException(status_code=500, detail="Chain not created")

    chain_id_str: str = str(chain.id) if chain else (blocked_chain_id or "")
    seals_raw = await ledger.get_chain(chain_id_str, tenant_id=tenant_id)
    seal_responses = [_seal_to_response(s) for s in seals_raw]

    # Final result = last executed seal's tool_result (None for blocked chains)
    final_result = None
    for s in reversed(seals_raw):
        if s.tool_result is not None:
            final_result = s.tool_result
            break

    # Cost summary (zero if no cost tracker wired)
    cost = CostSummary(input_tokens=0, output_tokens=0, total_cost_usd=0.0)

    # Determine final status: completed chain uses chain.status, blocked uses "blocked"
    if blocked_chain_id:
        final_status = "blocked"
    elif chain is not None:
        final_status = chain.status.value if hasattr(chain.status, "value") else str(chain.status)
    else:
        final_status = "unknown"

    return ExecuteResponse(
        chain_id=chain_id_str,
        status=final_status,
        seals=seal_responses,
        result=final_result,
        cost=cost,
        duration_ms=duration_ms,
    )
