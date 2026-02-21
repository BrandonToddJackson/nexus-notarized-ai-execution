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

    try:
        chain = await engine.run(
            task=body.task,
            tenant_id=tenant_id,
            persona_name=body.persona,
        )
    except AnomalyDetected as exc:
        # Chain was blocked — return partial result with blocked seals
        # The chain is already finalized in the engine; surface as 200 with blocked status
        raise HTTPException(status_code=200, detail=str(exc))
    except (ChainAborted, EscalationRequired) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"[execute] Unhandled error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    duration_ms = int((time.time() - start) * 1000)

    # Collect seals from ledger (in-memory; already appended by engine)
    ledger = request.app.state.ledger
    seals_raw = await ledger.get_chain(chain.id)
    seal_responses = [_seal_to_response(s) for s in seals_raw]

    # Final result = last executed seal's tool_result
    final_result = None
    for s in reversed(seals_raw):
        if s.tool_result is not None:
            final_result = s.tool_result
            break

    # Cost summary (zero if no cost tracker wired)
    cost = CostSummary(input_tokens=0, output_tokens=0, total_cost_usd=0.0)

    return ExecuteResponse(
        chain_id=chain.id,
        status=chain.status.value if hasattr(chain.status, "value") else chain.status,
        seals=seal_responses,
        result=final_result,
        cost=cost,
        duration_ms=duration_ms,
    )
