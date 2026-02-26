"""Webhook catch-all endpoint — no JWT required.

All requests to /v2/webhooks/{path} are routed here. The AuthMiddleware
skips this path (it's in PUBLIC_PREFIXES).
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from nexus.exceptions import TriggerError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["webhooks"])


@router.api_route(
    "/v2/webhooks/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def webhook_catch_all(request: Request, path: str):
    """Catch-all webhook endpoint. No authentication required.

    Extracts method/headers/query/body from the request and delegates to
    ``app.state.webhook_handler.handle(webhook_path, method, headers,
    query_params, body)``. Returns 503 if the handler is not registered,
    404 if no trigger matches the path.
    """
    handler = getattr(request.app.state, "webhook_handler", None)
    if handler is None:
        return JSONResponse(
            {"error": "webhook service unavailable"},
            status_code=503,
        )

    # Reconstruct the webhook_path as stored in TriggerConfig.webhook_path
    # Triggers are stored as "/webhooks/<tenant>/<uuid>"; the route captures
    # everything after "/v2/webhooks/" as `path`.
    webhook_path = f"/webhooks/{path}"

    # Parse body — try JSON first, fall back to raw text
    body = None
    try:
        body = await request.json()
    except Exception:
        try:
            raw = await request.body()
            body = raw.decode("utf-8", errors="replace") if raw else None
        except Exception:
            body = None

    try:
        result = await handler.handle(
            webhook_path=webhook_path,
            method=request.method,
            headers=dict(request.headers),
            query_params=dict(request.query_params),
            body=body,
        )
    except TriggerError as exc:
        msg = str(exc).lower()
        if "unknown" in msg or "not found" in msg:
            return JSONResponse({"error": str(exc)}, status_code=404)
        return JSONResponse({"error": str(exc)}, status_code=422)
    except Exception as exc:
        logger.exception("Webhook handler error for path=%r: %s", path, exc)
        return JSONResponse(
            {"error": "internal webhook error", "detail": str(exc)},
            status_code=500,
        )

    # Serialize the result — WorkflowExecution (Pydantic) or dict from dispatcher
    if hasattr(result, "model_dump"):
        return JSONResponse(result.model_dump(mode="json"), status_code=200)
    if isinstance(result, dict):
        return JSONResponse(result, status_code=200)
    return JSONResponse({"status": "accepted"}, status_code=200)
