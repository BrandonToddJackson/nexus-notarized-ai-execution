"""ARQ task functions and WorkerSettings for NEXUS background workers."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ── Task functions ────────────────────────────────────────────────────────────

async def execute_workflow_task(
    ctx: dict,
    workflow_id: str,
    tenant_id: str,
    trigger_data: dict[str, Any],
) -> dict:
    """Execute a workflow in the background and persist the execution record."""
    engine = ctx["engine"]
    repo = ctx["repository"]

    started = time.monotonic()
    try:
        execution = await engine.run_workflow(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            trigger_data=trigger_data,
        )
        duration_ms = int((time.monotonic() - started) * 1000)

        # Persist execution record
        await repo.save_execution(execution)

        # Derive seal_count from chain
        seal_count = 0
        if execution.chain_id:
            seals = await repo.get_chain_seals(execution.chain_id)
            seal_count = len(seals)

        return {
            "execution_id": execution.id,
            "status": execution.status.value if hasattr(execution.status, "value") else str(execution.status),
            "seal_count": seal_count,
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        duration_ms = int((time.monotonic() - started) * 1000)
        logger.exception("execute_workflow_task failed for workflow=%s tenant=%s", workflow_id, tenant_id)
        return {
            "execution_id": None,
            "status": "failed",
            "seal_count": 0,
            "duration_ms": duration_ms,
            "error": str(exc),
        }


async def refresh_mcp_connections_task(
    ctx: dict,
    tenant_id: str,
) -> dict:
    """Reconnect all MCP servers for a tenant."""
    mcp_adapter = ctx["mcp_adapter"]

    try:
        result = await mcp_adapter.reconnect_all(tenant_id)
        return {
            "tenant_id": tenant_id,
            "reconnected": result.get("reconnected", []),
            "failed": result.get("failed", []),
            "errors": result.get("errors", {}),
        }
    except Exception as exc:
        logger.exception("refresh_mcp_connections_task failed for tenant=%s", tenant_id)
        return {
            "tenant_id": tenant_id,
            "reconnected": [],
            "failed": [],
            "errors": {"_task": str(exc)},
        }


# ── Worker lifecycle ──────────────────────────────────────────────────────────

async def startup(ctx: dict) -> None:
    """Initialize all NEXUS components for the worker process."""
    from nexus.config import config
    from nexus.db.database import init_db, async_session
    from nexus.cache.redis_client import RedisClient
    from nexus.cache.fingerprints import FingerprintCache
    from nexus.knowledge.embeddings import EmbeddingService
    from nexus.knowledge.store import KnowledgeStore
    from nexus.knowledge.context import ContextBuilder
    from nexus.core.personas import PersonaManager
    from nexus.core.anomaly import AnomalyEngine
    from nexus.core.notary import Notary
    from nexus.core.ledger import Ledger
    from nexus.core.chain import ChainManager
    from nexus.core.verifier import IntentVerifier
    from nexus.core.output_validator import OutputValidator
    from nexus.core.cot_logger import CoTLogger
    from nexus.tools.registry import ToolRegistry
    from nexus.tools.plugin import get_registered_tools
    import nexus.tools.builtin  # noqa: F401
    from nexus.tools.selector import ToolSelector
    from nexus.tools.sandbox import Sandbox
    from nexus.tools.executor import ToolExecutor
    from nexus.reasoning.think_act import ThinkActGate
    from nexus.reasoning.continue_complete import ContinueCompleteGate
    from nexus.reasoning.escalate import EscalateGate
    from nexus.llm.client import LLMClient
    from nexus.credentials.encryption import CredentialEncryption
    from nexus.credentials.vault import CredentialVault
    from nexus.mcp.client import MCPClient
    from nexus.mcp.adapter import MCPToolAdapter
    from nexus.core.engine import NexusEngine
    from nexus.db.repository import Repository
    from nexus.triggers import EventBus
    from nexus.types import PersonaContract, RiskLevel
    from nexus.workflows.manager import WorkflowManager

    logger.info("NEXUS worker starting up...")
    await init_db()

    redis_client = RedisClient()

    async with async_session() as session:
        repo = Repository(session)

        embedding_service = EmbeddingService(model_name=config.embedding_model)
        knowledge_store = KnowledgeStore(
            persist_dir=config.chroma_persist_dir,
            embedding_fn=embedding_service.embed,
        )

        # Load personas
        db_personas = await repo.list_personas("demo")
        persona_contracts = []
        for p in db_personas:
            try:
                risk = RiskLevel(p.risk_tolerance)
            except ValueError:
                risk = RiskLevel.MEDIUM
            persona_contracts.append(PersonaContract(
                id=str(p.id),
                name=str(p.name),
                description=str(p.description),
                allowed_tools=list(p.allowed_tools or []),
                resource_scopes=list(p.resource_scopes or []),
                intent_patterns=list(p.intent_patterns or []),
                max_ttl_seconds=int(p.max_ttl_seconds),
                risk_tolerance=risk,
            ))
        persona_manager = PersonaManager(persona_contracts)

    # Tool registry
    tool_registry = ToolRegistry()
    for tool_name, (definition, impl) in get_registered_tools().items():
        tool_registry.register(definition, impl)

    # Components
    fingerprint_store = FingerprintCache(redis_client)
    anomaly_engine = AnomalyEngine(
        config=config,
        embedding_service=embedding_service,
        fingerprint_store=fingerprint_store,
    )
    notary = Notary()
    ledger = Ledger()
    chain_manager = ChainManager()
    verifier = IntentVerifier()
    output_validator = OutputValidator()
    cot_logger = CoTLogger()
    context_builder = ContextBuilder(knowledge_store=knowledge_store)
    llm_client = LLMClient(task_type="general")
    tool_selector = ToolSelector(registry=tool_registry, llm_client=llm_client)
    sandbox = Sandbox()
    credential_encryption = CredentialEncryption(key=config.credential_encryption_key)
    vault = CredentialVault(encryption=credential_encryption)
    tool_executor = ToolExecutor(registry=tool_registry, sandbox=sandbox, verifier=verifier, vault=vault, config=config)
    think_act_gate = ThinkActGate()
    continue_complete_gate = ContinueCompleteGate()
    escalate_gate = EscalateGate()
    event_bus = EventBus()

    engine = NexusEngine(
        persona_manager=persona_manager,
        anomaly_engine=anomaly_engine,
        notary=notary,
        ledger=ledger,
        chain_manager=chain_manager,
        context_builder=context_builder,
        tool_registry=tool_registry,
        tool_selector=tool_selector,
        tool_executor=tool_executor,
        output_validator=output_validator,
        cot_logger=cot_logger,
        think_act_gate=think_act_gate,
        continue_complete_gate=continue_complete_gate,
        escalate_gate=escalate_gate,
        llm_client=llm_client,
        config=config,
        event_bus=event_bus,
    )

    # WorkflowManager — load from DB so workflows are available in the worker
    async with async_session() as wf_session:
        from nexus.db.repository import Repository as _WFRepo
        from nexus.db.models import WorkflowModel as _WFModel
        from nexus.types import WorkflowDefinition, WorkflowStatus
        from sqlalchemy import select as _sa_select
        wf_repo = _WFRepo(wf_session)
        workflow_manager = WorkflowManager(repository=wf_repo, config=config)
        _wf_rows = (await wf_session.execute(_sa_select(_WFModel))).scalars().all()
        for _row in _wf_rows:
            try:
                _steps = [{**s, "workflow_id": _row.id} for s in (_row.steps or [])]
                _edges = [{**e, "workflow_id": _row.id} for e in (_row.edges or [])]
                _wf = WorkflowDefinition(
                    id=_row.id,
                    tenant_id=_row.tenant_id,
                    name=_row.name,
                    description=_row.description or "",
                    version=_row.version,
                    status=WorkflowStatus(_row.status),
                    trigger_config=_row.trigger_config or {},
                    steps=_steps,
                    edges=_edges,
                    created_at=_row.created_at,
                    updated_at=_row.updated_at,
                    created_by=_row.created_by or "",
                    tags=_row.tags or [],
                    settings=_row.settings or {},
                )
                workflow_manager._store[_wf.id] = _wf
            except Exception:
                pass  # skip malformed rows
    engine.workflow_manager = workflow_manager

    mcp_client = MCPClient()
    mcp_adapter = MCPToolAdapter(tool_registry, mcp_client, None, vault)

    # Store in ctx for task functions
    ctx["engine"] = engine
    ctx["mcp_adapter"] = mcp_adapter
    ctx["_redis_client"] = redis_client
    ctx["_session_factory"] = async_session

    # For task use, create a fresh repository (long-lived session for worker)
    # Each task should open its own session; store factory in ctx
    ctx["repository"] = _WorkerRepository(async_session)

    logger.info("NEXUS worker startup complete — %d tools registered", len(tool_registry.list_tools()))


async def shutdown(ctx: dict) -> None:
    """Clean up worker resources."""
    logger.info("NEXUS worker shutting down...")
    mcp_adapter = ctx.get("mcp_adapter")
    if mcp_adapter:
        try:
            await mcp_adapter.disconnect_all()
        except Exception:
            pass

    redis_client = ctx.get("_redis_client")
    if redis_client:
        try:
            await redis_client.close()
        except Exception:
            pass


class _WorkerRepository:
    """Thin proxy that opens a fresh session per call to avoid stale state."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    async def save_execution(self, execution):
        from nexus.db.repository import Repository
        async with self._session_factory() as session:
            return await Repository(session).save_execution(execution)

    async def get_chain_seals(self, chain_id: str):
        from nexus.db.repository import Repository
        async with self._session_factory() as session:
            return await Repository(session).get_chain_seals(chain_id)


# ── WorkerSettings ────────────────────────────────────────────────────────────

try:
    from arq.connections import RedisSettings
    from nexus.config import config as _config

    class WorkerSettings:
        functions = [execute_workflow_task, refresh_mcp_connections_task]
        on_startup = startup
        on_shutdown = shutdown
        redis_settings = RedisSettings.from_dsn(_config.task_queue_url)
        max_jobs = _config.worker_concurrency
        job_timeout = 3600
        max_tries = 3
        keep_result = 86400

except ImportError:
    # arq not installed — WorkerSettings unavailable (tests can still import task functions)
    WorkerSettings = None  # type: ignore[assignment,misc]
