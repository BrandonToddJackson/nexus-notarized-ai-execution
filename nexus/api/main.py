"""FastAPI application factory with lifespan management."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from nexus.config import config
from nexus.version import __version__


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    # ── Startup ──
    logger.info(f"NEXUS v{__version__} starting...")

    # 1. Database
    from nexus.db.database import init_db, async_session
    await init_db()

    # 2. Redis
    from nexus.cache.redis_client import RedisClient
    redis_client = RedisClient()
    app.state.redis = redis_client

    # 3. Session factory (for per-request repositories)
    app.state.async_session = async_session

    # 4. Seed DB (idempotent) and build a one-time repository for startup use
    async with async_session() as session:
        from nexus.db.repository import Repository
        repo = Repository(session)

        # Ensure the demo tenant exists
        from nexus.db.seed import seed_database
        await seed_database(session)

        # 5. Embedding service
        from nexus.knowledge.embeddings import EmbeddingService
        embedding_service = EmbeddingService(model_name=config.embedding_model)
        app.state.embedding_service = embedding_service

        # 6. Knowledge store
        from nexus.knowledge.store import KnowledgeStore
        knowledge_store = KnowledgeStore(
            persist_dir=config.chroma_persist_dir,
            embedding_fn=embedding_service.embed,
        )
        app.state.knowledge_store = knowledge_store

        # 7. Load personas from DB → PersonaManager
        from nexus.core.personas import PersonaManager
        from nexus.types import PersonaContract, RiskLevel
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
        app.state.persona_manager = persona_manager

    # 8. Tool registry — register all built-in tools
    from nexus.tools.registry import ToolRegistry
    from nexus.tools.plugin import get_registered_tools
    import nexus.tools.builtin  # noqa: F401 — triggers @tool registrations
    tool_registry = ToolRegistry()
    for tool_name, (definition, impl) in get_registered_tools().items():
        tool_registry.register(definition, impl)
    app.state.tool_registry = tool_registry

    # 9. Build all remaining components
    from nexus.core.anomaly import AnomalyEngine
    from nexus.core.notary import Notary
    from nexus.core.ledger import Ledger
    from nexus.core.chain import ChainManager
    from nexus.core.verifier import IntentVerifier
    from nexus.core.output_validator import OutputValidator
    from nexus.core.cot_logger import CoTLogger
    from nexus.knowledge.context import ContextBuilder
    from nexus.tools.selector import ToolSelector
    from nexus.tools.sandbox import Sandbox
    from nexus.tools.executor import ToolExecutor
    from nexus.reasoning.think_act import ThinkActGate
    from nexus.reasoning.continue_complete import ContinueCompleteGate
    from nexus.reasoning.escalate import EscalateGate
    from nexus.cache.fingerprints import FingerprintCache

    fingerprint_store = FingerprintCache(redis_client)
    anomaly_engine = AnomalyEngine(
        config=config,
        embedding_service=embedding_service,
        fingerprint_store=fingerprint_store,
    )
    notary = Notary()
    ledger = Ledger()  # in-memory; DB persistence via repository in routes
    app.state.ledger = ledger

    chain_manager = ChainManager()
    verifier = IntentVerifier()
    output_validator = OutputValidator()
    cot_logger = CoTLogger()
    context_builder = ContextBuilder(knowledge_store=knowledge_store)
    # LLM client — enables intelligent task decomposition + tool selection via Ollama
    from nexus.llm.client import LLMClient
    llm_client = LLMClient(task_type="general")

    tool_selector = ToolSelector(registry=tool_registry, llm_client=llm_client)
    sandbox = Sandbox()
    from nexus.credentials.encryption import CredentialEncryption
    from nexus.credentials.vault import CredentialVault
    credential_encryption = CredentialEncryption(key=config.credential_encryption_key)
    vault = CredentialVault(encryption=credential_encryption)
    app.state.vault = vault
    tool_executor = ToolExecutor(registry=tool_registry, sandbox=sandbox, verifier=verifier, vault=vault)
    think_act_gate = ThinkActGate()
    continue_complete_gate = ContinueCompleteGate()
    escalate_gate = EscalateGate()

    # 10. NexusEngine (event_bus injected below — forward reference resolved via set_event_bus)
    from nexus.core.engine import NexusEngine
    from nexus.triggers import EventBus, TriggerManager, CronScheduler, WebhookHandler
    from nexus.workflows.manager import WorkflowManager

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
    app.state.engine = engine

    # 11. Trigger system
    async with async_session() as trigger_session:
        from nexus.db.repository import Repository as TriggerRepo
        trigger_repo = TriggerRepo(trigger_session)
        workflow_manager = WorkflowManager(repository=trigger_repo, config=config)
        trigger_manager = TriggerManager(engine, workflow_manager, trigger_repo, event_bus, config)
        cron_scheduler = CronScheduler(trigger_manager, config)
        webhook_handler = WebhookHandler(trigger_manager, trigger_repo)
        trigger_manager.set_cron_scheduler(cron_scheduler)
        await cron_scheduler.start()

    app.state.event_bus        = event_bus
    app.state.workflow_manager = workflow_manager
    app.state.trigger_manager  = trigger_manager
    app.state.webhook_handler  = webhook_handler
    app.state.cron_scheduler   = cron_scheduler

    # 12. AmbiguityResolver + WorkflowGenerator (Phase 23.1)
    from nexus.workflows.ambiguity import AmbiguityResolver
    from nexus.workflows.generator import WorkflowGenerator

    # 13. SkillManager (Phase 25)
    from nexus.skills.manager import SkillManager
    skill_manager = SkillManager(
        repository=None,
        embedding_service=embedding_service if hasattr(app.state, 'embedding_service') else None,
        config=config,
    )
    app.state.skill_manager = skill_manager

    # 14. MCP adapter (Phase 25)
    from nexus.mcp.client import MCPClient
    from nexus.mcp.adapter import MCPToolAdapter
    mcp_client = MCPClient()
    mcp_adapter = MCPToolAdapter(tool_registry, mcp_client, None, vault)
    app.state.mcp_adapter = mcp_adapter

    app.state.ambiguity_resolver = AmbiguityResolver(
        llm_client=llm_client,
        tool_registry=tool_registry,
        persona_manager=persona_manager,
        config=config,
    )
    app.state.workflow_generator = WorkflowGenerator(
        llm_client=llm_client,
        tool_registry=tool_registry,
        persona_manager=persona_manager,
        workflow_manager=workflow_manager,
        config=config,
        skill_manager=skill_manager,
    )

    # 15. ARQ pool + WorkflowDispatcher (Phase 26)
    try:
        from arq import create_pool
        from arq.connections import RedisSettings
        from nexus.workers.dispatcher import WorkflowDispatcher

        arq_pool = await create_pool(RedisSettings.from_dsn(config.task_queue_url))
        app.state.arq_pool = arq_pool
        app.state.dispatcher = WorkflowDispatcher(engine, arq_pool, config)
        trigger_manager.set_dispatcher(app.state.dispatcher)
        logger.info("ARQ pool connected — background workers enabled")
    except ImportError:
        app.state.arq_pool = None
        app.state.dispatcher = None
        logger.warning("arq not installed — background workers disabled")

    logger.info(f"NEXUS v{__version__} ready — {len(tool_registry.list_tools())} tools registered")

    yield

    # ── Shutdown ──
    logger.info("NEXUS shutting down...")
    await cron_scheduler.stop()
    if getattr(app.state, "arq_pool", None) is not None:
        await app.state.arq_pool.close()
    await redis_client.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NEXUS",
        description="Notarized AI Execution — The agent framework where AI actions are accountable.",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )

    # Auth middleware (needs repository — injected per-request via app.state)
    from nexus.auth.middleware import AuthMiddleware
    from nexus.auth.jwt import JWTManager
    app.add_middleware(AuthMiddleware, jwt_manager=JWTManager())

    # Security headers — outermost middleware, applied to all responses
    app.add_middleware(SecurityHeadersMiddleware)

    # Routes
    from nexus.api.routes import execute, stream, ledger, personas, tools, knowledge, health, auth, workflows
    from nexus.api.routes import skills, credentials, mcp_servers, executions, events
    from nexus.api.routes.jobs import router as jobs_router
    app.include_router(execute.router, prefix="/v1")
    app.include_router(stream.router, prefix="/v1")
    app.include_router(ledger.router, prefix="/v1")
    app.include_router(personas.router, prefix="/v1")
    app.include_router(tools.router, prefix="/v1")
    app.include_router(knowledge.router, prefix="/v1")
    app.include_router(health.router, prefix="/v1")
    app.include_router(auth.router, prefix="/v1")
    app.include_router(workflows.router)
    app.include_router(skills.router, prefix="/v2")
    app.include_router(credentials.router, prefix="/v2")
    app.include_router(mcp_servers.router, prefix="/v2")
    app.include_router(executions.router, prefix="/v2")
    app.include_router(events.router, prefix="/v2")
    app.include_router(jobs_router, prefix="/v2/jobs", tags=["jobs"])

    return app


app = create_app()
