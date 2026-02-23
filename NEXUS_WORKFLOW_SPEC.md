# NEXUS v2 — PHASED PROMPT BUILD
## The n8n Eclipse: From AI Agent Framework → AI Automation Platform

> **Prerequisite:** All original Phases 0–14 are complete and passing. This picks up where that left off.
> **Goal:** Transform NEXUS from a single-shot AI agent framework into a persistent, trigger-driven, visually-editable automation platform that replaces n8n — with AI reasoning + cryptographic accountability on top.
> **Rule:** Same as before — if something isn't specified here, it's not in v2. Don't improvise. Every step still passes through all 4 gates and gets sealed. That's non-negotiable.

---

## BUILD PROGRESS (v2)

| Phase | Status | Tests | Notes |
|-------|--------|-------|-------|
| Phase 15: Foundation v2 | ✅ COMPLETE | — | v2 types (WorkflowDefinition, WorkflowStep, WorkflowEdge, WorkflowExecution, TriggerConfig, CredentialRecord, MCPServerConfig); v2 exceptions (WorkflowValidationError, CredentialError, MCPConnectionError, SandboxError, etc.); v2 config fields; v2 ORM models (WorkflowModel, WorkflowExecutionModel, TriggerModel, CredentialModel, MCPServerModel) |
| Phase 16: Workflow Definition | ✅ COMPLETE | 118/118 ✓ | nexus/workflows/: dag.py (topological_sort Kahn's BFS, evaluate_condition safe AST walker — no eval()), validator.py (9 structural checks + LOOP_BACK exclusion from cycle/entry-point checks), manager.py (full lifecycle: create, update, activate, pause, archive, version history, rollback, duplicate, export/import) |
| Phase 17: DAG Execution Engine | ✅ COMPLETE | 74/74 ✓ | engine.py extended: run_workflow (loads workflow, verifies active, builds context, entry points excluding LOOP_BACK), _execute_dag_layer (recursive DAG traversal via context["_done"]), _execute_action_step (full 4-gate pipeline per step — all gates apply), _execute_branch_step (condition routing + marks unchosen branches in _done), _execute_loop_step (self-loop model: LOOP_BACK is structural marker, iterates own tool_name N times), _execute_parallel_step (edge-based siblings + config["branches"] fan-out), _execute_sub_workflow_step (recursive run_workflow), _execute_approval_step (wait/poll/escalate), _resolve_params ({{step.field}} and {{trigger.field}} template resolution). run() fully backwards-compatible. 25/25 real integration checks with live engine (no mocks). |
| Phase 18: Credential Vault | ✅ COMPLETE | 45/45 ✓ | nexus/credentials/: CredentialEncryption (Fernet), CredentialVault (CRUD, inject_credentials, get_env_vars, refresh_oauth2), sanitize_tool_params; ToolExecutor wired; 45 tests |
| Phase 19: MCP Integration | ✅ COMPLETE | 118/118 ✓ | nexus/mcp/: MCPClient (stdio/sse/streamable_http, namespaced tools, env merge), MCPToolAdapter (register/unregister/reconnect_all, vault credential injection); MCPServerConfig.credential_id; proven live with sequential-thinking + Context7 |
| Phase 20: Universal HTTP Tool | ✅ COMPLETE | 74/74 ✓ | nexus/tools/builtin/http_request.py — http_request registered directly in _registered_tools (complex nested schema); method/url/headers/query_params/body/body_format (json/form/multipart/graphql/raw)/response_format (auto/json/text/binary)/response_path (JMESPath)/timeout/max_retries/retry_on/verify_ssl/http2/proxy/response_size_limit_kb/auth (basic/bearer/api_key)/pagination (cursor/offset/link_header)/rate_limit; 4xx/5xx returns error dict; client.stream() for size-limited reads. nexus/tools/builtin/data_transform.py — 15 operations: filter (14 operators + dot-notation), map (template/math_op), sort, group_by, flatten, pick, omit, rename, deduplicate, limit, skip, aggregate (sum/avg/min/max/count/first/last/concat/distinct), merge, set, cast; deep-copies input; jmespath>=1.0.0 added to pyproject.toml |
| Phase 21: Code Sandbox v2 | ✅ COMPLETE | 95/96 ✓ (1 skip) | nexus/tools/sandbox_v2.py — CodeSandbox with execute_python (ast.parse import validation, venv creation, resource.RLIMIT_AS memory limit, runpy wrapper), execute_javascript (ESM .mjs / CJS .cjs, preamble inject, npm install), execute_typescript (3-tier tsx resolution: A=global binary, B=shared cache dir — tsx installed once reused across calls, C=per-execution install; process-group kill via os.setsid+os.killpg to terminate tsx wrapper + child node process on timeout; TSX_TSCONFIG_PATH env, default tsconfig ES2022/bundler); sandbox_ts_cache_dir config field for shared tsx install; registered as code_execute_python/code_execute_javascript/code_execute_typescript (RiskLevel.MEDIUM, resource_pattern=code:*); NEXUS_* env var keys filtered; network isolation via proxy env injection; auto/json/text output formats; shutil.rmtree cleanup in finally block |
| Phase 22: Trigger System | 🔲 TODO | — | Webhooks, cron scheduler, event bus |
| Phase 23: NL Workflow Generation | 🔲 TODO | — | LLM → validated DAG via WorkflowGenerator + refine/explain |
| Phase 24: Visual Canvas | 🔲 TODO | — | React Flow drag-and-drop workflow editor with NL generation |
| Phase 25: Frontend v2 | 🔲 TODO | — | Credentials UI, MCP servers, execution history, marketplace browser |
| Phase 26: Background Execution | 🔲 TODO | — | ARQ task queue, async workflow runs, status polling |
| Phase 27: Plugin Marketplace | 🔲 TODO | — | Plugin SDK, PyPI registry, install/uninstall |
| Phase 28: Persistence v2 | 🔲 TODO | — | Alembic migrations for all v2 models |
| Phase 29: API v2 | 🔲 TODO | — | /v2/ routes: workflows, triggers, credentials, webhooks, MCP, marketplace |
| Phase 30: Tests v2 | 🔲 TODO | — | Full v2 test coverage (dag_engine, credentials, triggers, mcp, http_tool, sandbox) |
| Phase 31: Infrastructure v2 | 🔲 TODO | — | Worker/scheduler services, Docker updates, nginx config |
| Phase 32: Examples & Docs v2 | 🔲 TODO | — | Slack→Sheets, email classifier, scheduled report, MCP GitHub examples; README v2 section |

**Last verified:** 2026-02-23
**Test suite:** `.venv312/bin/pytest tests/ -m "not slow"` → **1213/1213 passed** (v1 phases 0-14 + v2 phases 15-21 + TypeScript amendment)

---

## BUILD ORDER (Strict — Follow This Sequence)

```
Phase 15: Foundation v2        → New types, enums, config, DB models for workflows/triggers/credentials
Phase 16: Workflow Definition  → DAG schema, workflow CRUD, versioning, step types (branch/loop/parallel)
Phase 17: DAG Execution Engine  → Evolve engine.py from linear chains to DAG execution with parallel, branch, loop, retry
Phase 18: Credential Vault     → Encrypted secret store, OAuth2 flows, credential injection at execution time
Phase 19: MCP Integration      → Model Context Protocol client, MCP tool adapter, federated tool registry
Phase 20: Universal HTTP Tool  → Generic REST API caller with auth injection, response mapping, pagination
Phase 21: Code Sandbox v2      → Python/JS execution sandbox with pip/npm, stdin/stdout, timeout, memory limits
Phase 22: Trigger System       → Webhooks, cron scheduler, event bus, external adapters
Phase 23: NL Workflow Gen      → Natural language → DAG generation via LLM, iterative refinement
Phase 24: Visual Canvas        → React Flow canvas, drag-and-drop nodes, edge wiring, hybrid AI+manual editing
Phase 25: Frontend v2          → Credential management UI, workflow list/editor, marketplace browser, execution history
Phase 26: Background Execution → Task queue (ARQ/Celery), async workflow runs, execution logs, status polling
Phase 27: Plugin Marketplace   → Plugin SDK, package registry, install/uninstall, community templates
Phase 28: Persistence v2       → Alembic migrations for all new models, seed data, data migration from v1
Phase 29: API v2               → All new routes (workflows, triggers, credentials, marketplace, webhook receivers)
Phase 30: Tests v2             → Full test coverage for Phases 15–29
Phase 31: Infrastructure v2   → Docker updates, scheduler service, worker service, nginx config
Phase 32: Examples & Docs v2   → Working examples: "Slack→Sheets", "Email→CRM", "Webhook→Pipeline", README update
```

---

## TABLE OF CONTENTS

1. [Phase 15: Foundation v2](#phase-15--foundation-v2)
2. [Phase 16: Workflow Definition Layer](#phase-16--workflow-definition-layer)
3. [Phase 17: DAG Execution Engine](#phase-17--dag-execution-engine)
4. [Phase 18: Credential Vault](#phase-18--credential-vault)
5. [Phase 19: MCP Integration](#phase-19--mcp-integration)
6. [Phase 20: Universal HTTP Tool](#phase-20--universal-http-tool)
7. [Phase 21: Code Sandbox v2](#phase-21--code-sandbox-v2)
8. [Phase 22: Trigger System](#phase-22--trigger-system)
9. [Phase 23: NL Workflow Generation](#phase-23--nl-workflow-generation)
10. [Phase 24: Visual Canvas (Frontend)](#phase-24--visual-canvas-frontend)
11. [Phase 25: Frontend v2](#phase-25--frontend-v2)
12. [Phase 26: Background Execution](#phase-26--background-execution)
13. [Phase 27: Plugin Marketplace](#phase-27--plugin-marketplace)
14. [Phase 28: Persistence v2](#phase-28--persistence-v2)
15. [Phase 29: API v2](#phase-29--api-v2)
16. [Phase 30: Tests v2](#phase-30--tests-v2)
17. [Phase 31: Infrastructure v2](#phase-31--infrastructure-v2)
18. [Phase 32: Examples & Docs v2](#phase-32--examples--docs-v2)
19. [Acceptance Criteria — v2](#acceptance-criteria--v2)

---

## Phase 15 — Foundation v2

Read the existing `nexus/types.py`, `nexus/config.py`, `nexus/exceptions.py`, and `nexus/db/models.py`. Then extend them with the new types needed for workflows, triggers, and credentials. Do NOT modify existing types — add alongside them.

### nexus/types.py — Add these new enums and models AFTER the existing ones:

```python
# New Enums:
WorkflowStatus     # draft, active, paused, archived, error
TriggerType        # manual, webhook, cron, event, workflow_complete
StepType           # action, branch, loop, parallel, sub_workflow, wait, human_approval
CredentialType     # api_key, oauth2, basic_auth, bearer_token, custom
EdgeType           # default, conditional, error, loop_back
NodePosition       # Pydantic model with x: float, y: float (for canvas layout)

# New Models:
WorkflowDefinition # id, tenant_id, name, description, version, status: WorkflowStatus, trigger_config: dict,
                   # steps: list[WorkflowStep], edges: list[WorkflowEdge], created_at, updated_at,
                   # created_by: str, tags: list[str], settings: dict (timeout, retry_policy, error_workflow_id)
WorkflowStep       # id, workflow_id, step_type: StepType, name, description, tool_name: Optional[str],
                   # tool_params: dict, persona_name: str, position: NodePosition,
                   # config: dict (for branch conditions, loop iterators, parallel groups, sub_workflow_id),
                   # timeout_seconds: int, retry_policy: dict
WorkflowEdge       # id, workflow_id, source_step_id, target_step_id, edge_type: EdgeType,
                   # condition: Optional[str] (expression or "default"), data_mapping: dict
WorkflowExecution  # id, workflow_id, workflow_version: int, tenant_id, trigger_type: TriggerType,
                   # trigger_data: dict, chain_id: str, status: ChainStatus,
                   # started_at, completed_at, error: Optional[str], step_results: dict
TriggerConfig      # id, workflow_id, tenant_id, trigger_type: TriggerType, enabled: bool,
                   # config: dict (cron_expression, webhook_path, event_name, source_workflow_id),
                   # last_triggered_at, created_at
CredentialRecord   # id, tenant_id, name, credential_type: CredentialType, service_name: str,
                   # encrypted_data: str, scoped_personas: list[str], created_at, updated_at, expires_at
MCPServerConfig    # id, tenant_id, name, url: str, transport: str ("stdio"|"sse"|"streamable_http"),
                   # command: Optional[str], args: list[str], env: dict, enabled: bool,
                   # discovered_tools: list[str], last_connected_at
```

### nexus/exceptions.py — Add these new exception classes:

```python
WorkflowError(NexusError)         # base for workflow-related errors
WorkflowNotFound(WorkflowError)   # workflow ID does not exist or wrong tenant
WorkflowValidationError(WorkflowError)  # DAG has cycles, disconnected nodes, missing edges
TriggerError(NexusError)          # trigger creation/execution failed
CredentialError(NexusError)       # credential encryption/decryption/injection failed
CredentialNotFound(CredentialError)  # credential ID not found or wrong tenant
MCPConnectionError(NexusError)    # MCP server connection failed
MCPToolError(ToolError)           # MCP tool execution failed
SandboxError(ToolError)           # code sandbox execution failed (memory limit, timeout, forbidden import)
```

### nexus/config.py — Add these config fields to NexusConfig (AFTER existing fields):

```python
# ── Workflows ──
max_workflow_steps: int = 50                # max steps per workflow
max_concurrent_workflows: int = 10          # per tenant
workflow_execution_timeout: int = 3600      # 1 hour max per workflow run

# ── Triggers ──
webhook_base_url: str = "http://localhost:8000"  # public base URL for webhook URLs
cron_check_interval: int = 15               # seconds between cron checks
max_triggers_per_workflow: int = 5

# ── Credentials ──
credential_encryption_key: str = ""         # 32-byte Fernet key; generate with: Fernet.generate_key()
credential_max_per_tenant: int = 100

# ── MCP ──
mcp_connection_timeout: int = 10            # seconds
mcp_tool_timeout: int = 60                  # per-tool execution timeout
mcp_max_servers: int = 20                   # per tenant

# ── Code Sandbox ──
sandbox_max_memory_mb: int = 256
sandbox_max_execution_seconds: int = 30
sandbox_allowed_imports: list[str] = ["json", "math", "re", "datetime", "collections", "itertools", "functools", "hashlib", "base64", "urllib.parse", "csv", "io"]

# ── Background Execution ──
worker_concurrency: int = 4
task_queue_url: str = "redis://localhost:6379/1"  # separate DB from cache
```

### nexus/db/models.py — Add these new SQLAlchemy models AFTER the existing ones:

```python
WorkflowModel       # Table "workflows": id (UUID PK), tenant_id (FK), name, description, version (int),
                    # status (varchar), trigger_config (JSON), steps (JSON), edges (JSON),
                    # settings (JSON), tags (JSON), created_by, created_at, updated_at.
                    # Index on (tenant_id, status). Unique constraint on (tenant_id, name, version).

WorkflowExecutionModel  # Table "workflow_executions": id (UUID PK), workflow_id (FK), workflow_version,
                        # tenant_id (FK), trigger_type, trigger_data (JSON), chain_id,
                        # status, started_at, completed_at, error, step_results (JSON).
                        # Index on (tenant_id, workflow_id, started_at).

TriggerModel        # Table "triggers": id (UUID PK), workflow_id (FK), tenant_id (FK), trigger_type,
                    # enabled (bool), config (JSON), webhook_path (varchar unique),
                    # last_triggered_at, created_at.
                    # Index on (tenant_id, enabled). Index on (webhook_path) for fast lookup.

CredentialModel     # Table "credentials": id (UUID PK), tenant_id (FK), name, credential_type,
                    # service_name, encrypted_data (text), scoped_personas (JSON),
                    # created_at, updated_at, expires_at.
                    # Index on (tenant_id, service_name).

MCPServerModel      # Table "mcp_servers": id (UUID PK), tenant_id (FK), name, url, transport,
                    # command, args (JSON), env (JSON), enabled (bool),
                    # discovered_tools (JSON), last_connected_at, created_at.
                    # Index on (tenant_id, enabled).
```

Don't create migration files yet — that's Phase 28. Just add the models.

---

## Phase 16 — Workflow Definition Layer

Create the workflow definition system. This is the missing persistence layer that turns ad-hoc chains into reusable, versionable automation blueprints.

### nexus/workflows/__init__.py

Empty init, exports `WorkflowManager`, `WorkflowValidator`.

### nexus/workflows/validator.py — Implement WorkflowValidator class:

```python
validate(workflow: WorkflowDefinition) → list[str]  # errors, empty = valid

# Checks:
# 1. DAG acyclicity — topological sort of steps via edges. If cycle detected, return error with the cycle path.
#    Use Kahn's algorithm: compute in-degrees, BFS from zero-in-degree nodes. If not all nodes visited → cycle.
# 2. Connectivity — every step must be reachable from at least one step with no incoming edges (entry points).
#    Disconnected nodes are errors.
# 3. Edge validity — every edge's source_step_id and target_step_id must reference existing step IDs.
# 4. Branch completeness — every step of type "branch" must have at least 2 outgoing edges,
#    one of which has edge_type "default" (the else case).
# 5. Loop safety — loop steps must have exactly one outgoing edge with edge_type "loop_back"
#    AND one with edge_type "default" (exit condition).
# 6. Parallel convergence — parallel steps must have a downstream step where all parallel branches converge.
#    (Warn if missing, don't hard-fail.)
# 7. Step limit — len(steps) <= config.max_workflow_steps.
# 8. Tool references — every step with tool_name must reference a tool that exists in the ToolRegistry
#    (pass registry as optional param; skip check if None).
# 9. Persona references — every step's persona_name must exist in PersonaManager (pass as optional; skip if None).
```

### nexus/workflows/manager.py — Implement WorkflowManager class:

```python
__init__(self, repository, validator: WorkflowValidator, config: NexusConfig)

create(...)           # Validate, version=1, status=draft, persist. Return WorkflowDefinition.
update(...)           # If steps/edges changed: increment version, re-validate, persist as NEW version.
get(...)              # Load, verify tenant, return. 404 → WorkflowNotFound.
list(...)             # Filter by status, paginated.
activate(...)         # Validate, set status=active, enable triggers.
pause(...)            # status=paused, disable triggers.
archive(...)          # status=archived, disable triggers.
get_version_history(...)
rollback(...)         # Load target version, create new version with that steps/edges.
duplicate(...)        # Deep copy, version=1, status=draft.
export_json(...)      # Full JSON export.
import_json(...)      # Validate and import, remap IDs.
```

### nexus/workflows/dag.py — Implement DAG utilities:

```python
topological_sort(steps, edges) → list[str]   # Raises WorkflowValidationError on cycle.
get_entry_points(steps, edges) → list[str]
get_exit_points(steps, edges) → list[str]
get_children(step_id, edges) → list[tuple[str, WorkflowEdge]]
get_parents(step_id, edges) → list[tuple[str, WorkflowEdge]]
get_parallel_group(step_id, steps, edges) → list[str]
evaluate_condition(condition: str, context: dict) → bool   # Safe expression eval; NO eval(). Use ast.literal_eval / manual parse.
```

---

## Phase 17 — DAG Execution Engine

Evolve `nexus/core/engine.py` from linear chain execution to full DAG execution. Keep existing `run()` for backwards compatibility. Add `run_workflow()` and layer/step execution methods.

### nexus/core/engine.py — Add:

- **run_workflow(workflow_id, tenant_id, trigger_data, persona_override)** — Load workflow, verify active, create WorkflowExecution and chain, build context, get entry points, call _execute_dag_layer.
- **_execute_dag_layer(step_ids, workflow, chain, context)** — For each step_id: lookup step, match step_type → _execute_action_step | _execute_branch_step | _execute_loop_step | _execute_parallel_step | _execute_sub_workflow_step | wait | _execute_approval_step. Then determine next layer via get_children + condition evaluation, recurse.
- **_execute_action_step** — Resolve params from context, build KB context, think/act gate, IntentDeclaration, activate persona, anomaly check, seal (PENDING), gate fail → BLOCKED + ledger + revoke; else verify intent → execute tool → validate → finalize seal → ledger → store result in context["steps"].
- **_execute_branch_step** — Evaluate conditional edges, select branch, _execute_dag_layer(selected_step_id).
- **_execute_loop_step** — Resolve iterator from context, for each item set context["loop_current"], context["loop_index"], execute body via _execute_dag_layer, then follow default exit edge.
- **_execute_parallel_step** — asyncio.gather on branches; fail_fast or collect results.
- **_execute_sub_workflow_step** — run_workflow(sub_workflow_id, ...), store execution in context.
- **_execute_approval_step** — Create approval request, callback/webhook, poll with timeout; approved → continue, denied → fail, timeout → escalate.
- **_resolve_params(params, context)** — Replace {{step_name.field}} and {{trigger.field}} via regex and context navigation.

---

## Phase 18 — Credential Vault

Encrypted credential storage. Tenant- and persona-scoped; never exposed to AI or ledger.

### nexus/credentials/encryption.py

- **CredentialEncryption** — Fernet init (validate key; if empty, generate and log WARNING). `encrypt(plaintext)` / `decrypt(ciphertext)`; InvalidToken → CredentialError.

### nexus/credentials/vault.py — CredentialVault:

- **store** — Validate, encrypt JSON data, persist record (return without encrypted_data).
- **retrieve** — Load, tenant check, persona scope check, expiry check, decrypt, return dict.
- **update** / **delete** / **list** (metadata only).
- **inject_credentials** — Retrieve, merge into tool_params by credential_type (api_key, oauth2, basic_auth, bearer_token, custom).
- **refresh_oauth2** — POST to token_url, update stored access_token/expires_at.

Executor: call vault after gates, before tool run; **sanitize params** (strip Authorization, passwords, tokens) before sealing.

---

## Phase 19 — MCP Integration

### nexus/mcp/client.py — MCPClient:

- **connect(server_config)** — By transport (stdio / sse / streamable_http): create session, initialize, list_tools; convert to NEXUS ToolDefinition (namespaced `mcp_{server_name}_{tool_name}`); store session and tools.
- **call_tool(server_id, tool_name, params)** — Strip namespace, call tool, return content or raise MCPToolError.
- **disconnect** / **disconnect_all** / **refresh_tools**.

### nexus/mcp/adapter.py — MCPToolAdapter:

- **register_server(tenant_id, server_config)** — Connect, register wrapper fns in ToolRegistry (source="mcp"), persist config.
- **unregister_server** — Remove tools, disconnect, delete from repo.
- **list_servers** / **reconnect_all** (on API startup).

### nexus/tools/registry.py — Add:

- **register(..., source="local")**, **get_by_source(source)**, **unregister(name)**.

---

## Phase 20 — Universal HTTP Tool

### nexus/tools/builtin/http_request.py

- Tool **http_request**: url, method, headers, body, query_params, credential_id (injected by executor), response_path (JMESPath), timeout, follow_redirects, pagination (cursor/offset). Use httpx; on 4xx/5xx return error dict; on connection error raise ToolError.

### nexus/tools/builtin/data_transform.py

- Tool **data_transform**: input_data, operations (filter, map, sort, group_by, flatten, pick, omit, rename, deduplicate, limit, aggregate). Apply sequentially; safe evaluation only.

---

## Phase 21 — Code Sandbox v2

### nexus/tools/sandbox_v2.py — CodeSandbox:

- **execute_python(code, stdin, timeout, max_memory_mb)** — ast.parse, validate imports against config.sandbox_allowed_imports; temp dir, subprocess with resource limits; return {stdout, stderr, exit_code}; timeout → SandboxError.
- **execute_javascript(code, stdin, timeout)** — Same pattern with node.

Register built-in tools: **code_execute_python**, **code_execute_javascript** (RiskLevel.MEDIUM).

---

## Phase 22 — Trigger System

### nexus/triggers/manager.py — TriggerManager:

- **create_trigger(tenant_id, workflow_id, trigger_type, config)** — Validate workflow and type-specific config (webhook: path auto-generated; cron: expression; event: event_name; workflow_complete: source_workflow_id). Persist; register cron with CronScheduler, event with EventBus.
- **enable** / **disable** / **delete** / **list**.
- **fire(trigger, trigger_data)** — Verify enabled, engine.run_workflow(), update last_triggered_at.

### nexus/triggers/webhook.py — WebhookHandler:

- **handle(webhook_path, method, headers, body, query_params)** — Lookup trigger by path, build trigger_data, trigger_manager.fire().

### nexus/triggers/cron.py — CronScheduler:

- **start** — Load enabled cron triggers, register each; loop: sleep cron_check_interval, check_and_fire.
- **register(trigger)** / **unregister(trigger_id)** / **check_and_fire** (croniter for next run).

### nexus/triggers/event_bus.py — EventBus:

- **subscribe(event_name, callback)** / **unsubscribe** / **emit(event_name, data)**.
- Predefined: workflow.completed, workflow.failed, seal.blocked. Engine emits after workflow completion.

---

## Phase 23 — NL Workflow Generation

### nexus/workflows/generator.py — WorkflowGenerator:

- **generate(tenant_id, description, context)** — Build prompt with tools/personas/credentials, LLM complete, parse JSON, validate, create via workflow_manager; on validation fail call refine.
- **refine(tenant_id, workflow_id, feedback)** — Load workflow, refinement prompt with feedback/errors, LLM, validate, update (new version).
- **explain(tenant_id, workflow_id)** — LLM explanation of workflow in plain English.

### nexus/llm/prompts.py — Add:

- **GENERATE_WORKFLOW**, **REFINE_WORKFLOW**, **EXPLAIN_WORKFLOW** (templates as specified).

---

## Phase 24 — Visual Canvas (Frontend)

- **Dependencies:** `reactflow`, `dagre`.
- **frontend/src/pages/WorkflowEditor.jsx** — Full-screen React Flow; left toolbox (draggable node types); right properties panel (by node type); top: name, Save, Activate/Pause, Run Now, NL input → generate; version + history; bottom: validation status, zoom, fit view, minimap. Custom nodes (Action, Branch, Loop, Parallel, Sub-Workflow, Wait, Approval, Trigger). Save → POST /v2/workflows; load → convert to React Flow; NL → POST generate, dagre layout.
- **frontend/src/components/workflow/** — ActionNode, BranchNode, LoopNode, ParallelNode, TriggerNode, PropertiesPanel (tool/persona/param/credential selectors, condition editor, cron/webhook config).

---

## Phase 25 — Frontend v2

- **Workflows.jsx** — List workflows; filters; New Workflow; Generate with AI; card → editor; duplicate, export, archive, delete.
- **Credentials.jsx** — List credentials (metadata); Add Credential modal (type-specific fields, OAuth flow, persona scope); never show secrets; reveal toggle 5s.
- **MCPServers.jsx** — List servers; Add with Test Connection; show discovered tools; reconnect, remove.
- **Executions.jsx** — List workflow executions; expand chain + seals; filter; Replay in editor (read-only + gate overlay).
- **App.jsx** — Nav: Automation (Workflows, Executions), Connections (Credentials, MCP Servers), Build (Tools, Personas, Knowledge); Dashboard, Execute unchanged.

---

## Phase 26 — Background Execution

### nexus/workers/queue.py

- ARQ: **execute_workflow_task(ctx, workflow_id, tenant_id, trigger_data)**, **refresh_mcp_connections_task(ctx, tenant_id)**. WorkerSettings with startup (create engine/deps), shutdown, redis from config.

### nexus/workers/dispatcher.py — WorkflowDispatcher:

- **dispatch(workflow_id, tenant_id, trigger_data, force_background)** — If force_background or webhook/cron or steps > 5 → enqueue job; else run inline. Return {job_id, status} or {execution_id, status}.
- **get_job_status(job_id)** — Poll ARQ job.

---

## Phase 27 — Plugin Marketplace

### nexus/marketplace/plugin_sdk.py

- **PluginManifest** (name, version, description, author, tools, personas, dependencies, nexus_version, homepage, license). Package layout: nexus_plugin_&lt;name&gt;, manifest, tools (e.g. @nexus_tool), optional personas.yaml.

### nexus/marketplace/registry.py — PluginRegistry:

- **install(package_name)** — pip install, import, get manifest, register tools (source="marketplace"), store in _installed.
- **uninstall(plugin_name)** — Unregister tools, pip uninstall.
- **list_installed** / **search(query)** (PyPI API).

---

## Phase 28 — Persistence v2

### nexus/db/repository.py — Add:

- Workflows: save_workflow, get_workflow, list_workflows, update_workflow, delete_workflow.
- Executions: save_execution, get_execution, list_executions, update_execution.
- Triggers: save_trigger, get_trigger, get_trigger_by_webhook_path, list_triggers, update_trigger, delete_trigger.
- Credentials: save_credential, get_credential, list_credentials, update_credential, delete_credential.
- MCP: save_mcp_server, get_mcp_server, list_mcp_servers, update_mcp_server, delete_mcp_server.

**Alembic:** `alembic revision --autogenerate -m "v2_workflows_triggers_credentials"`. Verify all 5 new tables, columns, indexes.

---

## Phase 29 — API v2

Mount under `/v2/`. Add routes:

- **workflows.py** — CRUD, activate, pause, duplicate, versions, rollback, run, export, import, generate, refine, explain.
- **executions.py** — List, get, stream, delete.
- **triggers.py** — CRUD, enable, disable.
- **webhooks.py** — Catch-all `/webhooks/{path:path}` → WebhookHandler.handle (no JWT).
- **credentials.py** — CRUD, test.
- **mcp.py** — Servers CRUD, reconnect, refresh.
- **marketplace.py** — search, install, uninstall, installed.
- **jobs.py** — get status, get result.

**main.py lifespan:** Startup: CredentialVault, WorkflowManager, MCPClient + adapter (reconnect_all), TriggerManager, WebhookHandler, CronScheduler.start(), EventBus, WorkflowDispatcher, WorkflowGenerator. Shutdown: stop scheduler, disconnect MCP, close worker pool.

---

## Phase 30 — Tests v2

- **tests/test_workflows.py** — create, invalid DAG, versioning, activate/pause, rollback, duplicate, export/import, topological_sort, branch evaluation, loop, parallel, parameter resolution.
- **tests/test_dag_engine.py** — linear, branch, loop, parallel, gates still apply, error handling, sub_workflow, all seals notarized.
- **tests/test_credentials.py** — store/retrieve, encryption, tenant isolation, persona scoping, expired, injection, oauth2 refresh, params sanitization.
- **tests/test_mcp.py** — connect/discover, tool execution, disconnect, gated, namespacing.
- **tests/test_triggers.py** — webhook create/fire, cron, event bus, enable/disable, unknown webhook 404.
- **tests/test_http_tool.py** — GET/POST, response_path, pagination, connection error.
- **tests/test_code_sandbox.py** — python run, forbidden import, timeout; javascript run; stdin.
- **tests/test_api_v2.py** — Workflow CRUD/run/generate, trigger CRUD, webhook receiver, credential CRUD, MCP CRUD, job status (TestClient).

---

## Phase 31 — Infrastructure v2

- **docker-compose.yml** — Add nexus-worker (arq), nexus-scheduler (cron_runner).
- **nexus/triggers/cron_runner.py** — Standalone asyncio entry for CronScheduler.
- **Dockerfile** — Ensure deps: mcp, arq, croniter, httpx, jmespath, cryptography, apscheduler.
- **pyproject.toml** — Optional dependency group `v2` with above.
- **frontend/package.json** — reactflow, dagre.

---

## Phase 32 — Examples & Docs v2

- **examples/slack_to_sheets/** — Webhook → extract → http_request + data_transform to Sheets (mock).
- **examples/email_classifier/** — Webhook → LLM classify → branch (billing / support / spam).
- **examples/scheduled_report/** — Cron → HTTP → LLM summary → email.
- **examples/mcp_github/** — MCP GitHub, list PRs, loop summarize.
- **examples/nl_workflow_generation/** — CLI: description → generate → explain → optional run.
- **README.md** — v2 section: persistent workflows, triggers, canvas + NL, MCP, credential vault, background execution, marketplace; comparison with n8n/Zapier; v2 architecture.

---

## ACCEPTANCE CRITERIA — v2

ALL of these must pass before v2 ships:

1. [ ] **Workflow CRUD works** — create, activate, pause, archive, version, rollback all functional
2. [ ] **DAG execution works** — linear, branch, loop, and parallel workflows all produce correct seals
3. [ ] **Every DAG step is still gated and sealed** — no step bypasses the 4 anomaly gates
4. [ ] **Webhook triggers fire workflows** — POST to webhook URL → workflow executes → seals created
5. [ ] **Cron triggers fire on schedule** — register cron, advance time, verify workflow runs
6. [ ] **NL → Workflow generation works** — describe workflow in English → valid DAG created on canvas
7. [ ] **Visual canvas renders and edits workflows** — React Flow canvas loads, nodes draggable, edges connectable, save persists
8. [ ] **Credentials are encrypted and persona-scoped** — store, retrieve, inject all work; secrets never in seals
9. [ ] **MCP tools are discoverable and gated** — connect MCP server → tools appear in registry → pass through all 4 gates
10. [ ] **HTTP tool calls external APIs** — GET/POST with auth injection and response extraction works
11. [ ] **Code sandbox executes safely** — Python/JS run in subprocess, forbidden imports blocked, timeouts enforced
12. [ ] **Background execution works** — large workflows dispatch to queue, status pollable
13. [ ] **All v1 functionality still works** — ad-hoc execute, CLI, existing dashboard, all v1 tests pass
14. [ ] **Tests pass** — `pytest tests/` with >80% green across both v1 and v2 tests
15. [ ] **Docker compose starts all services** — API, worker, scheduler, postgres, redis, frontend all healthy
