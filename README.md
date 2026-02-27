# NEXUS — Notarized AI Execution

> **Not multi-agent.** One agent, multiple personas. A persona is a constrained operating mode — not a separate entity.

> **The agent framework where AI actions are accountable.**

Every AI action is declared, verified, and sealed in an immutable ledger before execution. If it looks wrong, it's blocked.

```mermaid
flowchart LR
    T([User Task]) --> D[Decompose]
    D --> R[Retrieve Context]
    R --> I[Declare Intent]
    I --> P[Activate Persona]
    P --> G{4 Anomaly Gates}
    G -- fail --> B([Blocked & Sealed])
    G -- pass --> S[Notarize Seal]
    S --> E[Execute Tool]
    E --> V[Validate Output]
    V --> L[(Immutable Ledger)]
```

## What Makes NEXUS Different

| Feature | CrewAI | LangGraph | AutoGen | **NEXUS** |
|---------|--------|-----------|---------|-----------|
| Behavioral Contracts (Personas) | ❌ | ❌ | ❌ | ✅ |
| 4-Gate Anomaly Detection | ❌ | ❌ | ❌ | ✅ |
| Merkle-Chain Notarization | ❌ | ❌ | ❌ | ✅ |
| Immutable Audit Ledger | ❌ | ❌ | ❌ | ✅ |
| Intent vs Action Verification | ❌ | ❌ | ❌ | ✅ |
| Behavioral Drift Detection | ❌ | ❌ | ❌ | ✅ |
| Trust Graduation Tiers | ❌ | ❌ | ❌ | ✅ |
| Multi-tenant Isolation | ❌ | ❌ | ❌ | ✅ |
| Chain-of-Thought Capture | ❌ | ❌ | ❌ | ✅ |
| Persistent Workflow Automation | ❌ | partial | ❌ | ✅ |
| Encrypted Credential Vault | ❌ | ❌ | ❌ | ✅ |
| MCP Tool Integration | ❌ | ❌ | partial | ✅ |
| Code Sandbox (Python/JS/TypeScript) | ❌ | ❌ | ❌ | ✅ |
| RAG + Knowledge Scoping | ✅ | ✅ | ✅ | ✅ |
| Multi-provider LLM | ✅ | ✅ | ✅ | ✅ |
| Tool Framework | ✅ | ✅ | ✅ | ✅ |

## The 4 Anomaly Gates

Every action passes through 4 gates before execution:

```mermaid
flowchart LR
    A([Action]) --> G1{"Gate 1<br/>Scope<br/>Tool allowed?"}
    G1 -- fail --> B1([Blocked])
    G1 -- pass --> G2{"Gate 2<br/>Intent<br/>Cosine ≥ 0.75"}
    G2 -- fail --> B2([Blocked])
    G2 -- pass --> G3{"Gate 3<br/>TTL<br/>Not expired?"}
    G3 -- fail --> B3([Blocked])
    G3 -- pass --> G4{"Gate 4<br/>Drift<br/>Within 2.5σ?"}
    G4 -- fail --> B4([Blocked])
    G4 -- pass --> OK([Execute])
```

If **any** gate fails → action is **BLOCKED** and sealed as blocked in the ledger.

## Single engine vs multi-agent (first principles)

Research often shows that multiple specialized agents outperform a single agent on some tasks. NEXUS does **not** claim the single engine always wins on raw capability or throughput. It claims a **different tradeoff** and challenges the assumption that you need multiple agents to get specialization and safety.

| Dimension | Multi-agent frameworks | NEXUS (one engine, multiple personas) |
|-----------|------------------------|----------------------------------------|
| **Specialization** | Separate agents, separate LLM calls | One orchestrator; personas = constrained modes (allowed tools, intent patterns, TTL, drift). Same “brain,” different contracts. |
| **Auditability** | Scattered logs; no single chain of custody | One ledger, one notary, every action sealed. Easier to prove what happened. |
| **Throughput** | N agents ⇒ N concurrent tasks (at scale) | One chain at a time per engine instance. Scale out by running more instances. |
| **Workflows with multiple personas** | Natural: assign agents to steps | **Supported.** Each workflow step has a `persona_name`; the engine activates that persona for that step (scope, intent, TTL, drift). You can chain researcher → analyst → reviewer in one workflow. |
| **Parallel processing** | Often implicit (concurrent agents) | **Supported where it matters.** DAG workflows can use **PARALLEL** steps: branches run concurrently via `asyncio.gather`, each with its own persona and full 4-gate pipeline. Sibling steps in the same DAG layer are currently run sequentially; use an explicit PARALLEL step when you want concurrency. |

So: **Can the single engine “outperform” multi-agent?** On **safety, auditability, and a single chain of custody** — yes. On **raw task throughput or parallel “minds”** — no; we don’t pretend otherwise. We do support **multi-persona workflows** (different persona per step) and **parallel execution** (PARALLEL steps with multiple branches), so the single engine can still handle workflows that need several roles and concurrent work, without giving up one notarized trail.

## Quickstart

```bash
# Clone and install
git clone https://github.com/BrandonToddJackson/nexus-notarized-ai-execution.git
cd nexus-notarized-ai-execution
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Set NEXUS_LLM_API_KEY and NEXUS_SECRET_KEY in .env

# Start infrastructure
docker compose up postgres redis -d

# Seed with demo personas
make seed

# Run the API (port 8000)
make run

# In another terminal — get a JWT from the demo API key
TOKEN=$(curl -s -X POST http://localhost:8000/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key": "nxs_demo_key_12345"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

# Execute a task
curl -X POST http://localhost:8000/v1/execute \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"task": "What is NEXUS?"}'
```

### Frontend dashboard (port 5173)

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173 — log in with nxs_demo_key_12345
```

The dashboard streams gate results, seal cards, and CoT traces in real time as tasks execute.

## Architecture

### Full System Topology

```mermaid
flowchart TB
    CLIENT([API Client]) --> API
    WH([Webhook]) --> TM
    CR([Cron · Event]) --> TM

    subgraph API["API Layer (FastAPI)"]
        V1["v1 — Agent Execution"]
        V2["v2 — Workflow Automation"]
        AUTH["JWT · Rate Limiter\n60 req/min · 100 chains/hr"]
    end

    TM[Trigger Manager] --> WE[DAG Executor]

    subgraph ORCH["Orchestration"]
        ENG[NexusEngine]
        WE
        WM[Workflow Manager]
    end

    subgraph COG["Cognitive Plane"]
        KB["Knowledge Store (ChromaDB · RAG)"]
        CTX[Context Builder]
        TA[Think / Act Gate]
    end

    subgraph SEC["Security Plane — every action, always"]
        PM["Persona Manager\n(Trust Tiers)"]
        AE["Anomaly Engine\nGate 1 Scope · Gate 2 Intent\nGate 3 TTL · Gate 4 Drift"]
        NT["Notary · Verifier"]
    end

    subgraph EXEC["Execution Layer"]
        TR[Tool Registry]
        CV["Credential Vault\n(Fernet-encrypted)"]
        MCP["MCP Client\n(stdio · sse · http)"]
        SB["Code Sandbox\n(Python · JS · TypeScript)"]
        HTTP["HTTP Tool\n+ Data Transform"]
    end

    LED[("Merkle-Chain\nLedger")]

    subgraph STORE["Persistence"]
        PG[("PostgreSQL\nchains · personas\nworkflows · credentials")]
        RD[("Redis\nfingerprints · rate limits\ndistributed locks")]
    end

    API --> ORCH
    ORCH --> COG
    ORCH --> SEC
    COG --> CTX --> TA --> SEC
    SEC --> PM --> AE
    AE -->|"all 4 pass"| NT
    AE -->|"any fail"| LED
    NT --> EXEC
    NT --> LED
    EXEC --> TR
    EXEC --> CV
    EXEC --> MCP
    EXEC --> SB
    EXEC --> HTTP
    ORCH --> STORE
    LED --> PG
    KB --> PG
```

### Tool Ecosystem

```mermaid
flowchart LR
    subgraph BUILTIN["Built-in Tools"]
        WS[web_search · web_fetch]
        FILE[file_read · file_write]
        COMMS[send_email]
        DATA[knowledge_search · compute_stats]
        RAG["rag_ingest · rag_query\n(RAGAnything multimodal)"]
        INSTANTLY["instantly_audit · instantly_get_campaigns\ninstantly_create_campaign · instantly_activate_campaign\n+ 5 more campaign tools"]
        HTTP[http_request + data_transform\n15 pipeline operations]
        FE[generate_frontend_design]
    end

    subgraph SANDBOX["Code Sandbox (subprocess-isolated)"]
        PY["code_execute_python\nast import validation · venv · RLIMIT_AS"]
        JS["code_execute_javascript\nESM/CJS · npm install"]
        TS["code_execute_typescript\ntsx cache · process-group kill"]
    end

    subgraph MCP_TOOLS["MCP Tools (dynamic registration)"]
        MCPN["mcp_{server}_{tool} namespace"]
        MCPE["stdio · SSE · streamable_http\ntransports · credential injection"]
    end

    subgraph MARKET["Plugin Marketplace (community)"]
        PKG["nexus-plugin-{name} on PyPI\n@nexus_plugin_tool decorated\nSHA-256 tamper-verified"]
    end

    TR[Tool Registry] --> BUILTIN
    TR --> SANDBOX
    TR --> MCP_TOOLS
    TR --> MARKET
```

Same model throughout: one agent, multiple personas (see top of README). Workflows can use **multiple personas** (per-step `persona_name`) and **parallel branches** (PARALLEL step type).

## RAG Knowledge Tools

Two built-in tools for multimodal knowledge ingestion and retrieval — gated and sealed like every other tool call:

| Tool | Description | Risk |
|------|-------------|------|
| `rag_ingest` | Ingest text, URL, or uploaded file into a named knowledge namespace | MEDIUM |
| `rag_query` | Semantic search over ingested knowledge (hybrid mode by default) | LOW |

Use these to feed campaign briefs, ICP documents, or competitor research into a named namespace, then query that context in later workflow steps. Results flow through `session_history` into the next tool call automatically.

```bash
# Enable RAG
pip install -e ".[rag]"

# In .env
NEXUS_RAG_ANYTHING_ENABLED=true
NEXUS_RAG_ANYTHING_DIR=/tmp/nexus_rag   # working dir for vector store + parsed docs
```

```bash
# Ingest a campaign brief via the API
curl -X POST http://localhost:8000/v1/knowledge/multimodal \
  -H "Authorization: Bearer $TOKEN" \
  -F "content=Target audience: SaaS founders, 20-200 employees, using Salesforce" \
  -F "namespace=cold_campaign"

# Query it in a task
curl -X POST http://localhost:8000/v1/execute \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"task": "Find leads matching our ICP", "persona": "campaign_researcher"}'
```

## Campaign Automation (Instantly.ai)

Nine built-in tools cover the complete cold email campaign lifecycle — every call gated through all 4 anomaly gates and sealed in the ledger.

### Tools

| Tool | Purpose | Risk |
|------|---------|------|
| `instantly_audit` | Full health report: campaigns, lead counts, cross-campaign dedup, sender warmup, recommendations | LOW |
| `instantly_get_campaigns` | List all campaigns with status | LOW |
| `instantly_get_campaign_analytics` | Send/open/reply stats for a specific campaign | LOW |
| `instantly_get_leads` | Leads enrolled in a campaign | LOW |
| `instantly_get_sender_health` | Warmed sender accounts + daily send capacity | LOW |
| `instantly_add_leads` | Add leads to a campaign with automatic deduplication | MEDIUM |
| `instantly_move_leads` | Move leads between lists and campaigns | MEDIUM |
| `instantly_create_campaign` | Create campaign with email sequence + warmed senders (idempotent) | HIGH |
| `instantly_activate_campaign` | Launch campaign — emails begin sending on next scheduled window | HIGH |

`HIGH` risk tools have `requires_approval=True`. Gate 1 enforces persona scope; no campaign is created or activated unless the active persona explicitly has that tool in its `allowed_tools`.

### Campaign Personas

Three seeded personas map to the three phases of a campaign:

| Persona | Role | Key Tools |
|---------|------|-----------|
| `campaign_researcher` | ICP research, lead discovery, external data fetching | `rag_query`, `web_search`, `web_fetch`, `instantly_audit` |
| `campaign_outreach` | Build and launch campaigns, add leads | `instantly_create_campaign`, `instantly_add_leads`, `instantly_activate_campaign` |
| `campaign_optimizer` | Monitor performance, tune deliverability | `instantly_audit`, `instantly_get_sender_health`, `compute_stats` |

### Full Campaign Lifecycle

```bash
# 1. Audit current state — see all campaigns, sender warmup, duplicate leads
curl -X POST http://localhost:8000/v1/execute \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"task": "Run a full campaign audit", "persona": "campaign_optimizer"}'

# 2. Check sender capacity
curl -X POST http://localhost:8000/v1/execute \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"task": "Check sender health and warmup status", "persona": "campaign_optimizer"}'

# 3. Create a campaign (HIGH risk — Gate 1 enforces campaign_outreach scope)
curl -X POST http://localhost:8000/v1/execute \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "task": "Create a campaign called Q2 Outreach, subject Quick question about your data stack, body Hi {{first_name}} I noticed...",
    "persona": "campaign_outreach"
  }'

# 4. Add leads, then activate (returns campaign_id from previous step via session_history)
curl -X POST http://localhost:8000/v1/execute \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"task": "Add lead john@acme.com to campaign <id> then activate it", "persona": "campaign_outreach"}'
```

Or use the **Execute page** in the dashboard: select a persona, type a task in plain English — the gate visualizer shows all 4 gates evaluating in real time.

### Configuration

```bash
INSTANTLY_API_KEY=<your-instantly-v2-api-key>   # Bearer token from app.instantly.ai → Settings → API Keys
```

## Key Concepts

### Personas (Behavioral Contracts)
```yaml
# personas.yaml
- name: researcher
  allowed_tools: [knowledge_search, web_search]
  resource_scopes: ["kb:*", "web:*"]
  intent_patterns: ["search for information", "find data about"]
  max_ttl_seconds: 60        # Gate 3: persona expires after 60s
  risk_tolerance: low
```

### Seals (Notarized Records)
Every action produces an immutable seal containing: declared intent, gate results, tool output, Merkle fingerprint, chain-of-thought reasoning.

### Chains (Task Plans)
Complex tasks are decomposed into ordered steps. Each step is independently gated and sealed. The entire chain is verifiable via Merkle proof.

### Trust Tiers
Personas accumulate trust through consistent behavior:
- `cold_start` — new persona, strictest gates
- `established` — 50+ actions, relaxed drift threshold
- `trusted` — 500+ actions, <1% anomaly rate

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 15+ |
| ORM | SQLAlchemy 2.0+ (async) |
| Cache | Redis 7.4+ |
| Vector Store | ChromaDB |
| LLM | litellm (Anthropic, OpenAI, Ollama, 100+ providers) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384 dims) |
| Frontend | React 18 + Vite + Tailwind CSS |
| CLI | Typer + Rich |

## Project Structure

```
nexus/
├── core/           # Security plane: personas, anomaly, notary, ledger, engine (DAG executor)
├── knowledge/      # Cognitive plane: embeddings, vector store, context
├── reasoning/      # Decision gates: think/act, continue/complete, escalate
├── tools/          # Execution: registry, sandbox v2 (Py/JS/TS), executor, built-ins
│   └── builtin/    # web, file, comms, data, http_request, data_transform, frontend_design, rag_tools, instantly
├── workflows/      # DAG definition: dag.py, validator.py, manager.py (v2)
├── workers/        # ARQ background workers: queue.py (tasks), dispatcher.py (inline/bg routing)
├── credentials/    # Credential vault: encryption.py (Fernet), vault.py (v2)
├── mcp/            # MCP client + tool adapter + credential injection (v2)
├── triggers/       # Trigger system: webhook, cron, event bus, workflow-complete chaining (v2)
├── db/             # ORM models (v1 + v2), repository, Alembic migrations
├── llm/            # litellm integration + cost tracking + multi-provider routing
├── cache/          # Redis: fingerprint store, rate limiting, distributed locks
├── auth/           # JWT, middleware, rate limiter
├── api/            # FastAPI v1 routes + schemas
├── marketplace/    # Plugin Marketplace: SDK (PluginManifest, @nexus_plugin_tool), registry, validator, scaffolder
├── cli/            # Typer CLI commands + project templates
├── callbacks/      # NexusCallback protocol + LoggingCallback
├── config/         # NexusConfig (BaseSettings) + YAML loaders
frontend/           # React dashboard (Vite, port 5173) — 70+ source files; 246 Vitest tests; 6 Playwright E2E specs
examples/           # quickstart, custom_tool, local_llm, customer_support, code_review, mcp_integration
docs/               # quickstart.md, architecture.md, api-reference.md, tutorials/
sdk/python/         # Async HTTP client SDK (nexus_client.py)
tests/              # pytest suite (2090 tests — phases 0-32 + contract suite; + 5 slow Ollama tests)
```

## CLI

```bash
# Project setup
nexus init my-project     # Scaffold project dir with .env, personas.yaml, docker-compose.yml
nexus seed                # Seed database with 5 default personas

# Execution
nexus run "Analyze Q3"    # Execute task in-memory, print seal table + gate results + CoT

# Audit (no other framework has these)
nexus verify <chain-id>   # Verify Merkle chain integrity — detect any tampering
nexus replay <chain-id>   # Step-by-step chain replay with gate details
nexus inspect <seal-id>   # Full seal dump: intent, gates, CoT trace, fingerprints
nexus audit               # Paginated ledger export (table or JSON)
nexus gates [--stats]     # Show gate thresholds + per-gate pass/fail counts

# Introspection
nexus config              # Resolved config with env var names (secrets masked)
nexus tools               # Registered tools with risk levels

# Plugin Marketplace
nexus plugin install weather                      # Install from PyPI
nexus plugin install nexus-plugin-github==2.0.0  # Specific version
nexus plugin uninstall weather                    # Remove plugin + tools
nexus plugin list                                 # Installed plugins + tool counts
nexus plugin search slack                         # Search PyPI for nexus-plugin-* packages
nexus plugin upgrade weather                      # Upgrade to latest (or --version)
nexus plugin new my-tool                          # Scaffold a new plugin package
nexus plugin verify [plugin-name]                 # SHA-256 tamper detection (all or one)
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | /v1/auth/token | Exchange API key for JWT |
| POST | /v1/execute | Execute a task (returns chain + seals) |
| POST | /v1/execute/stream | Execute with SSE streaming (real-time gate events) |
| GET | /v1/ledger | Audit trail (paginated, tenant-scoped) |
| GET | /v1/ledger/{chain_id} | Chain detail (tenant-ownership enforced) |
| GET | /v1/personas | List personas |
| POST | /v1/personas | Create persona contract |
| PUT | /v1/personas/{id} | Update persona (tenant-ownership enforced) |
| GET | /v1/tools | List registered tools |
| POST | /v1/knowledge/ingest | Upload documents to vector store |
| GET | /v1/knowledge/query | Semantic search |
| GET | /v1/health | Health check with service probes |
| GET | /v2/workflows | List workflows |
| POST | /v2/workflows | Create workflow |
| GET | /v2/workflows/templates | List built-in workflow templates |
| POST | /v2/workflows/from-template/{template_id} | Create workflow from template |
| POST | /v2/workflows/generate | Generate workflow from natural language description |
| POST | /v2/workflows/import | Import workflow JSON |
| GET | /v2/workflows/{id} | Get workflow detail |
| PUT | /v2/workflows/{id} | Update workflow definition |
| PATCH | /v2/workflows/{id}/status | Update workflow status (active / paused / archived) |
| DELETE | /v2/workflows/{id} | Delete workflow |
| GET | /v2/workflows/{id}/export | Export workflow as JSON |
| POST | /v2/workflows/{id}/duplicate | Duplicate workflow |
| POST | /v2/workflows/{id}/activate | Activate a workflow |
| POST | /v2/workflows/{id}/pause | Pause an active workflow |
| GET | /v2/workflows/{id}/versions | Version history |
| POST | /v2/workflows/{id}/rollback/{version} | Roll back to a previous version |
| POST | /v2/workflows/{id}/refine | Refine workflow definition via LLM feedback |
| POST | /v2/workflows/{id}/explain | Plain-English explanation of a workflow |
| POST | /v2/workflows/{id}/run | Dispatch a manual workflow run (inline or background) |
| GET | /v2/executions | List executions |
| GET | /v2/executions/{id} | Execution detail with seal data |
| DELETE | /v2/executions/{id} | Delete a non-running execution |
| GET | /v2/executions/{id}/stream | SSE stream — replay seals then live events |
| GET | /v2/triggers | List triggers |
| POST | /v2/triggers | Create trigger (webhook / cron / event / manual) |
| GET | /v2/triggers/{id} | Get trigger |
| PUT | /v2/triggers/{id} | Update trigger (enable/disable) |
| POST | /v2/triggers/{id}/enable | Enable trigger |
| POST | /v2/triggers/{id}/disable | Disable trigger |
| DELETE | /v2/triggers/{id} | Delete trigger |
| GET/POST/… | /v2/webhooks/{path} | Catch-all webhook receiver (no JWT required) |
| GET | /v2/marketplace/search | Search plugin marketplace |
| GET | /v2/marketplace/installed | List installed plugins |
| POST | /v2/marketplace/install | Install a plugin from PyPI |
| DELETE | /v2/marketplace/{name} | Uninstall a plugin |
| GET | /v2/jobs/{job_id} | Poll background job status |
| GET | /v2/jobs/{job_id}/result | Get background job result |
| GET | /v2/credentials | List credentials (secrets never returned) |
| POST | /v2/credentials | Create credential |
| DELETE | /v2/credentials/{id} | Delete credential |
| GET | /v2/credentials/types | List supported credential types |
| POST | /v2/credentials/test | Test a credential (no :id — stateless probe) |
| POST | /v2/credentials/{id}/peek | Return last 4 chars of secret (read-only) |
| POST | /v2/credentials/{id}/rotate | Rotate credential secret |
| GET | /v2/skills | List skills |
| POST | /v2/skills | Create skill |
| GET | /v2/skills/{id} | Get skill |
| PATCH | /v2/skills/{id} | Update skill (auto-increments version) |
| DELETE | /v2/skills/{id} | Delete skill |
| POST | /v2/skills/{id}/duplicate | Duplicate skill |
| GET | /v2/skills/{id}/export | Export skill as JSON |
| POST | /v2/skills/import | Import skill JSON |
| GET | /v2/skills/{id}/diff | Diff two skill versions (`?from_version=1&to_version=2`) |
| GET | /v2/skills/{id}/invocations | Invocation history |
| GET | /v2/mcp/servers | List MCP servers |
| POST | /v2/mcp/servers | Register MCP server (stdio / sse / streamable_http) |
| DELETE | /v2/mcp/servers/{id} | Remove MCP server |
| GET | /v2/mcp/servers/{id}/tools | List tools exposed by a server |
| POST | /v2/mcp/servers/{id}/reconnect | Reconnect and refresh tool list |
| GET | /v2/events/stream | SSE event stream — requires `?token=<jwt>` |

Interactive API docs: `http://localhost:8000/docs`

## Security

NEXUS is designed for production agentic workloads. Security hardening applied across the full stack:

**Authentication & Authorization**
- JWT HS256 with minimum 32-byte key enforcement (RFC 7518 §3.2)
- API keys hashed with SHA-256 before storage — plaintext never persisted
- All queries scoped to `tenant_id` at the ORM level — another tenant's data is never returned; requests for another tenant's chain return 404 (existence not confirmed), collection endpoints return empty lists
- Demo fallback (`nxs_demo_key_12345`) disabled when `NEXUS_ENV=production`

**Prompt Injection Defense**
- LLM calls split into system (static instructions) + user (JSON-encoded data) messages
- KB context and session history passed as separate typed JSON fields (`kb_context`, `prior_results`) — never concatenated into the system prompt
- `SECURITY:` instruction in every LLM prompt marks external data as untrusted

**Input Validation**
- `task`: 10,000 char max; `content` (knowledge ingest): 500,000 char max
- `max_ttl_seconds`: 1–86400; `n_results` (vector query): 1–50
- ChromaDB namespace: `^[a-zA-Z0-9_-]{1,50}$` — path traversal blocked
- Redis keys namespaced and sanitized — persona IDs stripped of non-alphanumeric chars

**Infrastructure**
- Redis pinned to `7.4-alpine` (mitigates CVE-2025-49844 Lua RCE present in unversioned `7-alpine`); port not exposed externally
- LLM calls: 30s timeout for cloud providers; 120s for local Ollama models (accounts for model-swap latency); `num_ctx=8192` prevents silent prompt truncation — runaway generation blocked
- In-memory ledger capped at 10,000 seals — OOM via audit trail eliminated

**HTTP Hardening**
- `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block` on all responses
- CORS: explicit method list (`GET POST PUT DELETE`), not `*`
- Error responses return generic messages — no stack traces or internal details exposed
- JWT stored in `sessionStorage` (not `localStorage`) — cleared on tab close, not persisted to disk

**Plugin Supply Chain Security**
- Static pre-import scan rejects `os.system()`, `subprocess`, `eval()`, `exec()`, raw sockets, and HTTP calls at import time (CVE-2025-14009)
- SHA-256 checksum fingerprinting at install; `nexus plugin verify` detects post-install file tampering (Ultralytics incident pattern)
- Levenshtein edit-distance warning on package names ≤2 chars from any installed plugin (typosquatting defence, MUT-8694)
- `pip install --dry-run` before actual install flags dependency hijacking (CVE-2025-27607)
- 50,000-char cap on all string parameters passed to plugin tools (prompt injection guard)
- `setuptools>=78.1.1` enforced at build and runtime (CVE-2025-47273 path traversal + CVE-2024-6345 RCE)

**Rate Limiting**
- 60 requests/minute per tenant (API)
- 100 chains/hour per tenant

## Configuration

All config via environment variables with `NEXUS_` prefix. Copy `.env.example` to `.env`:

```bash
NEXUS_SECRET_KEY=<32+ byte random string>   # Required: openssl rand -hex 32
NEXUS_LLM_API_KEY=<your-api-key>            # ANTHROPIC_API_KEY or OPENAI_API_KEY
NEXUS_DATABASE_URL=postgresql+asyncpg://nexus:nexus@localhost:5432/nexus
NEXUS_REDIS_URL=redis://localhost:6379/0
NEXUS_DEFAULT_LLM_MODEL=anthropic/claude-sonnet-4-20250514
NEXUS_ENV=production                        # Disables demo key fallback
```

**Integrations:**

```bash
# Instantly.ai campaign tools
INSTANTLY_API_KEY=<your-instantly-v2-api-key>   # Bearer token — app.instantly.ai → Settings → API Keys

# RAG knowledge tools (optional — requires pip install -e ".[rag]")
NEXUS_RAG_ANYTHING_ENABLED=false                # Set true to enable multimodal RAG
NEXUS_RAG_ANYTHING_DIR=/tmp/nexus_rag           # Working directory for parsed docs + vector index
NEXUS_RAG_ANYTHING_PARSER=mineru                # Document parser: mineru | docling | paddleocr
```

**v2 additions — set before using workflow, credential, or MCP features (Phase 15+):**

```bash
# Credential vault
NEXUS_CREDENTIAL_ENCRYPTION_KEY=<fernet-key>    # Fernet key required in Phase 18+ (cryptography.fernet.Fernet.generate_key())

# Triggers
NEXUS_WEBHOOK_BASE_URL=https://yourdomain.com   # Public base URL registered with incoming webhook URLs
NEXUS_CRON_CHECK_INTERVAL=15                    # Seconds between cron schedule evaluations

# Background execution
NEXUS_TASK_QUEUE_URL=redis://localhost:6379/1   # Separate Redis DB from the fingerprint cache (/0)
NEXUS_WORKER_CONCURRENCY=4                      # Parallel background workflow workers

# Code sandbox
NEXUS_SANDBOX_MAX_MEMORY_MB=256
NEXUS_SANDBOX_MAX_EXECUTION_SECONDS=30

# MCP
NEXUS_MCP_CONNECTION_TIMEOUT=10                 # Seconds before MCP server connection attempt fails
NEXUS_MCP_TOOL_TIMEOUT=60                       # Per-tool execution timeout
```

## v2 — AI Automation Platform

NEXUS v2 transforms the single-shot agent framework into a **persistent, trigger-driven automation platform** — every workflow step still passes through all 4 anomaly gates and is sealed in the ledger.

### v2 DAG Execution Flow

```mermaid
flowchart LR
    TR([Trigger\ncron · webhook · manual]) --> WE["Workflow Engine\n(load + verify active)"]
    WE --> EP["Entry Steps\n(no incoming edges)"]

    subgraph PIPELINE["Per-Step Pipeline — all 4 gates applied to every DAG step"]
        CTX2[Build Context] --> TA2[Think/Act Gate]
        TA2 --> DECL[Declare Intent]
        DECL --> PA[Activate Persona]
        PA --> AE2{"Anomaly Engine\nGates 1–4"}
        AE2 -->|"any fail"| BLK([Block + Seal])
        AE2 -->|"all pass"| NT2[Notarize]
        NT2 --> EX2[Execute Tool]
        EX2 --> VL[Validate Output]
        VL --> LED2[(Ledger)]
    end

    EP --> PIPELINE
    LED2 --> NX["Resolve Next\n(edges + conditions)"]
    NX -->|"sequential"| EP
    NX -->|"parallel"| PAR["asyncio.gather\n(concurrent steps)"]
    PAR --> EP
    NX -->|"complete"| DONE([Workflow Complete])
```

**The key invariant:** no step in any workflow — regardless of complexity — can bypass the 4-gate security pipeline.

### Shipped Phases

| Phase | Feature | Status |
|-------|---------|--------|
| 15 | Foundation v2 — exception hierarchy, config fields, ORM models | ✅ Done |
| 16 | Workflow DAG Definition — step types, branching, loops, versioning | ✅ Done |
| 17 | DAG Execution Engine — parallel steps, branch, loop, sub-workflow, approval | ✅ Done |
| 18 | Credential Vault — Fernet-encrypted secrets, OAuth2, runtime injection | ✅ Done |
| 19 | MCP Integration — Model Context Protocol client + tool adapter + credential injection | ✅ Done |
| 20 | Universal HTTP Tool + Data Transform — REST caller (auth/pagination/retry/JMESPath) + 15-op pipeline | ✅ Done |
| 21 | Code Sandbox v2 — Python/JS/TypeScript subprocess isolation, tsx cache, process-group kill | ✅ Done |
| 22 | Trigger System — webhooks, cron scheduler, event bus, workflow-complete chaining | ✅ Done |
| 23 | NL Workflow Generation — natural language → DAG via LLM, iterative refinement, explain | ✅ Done |
| 23.1 | Ambiguity Resolution — multi-round clarification sessions, specificity scoring, plan sealing | ✅ Done |
| 24 | Visual Canvas — React Flow drag-and-drop workflow editor | ✅ Done |
| 25 | Frontend v2 + Skills System — Skills CRUD + versioning, Credentials vault UI (/peek, no secrets), MCP Servers, Executions + ChainReplay, GateBar/GateChip, React Query, Zustand, SSE event stream | ✅ Done |
| 26 | Background Execution — ARQ task queue, `WorkflowDispatcher` (inline/background routing, 5-step threshold), `POST /v2/workflows/{id}/run`, `GET /v2/jobs/{job_id}`, `TriggerManager.set_dispatcher()` | ✅ Done |
| 27 | Plugin Marketplace — `nexus-plugin-*` PyPI ecosystem, `PluginManifest` + `@nexus_plugin_tool` SDK, `PluginRegistry` (install/upgrade/verify), 15-threat CVE security model, `nexus plugin` CLI (7 sub-commands) | ✅ Done |
| 28 | Persistence v2 — Alembic migrations (async), 15 new repository methods (workflow/credential/MCP CRUD), seed v2 data, `CredentialModel.is_active`, `MCPServerModel.url` nullable | ✅ Done |
| 29 | API v2 Complete — triggers CRUD, webhook catch-all (no JWT), marketplace endpoints, workflow lifecycle (activate/pause/versions/rollback/refine/explain), execution SSE stream + delete, job result, MCP refresh | ✅ Done |
| 30 | Test suite v2 — 113 fast tests across 8 new files (workflows, dag_engine, credentials, mcp, triggers, http_tool, code_sandbox, api_v2) + 5 Ollama live-LLM tests; bugs fixed in `generator.py` (._validator), Ollama client (num_ctx=8192, 120s timeout) | ✅ Done |
| 31 | Infrastructure v2 — nexus-scheduler singleton (Redis distributed lock + heartbeat), nginx reverse proxy (prod profile), chroma service, API healthcheck gating worker/scheduler startup, `frontend/nginx.conf` v2 proxy, dev hot-reload overrides, Makefile targets (`infra`/`worker`/`scheduler`/`prod`/`scale-workers`/`test-fast`/`logs-*`/`fmt`), `.env.example` v2 vars | ✅ Done |
| 32 | Examples & Docs v2 — 6 real-world workflow examples (lead qualification, lead nurturing, AI email outreach, content repurposing, personal finance tracker, stock analysis); `examples/_shared/` utilities (`nexus_client`, `demo_data`); `nexus credential check` CLI command | ✅ Done |
| 33 | RAG + Campaign Automation — `rag_ingest`/`rag_query` multimodal knowledge tools; 9 Instantly.ai campaign tools (`instantly_audit`, `instantly_create_campaign`, `instantly_activate_campaign` + 6 more); 3 campaign personas (`campaign_researcher`, `campaign_outreach`, `campaign_optimizer`); selector fixes (LLM trust pass-through, wildcard resource targets) | ✅ Done |

## Testing

NEXUS uses a 3-tier test pyramid: backend HTTP contracts, frontend component tests, and full E2E browser flows.

### Run tests

```bash
# Tier 1 — Backend: unit, integration, and HTTP contract tests
pytest tests/ -m "not slow"                          # 2097 tests (~60s)
pytest tests/test_frontend_contracts.py -v           # 60 contract tests (C1–C60)

# Tier 2 — Frontend: component tests with MSW (no real server needed)
cd frontend && npm run test:run                       # 246 Vitest tests
npm run test:coverage                                # with v8 coverage report

# Tier 3 — E2E: Playwright full-browser tests (requires both servers running)
make run &                                           # API on :8000
cd frontend && npm run dev &                         # Frontend on :5173
npx playwright install && npx playwright test
```

### Test pyramid

| Tier | Tool | Count | Scope |
|------|------|-------|-------|
| 1 — HTTP contracts | pytest + ASGI transport | 60 | Response shapes, auth, tenant isolation, error bodies |
| 2 — Component tests | Vitest + MSW | 246 | Every UI page: auth, workflows, executions, credentials, skills, MCP servers, dashboard, SSE |
| 3 — E2E | Playwright (Chromium) | 6 specs | Login, full workflow lifecycle, credentials CRUD, skills CRUD, MCP servers, error resilience |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and PR process.

## License

Apache 2.0 — see [LICENSE](LICENSE).
