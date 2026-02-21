# NEXUS — Notarized AI Execution

> **The agent framework where AI actions are accountable.**

Every AI action is declared, verified, and sealed in an immutable ledger before execution. If it looks wrong, it's blocked.

```
User Task → Decompose → [Declare Intent → Activate Persona → 4 Anomaly Gates → Seal → Execute → Validate] → Audit Trail
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
| RAG + Knowledge Scoping | ✅ | ✅ | ✅ | ✅ |
| Multi-provider LLM | ✅ | ✅ | ✅ | ✅ |
| Tool Framework | ✅ | ✅ | ✅ | ✅ |

## The 4 Anomaly Gates

Every action passes through 4 gates before execution:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   GATE 1     │    │   GATE 2     │    │   GATE 3     │    │   GATE 4     │
│   SCOPE      │───→│   INTENT     │───→│   TTL        │───→│   DRIFT      │
│              │    │              │    │              │    │              │
│ Is this tool │    │ Does intent  │    │ Has persona  │    │ Is this      │
│ allowed for  │    │ match the    │    │ been active  │    │ action       │
│ this persona?│    │ persona's    │    │ too long?    │    │ consistent   │
│              │    │ patterns?    │    │              │    │ with history?│
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │                   │
    PASS/FAIL           PASS/FAIL           PASS/FAIL           PASS/FAIL
```

If **any** gate fails → action is **BLOCKED** and sealed as blocked in the ledger.

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

```
┌─────────────────────────────────────────────────────────────────┐
│                      NEXUS ENGINE                                │
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │   COGNITIVE PLANE     │    │     SECURITY PLANE            │   │
│  │                       │    │                               │   │
│  │  Knowledge Store      │    │  Persona Manager              │   │
│  │  Embedding Service    │    │  Anomaly Engine (4 gates)     │   │
│  │  Context Builder      │    │  Notary (Merkle seals)        │   │
│  │  Think/Act Gate       │    │  Ledger (immutable audit)     │   │
│  │  Continue/Complete    │    │  Intent Verifier              │   │
│  │  Escalate Gate        │    │  Output Validator             │   │
│  │                       │    │  CoT Logger                   │   │
│  └───────────┬───────────┘    └───────────────┬───────────────┘   │
│              │                                │                   │
│              └────────────┬───────────────────┘                   │
│                           │                                       │
│                    ┌──────┴──────┐                                │
│                    │  EXECUTION   │                                │
│                    │  Tool Reg    │                                │
│                    │  Selector    │                                │
│                    │  Sandbox     │                                │
│                    │  Executor    │                                │
│                    └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

**Not multi-agent.** One agent, multiple personas. A persona is a constrained operating mode — not a separate entity.

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
├── core/           # Security plane: personas, anomaly, notary, ledger, engine
├── knowledge/      # Cognitive plane: embeddings, vector store, context
├── reasoning/      # Decision gates: think/act, continue/complete, escalate
├── tools/          # Execution: registry, sandbox, executor, built-ins
├── db/             # Persistence: models, repository, migrations
├── llm/            # LLM integration via litellm + cost tracking
├── cache/          # Redis: fingerprint store, rate limiting
├── auth/           # JWT, middleware, rate limiter
├── api/            # FastAPI routes and schemas
├── cli/            # Typer CLI commands
frontend/           # React dashboard (Vite, port 5173)
tests/              # pytest suite (267 tests)
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

Interactive API docs: `http://localhost:8000/docs`

## Security

NEXUS is designed for production agentic workloads. Security hardening applied across the full stack:

**Authentication & Authorization**
- JWT HS256 with minimum 32-byte key enforcement (RFC 7518 §3.2)
- API keys hashed with SHA-256 before storage — plaintext never persisted
- All routes tenant-scoped; cross-tenant access returns 403, not 404
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
- Redis pinned to `7.4-alpine` (CVE-2025-49844 / "RediShell" CVSS 10.0 mitigated); port not exposed externally
- LLM calls wrapped in 30-second timeout — runaway generation blocked
- In-memory ledger capped at 10,000 seals — OOM via audit trail eliminated

**HTTP Hardening**
- `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block` on all responses
- CORS: explicit method list (`GET POST PUT DELETE`), not `*`
- Error responses return generic messages — no stack traces or internal details exposed
- JWT stored in `sessionStorage` (not `localStorage`) — cleared on tab close, not persisted to disk

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and PR process.

## License

Apache 2.0 — see [LICENSE](LICENSE).
