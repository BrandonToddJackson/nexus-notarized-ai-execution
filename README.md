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
│              │    │ patterns?    │    │              │    │ with history? │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │                   │
    PASS/FAIL           PASS/FAIL           PASS/FAIL           PASS/FAIL
```

If **any** gate fails → action is **BLOCKED** and sealed as blocked in the ledger.

## Quickstart

```bash
# Clone and install
git clone https://github.com/your-org/nexus.git
cd nexus
pip install -e ".[dev]"

# Start infrastructure
docker compose up postgres redis -d

# Seed with demo data
make seed

# Run the API
make run

# In another terminal — execute a task
curl -X POST http://localhost:8000/v1/execute \
  -H "Authorization: nxs_demo_key_12345" \
  -H "Content-Type: application/json" \
  -d '{"task": "What is NEXUS?"}'
```

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
  max_ttl_seconds: 60
  risk_tolerance: low
```

### Seals (Notarized Records)
Every action produces an immutable seal containing: declared intent, gate results, tool output, Merkle fingerprint, chain-of-thought reasoning.

### Chains (Task Plans)
Complex tasks are decomposed into ordered steps. Each step is independently gated and sealed.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 15+ |
| ORM | SQLAlchemy 2.0+ (async) |
| Cache | Redis 7+ |
| Vector Store | ChromaDB |
| LLM | litellm (100+ providers) |
| Embeddings | sentence-transformers |
| Frontend | React + Vite + Tailwind |
| CLI | Typer + Rich |

## Project Structure

```
nexus/
├── core/           # Security plane: personas, anomaly, notary, ledger, engine
├── knowledge/      # Cognitive plane: embeddings, vector store, context
├── reasoning/      # Decision gates: think/act, continue/complete, escalate
├── tools/          # Execution: registry, sandbox, executor, built-ins
├── skills/         # Named tool bundles
├── db/             # Persistence: models, repository, migrations
├── llm/            # LLM integration via litellm
├── cache/          # Redis: fingerprints, locks
├── auth/           # JWT, middleware, rate limiting
├── api/            # FastAPI routes and schemas
├── cli/            # Typer CLI commands
frontend/           # React dashboard
tests/              # pytest test suite
examples/           # Working examples
```

## CLI

```bash
nexus init my-project     # Scaffold new project
nexus dev                 # Start dev server with hot reload
nexus run "Analyze Q3"    # Execute task, print seal summary
nexus seed                # Seed database with defaults
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | /v1/execute | Execute a task (returns chain + seals) |
| POST | /v1/execute/stream | Execute with SSE streaming |
| GET | /v1/ledger | Audit trail (paginated) |
| GET | /v1/ledger/{chain_id} | Chain detail |
| GET/POST | /v1/personas | List/create personas |
| GET | /v1/tools | List registered tools |
| POST | /v1/knowledge/ingest | Upload documents |
| GET | /v1/health | Health check |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and PR process.

## License

Apache 2.0 — see [LICENSE](LICENSE).
