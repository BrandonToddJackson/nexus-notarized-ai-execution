# NEXUS API Reference

Base URL: `http://localhost:8000`
All routes under `/v1/`.
Authentication: `Authorization: <api_key>` or Bearer JWT.

---

## Authentication

### POST /v1/auth/token

Exchange API key for JWT.

**Request:**
```json
{ "api_key": "nxs_demo_key_12345" }
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

## Execute

### POST /v1/execute

Execute a task synchronously. Returns when the chain completes.

**Request:**
```json
{
  "task": "What is NEXUS?",
  "tenant_id": "demo",
  "persona": "researcher"
}
```

**Response:**
```json
{
  "chain_id": "uuid",
  "status": "completed",
  "task": "What is NEXUS?",
  "seals": [
    {
      "id": "uuid",
      "tool_name": "knowledge_search",
      "status": "executed",
      "anomaly_result": {
        "overall_verdict": "pass",
        "gates": [
          { "gate_name": "scope",  "verdict": "pass", "score": 1.0, "threshold": 1.0, "details": "..." },
          { "gate_name": "intent", "verdict": "pass", "score": 0.82, "threshold": 0.75, "details": "..." },
          { "gate_name": "ttl",    "verdict": "pass", "score": 0.0, "threshold": 120, "details": "..." },
          { "gate_name": "drift",  "verdict": "skip", "score": 0.0, "threshold": 2.5, "details": "..." }
        ]
      },
      "cot_trace": ["Step 1: ...", "Step 2: ..."],
      "fingerprint": "abc123...",
      "parent_fingerprint": ""
    }
  ],
  "cost_usd": 0.0012
}
```

### POST /v1/execute/stream

Execute a task with real-time SSE events.

**Request:** same as `/v1/execute`

**Response:** `text/event-stream`

Event types:
```
event: chain_started
data: {"chain_id": "...", "task": "..."}

event: step_started
data: {"step_index": 0, "tool_name": "knowledge_search"}

event: gate_result
data: {"gate_name": "scope", "verdict": "pass", "score": 1.0, "details": "..."}

event: seal_created
data: {"seal_id": "...", "tool_name": "...", "status": "executed"}

event: step_completed
data: {"step_index": 0, "status": "executed"}

event: chain_completed
data: {"chain_id": "...", "status": "completed", "seal_count": 1}
```

---

## Ledger

### GET /v1/ledger

Query the immutable seal ledger.

**Query params:**
- `tenant_id` (required)
- `chain_id` (optional)
- `persona_id` (optional)
- `limit` (default 50, max 500)
- `offset` (default 0)

**Response:** Array of seal objects (same shape as in `/v1/execute`)

---

## Personas

### GET /v1/personas

List available personas.

**Query params:** `tenant_id`

**Response:**
```json
[
  {
    "id": "uuid",
    "name": "researcher",
    "allowed_tools": ["knowledge_search", "web_search"],
    "resource_scopes": ["kb:*", "web:*"],
    "trust_tier": "cold_start"
  }
]
```

---

## Knowledge

### POST /v1/knowledge

Upload a document to the vector store.

**Request:**
```json
{
  "content": "Full document text...",
  "namespace": "product_docs",
  "source": "onboarding.md",
  "tenant_id": "demo",
  "access_level": "internal"
}
```

### GET /v1/knowledge/search

Semantic search.

**Query params:** `q` (query), `tenant_id`, `namespace`, `limit`

---

## Tools

### GET /v1/tools

List registered tools with metadata.

---

## Health

### GET /v1/health

**Response:**
```json
{
  "status": "healthy",
  "db": "connected",
  "redis": "connected",
  "version": "0.1.0"
}
```
