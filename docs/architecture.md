# NEXUS Architecture

## System overview

```
User Task
    │
    ▼
ChainManager.decompose()          ← LLM or single-step fallback
    │
    ├─ for each step:
    │       │
    │       ▼
    │   ContextBuilder.build()    ← RAG retrieval + persona priming
    │       │
    │       ▼
    │   ThinkActGate.decide()     ← Confidence ≥ threshold? ACT : THINK (loop)
    │       │
    │       ▼
    │   ToolSelector.select()     ← Rule-based or LLM tool selection
    │       │
    │       ▼
    │   IntentVerifier.declare()  ← Compose IntentDeclaration
    │       │
    │       ▼
    │   PersonaManager.activate() ← Load behavioral contract + start TTL
    │       │
    │       ▼
    │   AnomalyEngine.check()     ← 4 gates (scope, intent, TTL, drift)
    │       │
    │       ├─ FAIL → Notary.seal(BLOCKED) → Ledger.append() ──────────────┐
    │       │                                                                │
    │       ▼ PASS                                                           │
    │   Notary.create_seal()      ← Initialize seal with intent + gates     │
    │       │                                                                │
    │       ▼                                                                │
    │   ToolExecutor.run()        ← Execute in Sandbox (asyncio timeout)    │
    │       │                                                                │
    │       ▼                                                                │
    │   OutputValidator.validate()← PII scan + result/intent match          │
    │       │                                                                │
    │       ▼                                                                │
    │   Notary.finalize_seal()    ← Merkle fingerprint                      │
    │       │                                                                │
    │       ▼                                                                │
    │   Ledger.append()           ← Immutable record ◄─────────────────────┘
    │       │
    │       ▼
    │   CostTracker.record()      ← Token usage + USD cost
    │       │
    │       ▼
    │   PersonaManager.revoke()   ← Ephemeral — destroyed after each step
    │       │
    │       ▼
    │   ContinueCompleteGate()    ← Done? COMPLETE : CONTINUE/RETRY/ESCALATE
    │
    ▼
ChainPlan (completed/failed/escalated)
```

## Component map

| Component | File | Responsibility |
|-----------|------|----------------|
| NexusEngine | `nexus/core/engine.py` | Orchestrator — wires all components |
| AnomalyEngine | `nexus/core/anomaly.py` | 4 gates: scope, intent, TTL, drift |
| Notary | `nexus/core/notary.py` | Create + finalize + verify seals |
| Ledger | `nexus/core/ledger.py` | Append-only seal storage |
| ChainManager | `nexus/core/chain.py` | Task decomposition + step tracking |
| IntentVerifier | `nexus/core/verifier.py` | Cross-check declared vs. actual intent |
| OutputValidator | `nexus/core/output_validator.py` | PII scan + result validation |
| CoTLogger | `nexus/core/cot_logger.py` | Capture reasoning traces |
| PersonaManager | `nexus/core/personas.py` | Activate / revoke behavioral contracts |
| EmbeddingService | `nexus/knowledge/embeddings.py` | sentence-transformers (all-MiniLM-L6-v2) |
| KnowledgeStore | `nexus/knowledge/store.py` | ChromaDB vector store |
| ContextBuilder | `nexus/knowledge/context.py` | RAG + session memory + persona priming |
| ToolRegistry | `nexus/tools/registry.py` | Register + lookup tool implementations |
| ToolSelector | `nexus/tools/selector.py` | Select tool from task description |
| ToolExecutor | `nexus/tools/executor.py` | Execute with timeout + sandbox |
| FingerprintCache | `nexus/cache/fingerprints.py` | Redis behavioral baseline for Gate 4 |
| CostTracker | `nexus/llm/cost_tracker.py` | Token + USD tracking |

## The 4 Anomaly Gates

### Gate 1 — Scope
```python
tool_name in persona.allowed_tools
AND
fnmatch(resource_target, persona.resource_scope)
```
Binary: pass or fail. Most common failure mode.

### Gate 2 — Intent
```python
max(cosine_similarity(embed(declared_intent), embed(pattern))
    for pattern in persona.intent_patterns) >= threshold
```
Semantic: embedding similarity. Threshold default 0.75. Catches prompt injection.

### Gate 3 — TTL
```python
(now - activation_time) <= persona.max_ttl_seconds
```
Time-based. Prevents long-running persona accumulation.

### Gate 4 — Drift
```python
freq = action_fingerprint_count / total_fingerprints
z_score = (freq - mean_freq) / std_freq
abs(z_score) <= sigma  # default 2.5
```
Statistical: requires 10+ historical samples (SKIP on cold start).

## Data flow: Seal lifecycle

```
PENDING → DECLARED → EXECUTING → EXECUTED
                    ↓
                 BLOCKED  (any gate fails)
                    ↓
                 FAILED   (tool error / timeout)
```

## Infrastructure

```
┌─────────────────────────────────────────────┐
│  NEXUS API (FastAPI + uvicorn, port 8000)   │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │  NexusEngine    │  │  SSE /stream     │  │
│  │  (in app.state) │  │  (gate events)   │  │
│  └─────────────────┘  └──────────────────┘  │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┼──────────────┐
    ▼             ▼              ▼
PostgreSQL 15  Redis 7.4    ChromaDB
(seals, chains, (fingerprints,  (vector store,
 personas,      rate limits,    ./data/chroma)
 tenants)       locks)
```
