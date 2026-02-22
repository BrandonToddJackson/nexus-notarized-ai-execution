# NEXUS vs. Other Frameworks

> **The honest comparison.** We address "just a wrapper" in paragraph one.

NEXUS is not a wrapper around LangChain, CrewAI, or any other framework. It is built from scratch around one idea: **every AI action must be notarized before execution**. No other framework ships this.

---

## Feature Comparison

| Feature | NEXUS | LangChain | LangGraph | CrewAI | AutoGen |
|---------|-------|-----------|-----------|--------|---------|
| Immutable audit ledger | ✅ | ❌ | ❌ | ❌ | ❌ |
| 4-gate anomaly engine | ✅ | ❌ | ❌ | ❌ | ❌ |
| Merkle-chain seals | ✅ | ❌ | ❌ | ❌ | ❌ |
| Intent vs. target verification | ✅ | ❌ | ❌ | ❌ | ❌ |
| Persona TTL (ephemeral identity) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Output PII scanning | ✅ | ❌ | ❌ | ❌ | ❌ |
| CoT in every seal | ✅ | ❌ | partial | ❌ | ❌ |
| Trust tiers (cold_start → trusted) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Multi-tenant isolation | ✅ | ❌ | ❌ | ❌ | ❌ |
| Live gate visualizer | ✅ | ❌ | ❌ | ❌ | ❌ |
| Chain replay | ✅ | ❌ | ❌ | ❌ | ❌ |
| Semantic intent matching | ✅ | partial | partial | ❌ | ❌ |
| Behavioral drift detection | ✅ | ❌ | ❌ | ❌ | ❌ |
| RAG (built-in) | ✅ | ✅ | ✅ | ✅ | partial |
| Multi-agent | ❌ (v1) | ✅ | ✅ | ✅ | ✅ |
| Tool ecosystem | small | large | large | medium | medium |

---

## "Isn't NEXUS just a wrapper?"

No. LangChain wraps LLM providers and tool libraries. NEXUS is a **security and accountability layer** that enforces constraints on what an AI agent is allowed to do — before it does it.

LangChain gives you tools. NEXUS gives you a guarantee that every tool call was authorized, scoped, inspected, and recorded.

The two can coexist: use LangChain tools inside a NEXUS persona.

---

## NEXUS vs. LangGraph

LangGraph handles workflow orchestration (DAGs, state machines). NEXUS handles behavioral governance (who can do what, and proof that they only did that).

LangGraph answers: "In what order did the agent execute steps?"
NEXUS answers: "Was the agent *allowed* to execute each step, and do we have cryptographic proof?"

---

## NEXUS vs. CrewAI

CrewAI orchestrates multiple cooperating agents. NEXUS is intentionally **single-agent** with multiple behavioral personas. This is an architectural choice, not a limitation:

- CrewAI: 3 agents (researcher + writer + reviewer) run as separate entities
- NEXUS: 1 agent switches between researcher/creator/reviewer personas, each scoped and TTL'd

The benefit: a single audit trail, a single trust model, no inter-agent trust escalation problem.

---

## When to choose NEXUS

- You need to **audit and prove** what your AI agent did (compliance, SOC 2, legal)
- You're operating in a regulated environment (finance, healthcare, legal)
- You want to **block out-of-scope actions** before they execute
- You need a **live dashboard** showing gate results and sealed actions in real time
- You're building on top of another framework but need an accountability layer

## When NOT to choose NEXUS (yet)

- You need a large ecosystem of pre-built tools (use LangChain + NEXUS together)
- You need true multi-agent coordination (v2 roadmap)
- You need OpenTelemetry/OTEL tracing (v2 roadmap)
