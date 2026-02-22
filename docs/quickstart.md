# NEXUS Quickstart

Get your first sealed AI action in under 5 minutes. No cloud API key required for the demo.

## Option A: In-memory demo (0 dependencies)

```bash
git clone https://github.com/nexus-ai/nexus
cd nexus
pip install -e ".[dev]"
python examples/quickstart/main.py
```

You'll see:
- Gate 1 (Scope): tool vs persona resource scope check
- Gate 2 (Intent): semantic similarity between declared intent and persona patterns
- Gate 3 (TTL): persona lifetime check
- Gate 4 (Drift): behavioral baseline check (skipped — no Redis in-memory)
- A notarized **Seal** with Merkle fingerprint
- Chain-of-thought trace inside the seal

## Option B: Full stack (Docker)

```bash
git clone https://github.com/nexus-ai/nexus
cd nexus
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY or OPENAI_API_KEY
docker compose up --build
```

Then:
```bash
curl -X POST http://localhost:8000/v1/execute \
  -H "Content-Type: application/json" \
  -H "Authorization: nxs_demo_key_12345" \
  -d '{"task": "What is NEXUS?"}'
```

Frontend dashboard: http://localhost:5173

## Option C: CLI

```bash
pip install -e ".[dev]"
docker compose up postgres redis -d
nexus seed
nexus run "What is NEXUS?"
```

## What you get

Every task produces a **Chain** containing one or more **Seals**. Each seal records:

| Field | Description |
|-------|-------------|
| `tool_name` | Which tool executed |
| `status` | `executed` or `blocked` |
| `anomaly_result` | 4 gate results with scores |
| `cot_trace` | Reasoning steps |
| `fingerprint` | SHA-256 Merkle hash |
| `parent_fingerprint` | Links to previous seal |

## Next steps

- [Architecture](architecture.md) — how the 4-gate pipeline works
- [Philosophy](philosophy.md) — accountability vs. determinism
- [API Reference](api-reference.md) — full REST API docs
- [Tutorials: First Agent](tutorials/first-agent.md) — build a custom agent
