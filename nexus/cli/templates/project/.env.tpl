# {{project_name}} — NEXUS environment configuration
# Copy to .env and fill in your values.

# ── LLM provider (pick one) ───────────────────────────────────────────────────
#
# Option A — Cloud (Anthropic)
# NEXUS_DEFAULT_LLM_MODEL=anthropic/claude-sonnet-4-20250514
# ANTHROPIC_API_KEY=sk-ant-...
#
# Option B — Cloud (OpenAI)
# NEXUS_DEFAULT_LLM_MODEL=openai/gpt-4o
# OPENAI_API_KEY=sk-...
#
# Option C — Local (Ollama, no API key required)
#   Install: https://ollama.com → `ollama pull qwen2.5-coder:7b`
# NEXUS_DEFAULT_LLM_MODEL=ollama/qwen2.5-coder:7b
# NEXUS_OLLAMA_BASE_URL=http://localhost:11434
#
ANTHROPIC_API_KEY=sk-ant-...

# ── Infrastructure (defaults work for local Docker) ───────────────────────────
NEXUS_DATABASE_URL=postgresql+asyncpg://nexus:nexus@localhost:5432/nexus
NEXUS_REDIS_URL=redis://localhost:6379/0

# ── Security ──────────────────────────────────────────────────────────────────
NEXUS_SECRET_KEY={{secret_key}}
NEXUS_API_KEY_PREFIX=nxs_

# ── Gate thresholds (production defaults) ────────────────────────────────────
# Raise NEXUS_GATE_INTENT_THRESHOLD to 0.75 when using an LLM with API key.
# Without LLM, rule-based planned_actions score ~0.33–0.45, so use 0.3 locally.
NEXUS_GATE_INTENT_THRESHOLD=0.75
NEXUS_GATE_DRIFT_SIGMA=2.5

# ── Limits ────────────────────────────────────────────────────────────────────
NEXUS_RATE_LIMIT_PER_MINUTE=60
NEXUS_MAX_CHAINS_PER_HOUR=100
NEXUS_TASK_BUDGET_USD=10.0
