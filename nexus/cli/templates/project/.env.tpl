# {{project_name}} — NEXUS environment configuration
# Copy to .env and fill in your values.

# LLM provider (pick one)
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...

# Infrastructure (defaults work for local Docker)
NEXUS_DATABASE_URL=postgresql+asyncpg://nexus:nexus@localhost:5432/nexus
NEXUS_REDIS_URL=redis://localhost:6379/0

# Security
NEXUS_SECRET_KEY={{secret_key}}
NEXUS_API_KEY_PREFIX=nxs_

# Gate thresholds (production defaults)
NEXUS_GATE_INTENT_THRESHOLD=0.75
NEXUS_GATE_DRIFT_SIGMA=2.5

# Limits
NEXUS_RATE_LIMIT_PER_MINUTE=60
NEXUS_MAX_CHAINS_PER_HOUR=100
NEXUS_TASK_BUDGET_USD=10.0
