"""nexus init — Scaffold a new NEXUS project."""

import os
import typer
from rich.console import Console

console = Console()

_ENV_TEMPLATE = """\
# NEXUS Configuration
# Copy to .env and fill in your values

# ── App ──
NEXUS_DEBUG=true
NEXUS_SECRET_KEY=change-me-in-production
NEXUS_LOG_LEVEL=INFO

# ── Database ──
NEXUS_DATABASE_URL=postgresql+asyncpg://nexus:nexus@localhost:5432/nexus

# ── Redis ──
NEXUS_REDIS_URL=redis://localhost:6379/0

# ── LLM ──
# Set ONE of these based on your provider:
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=
# For local LLMs (Ollama):
# NEXUS_DEFAULT_LLM_MODEL=ollama/llama3
NEXUS_DEFAULT_LLM_MODEL=anthropic/claude-sonnet-4-20250514

# ── Budget ──
NEXUS_DEFAULT_BUDGET_USD=50.0

# ── CORS ──
NEXUS_CORS_ORIGINS=["http://localhost:5173"]
"""

_PERSONAS_YAML = """\
# NEXUS Default Personas
# Each persona is a behavioral contract that controls what the agent can do.

personas:
  - name: researcher
    description: Searches and retrieves information from knowledge bases and the web
    allowed_tools:
      - knowledge_search
      - web_search
      - web_fetch
      - file_read
    resource_scopes:
      - "kb:*"
      - "web:*"
      - "file:read:*"
    intent_patterns:
      - search for information
      - find data about
      - look up
      - research
    risk_tolerance: low
    max_ttl_seconds: 60

  - name: analyst
    description: Analyzes data and computes statistics
    allowed_tools:
      - knowledge_search
      - compute_stats
      - file_read
      - file_write
    resource_scopes:
      - "kb:*"
      - "file:*"
      - "data:*"
    intent_patterns:
      - analyze data
      - compute statistics
      - calculate
      - summarize findings
    risk_tolerance: medium
    max_ttl_seconds: 120

  - name: creator
    description: Creates content: documents, reports, summaries
    allowed_tools:
      - knowledge_search
      - file_write
    resource_scopes:
      - "kb:*"
      - "file:write:*"
    intent_patterns:
      - write
      - create
      - draft
      - generate content
      - compose
    risk_tolerance: low
    max_ttl_seconds: 90

  - name: communicator
    description: Sends emails and messages
    allowed_tools:
      - knowledge_search
      - send_email
      - file_read
    resource_scopes:
      - "kb:*"
      - "email:*"
      - "file:read:*"
    intent_patterns:
      - send email
      - notify
      - communicate
      - message
    risk_tolerance: high
    max_ttl_seconds: 60

  - name: operator
    description: Executes code and system operations
    allowed_tools:
      - knowledge_search
      - file_read
      - file_write
      - compute_stats
    resource_scopes:
      - "kb:*"
      - "file:*"
      - "system:*"
    intent_patterns:
      - execute
      - run
      - deploy
      - configure
      - operate
    risk_tolerance: high
    max_ttl_seconds: 180
"""

_TOOLS_YAML = """\
# NEXUS Built-in Tools
# These tools are registered automatically when NEXUS starts.

tools:
  - name: knowledge_search
    description: Search the local knowledge base
    risk_level: low

  - name: web_search
    description: Search the web for information
    risk_level: low

  - name: web_fetch
    description: Fetch content from a URL
    risk_level: low

  - name: file_read
    description: Read a file from the filesystem
    risk_level: low

  - name: file_write
    description: Write content to a file
    risk_level: medium

  - name: send_email
    description: Send an email message
    risk_level: high

  - name: compute_stats
    description: Compute statistics on data
    risk_level: low
"""

_MAIN_PY = """\
\"\"\"NEXUS entry point for this project.\"\"\"

from nexus.api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

_DOCKER_COMPOSE = """\
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - chromadata:/app/data/chroma

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    depends_on:
      - api

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: nexus
      POSTGRES_PASSWORD: nexus
      POSTGRES_DB: nexus
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U nexus"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
  chromadata:
"""


def init_project(
    name: str = typer.Argument("my-nexus-project", help="Project directory name"),
):
    """Scaffold a new NEXUS project directory.

    Creates:
    - .env with placeholder API keys
    - personas.yaml with 5 default personas
    - tools.yaml with built-in tool config
    - main.py entry point
    - knowledge/ directory for knowledge docs
    - docker-compose.yml for local development
    """
    if os.path.exists(name):
        console.print(f"[red]Error:[/red] Directory '{name}' already exists.")
        raise typer.Exit(1)

    console.print(f"[green]Creating NEXUS project:[/green] {name}")

    os.makedirs(name)
    os.makedirs(os.path.join(name, "knowledge"))

    files = {
        ".env": _ENV_TEMPLATE,
        "personas.yaml": _PERSONAS_YAML,
        "tools.yaml": _TOOLS_YAML,
        "main.py": _MAIN_PY,
        "docker-compose.yml": _DOCKER_COMPOSE,
    }

    for filename, content in files.items():
        path = os.path.join(name, filename)
        with open(path, "w") as f:
            f.write(content)
        console.print(f"  [dim]created[/dim]  {filename}")

    console.print("  [dim]created[/dim]  knowledge/")
    console.print()
    console.print(f"[bold green]Project '{name}' ready![/bold green]")
    console.print()
    console.print("Next steps:")
    console.print(f"  cd {name}")
    console.print("  # Edit .env and set ANTHROPIC_API_KEY or OPENAI_API_KEY")
    console.print("  docker compose up postgres redis -d")
    console.print("  nexus seed")
    console.print("  nexus run \"What is NEXUS?\"")
