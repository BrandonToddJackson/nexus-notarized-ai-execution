.PHONY: dev test lint seed migrate clean install \
        infra api-local worker-local scheduler-local \
        worker scheduler prod scale-workers \
        test-fast logs-api logs-worker logs-scheduler fmt run

install:
	pip install -e ".[dev]"

dev:
	docker compose up --build

# ── Infrastructure ────────────────────────────────────────────────────────────

infra:
	docker compose up postgres redis chroma -d

# ── Local development (no Docker) ────────────────────────────────────────────

run:
	uvicorn nexus.api.main:app --reload --port 8000

api-local:
	uvicorn nexus.api.main:app --reload --reload-dir nexus --host 0.0.0.0 --port 8000

worker-local:
	arq nexus.workers.queue.WorkerSettings

scheduler-local:
	python -m nexus.triggers.cron_runner

# ── Docker services ───────────────────────────────────────────────────────────

worker:
	docker compose up nexus-worker --build

scheduler:
	docker compose up nexus-scheduler --build

prod:
	docker compose --profile prod up --build -d

scale-workers:
	docker compose up nexus-worker --scale nexus-worker=4 -d

# ── Database ──────────────────────────────────────────────────────────────────

migrate:
	alembic upgrade head

seed:
	python -m nexus.db.seed

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v --tb=short -m "not slow"

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	ruff check . && mypy nexus/

fmt:
	ruff format nexus/

# ── Logs ─────────────────────────────────────────────────────────────────────

logs-api:
	docker compose logs api -f --tail=100

logs-worker:
	docker compose logs nexus-worker -f --tail=100

logs-scheduler:
	docker compose logs nexus-scheduler -f --tail=100

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	docker compose down -v
