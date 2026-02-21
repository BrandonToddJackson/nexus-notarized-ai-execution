.PHONY: dev test lint seed migrate clean install

install:
	pip install -e ".[dev]"

dev:
	docker compose up --build

test:
	pytest tests/ -v

lint:
	ruff check . && mypy nexus/

seed:
	python -m nexus.db.seed

migrate:
	alembic upgrade head

clean:
	docker compose down -v

run:
	uvicorn nexus.api.main:app --reload --port 8000
