# Contributing to NEXUS

Thank you for your interest in contributing to NEXUS!

## Development Setup

```bash
git clone https://github.com/your-org/nexus.git
cd nexus
pip install -e ".[dev]"
docker compose up postgres redis -d
make seed
make test
```

## PR Process

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure `make test` and `make lint` pass
5. Commit with conventional commits (`feat:`, `fix:`, `docs:`)
6. Push and open a PR

## Code Style

- Python: Ruff for linting, mypy for type checking
- All functions need docstrings
- All public APIs need type annotations
- Every database query must be tenant-scoped
