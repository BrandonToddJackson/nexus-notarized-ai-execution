FROM python:3.11-slim AS builder
WORKDIR /app

# Copy dependency manifest + source so pip install . can find the package
COPY pyproject.toml README.md ./
COPY nexus/ ./nexus/
RUN pip install --no-cache-dir .

# Copy remaining files (alembic, examples, etc.) after deps are cached
COPY . .

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
USER nobody
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1
CMD ["uvicorn", "nexus.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
