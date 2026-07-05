# Stage 1: Build
FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
COPY src/ src/
COPY config/ config/

RUN uv sync --extra dev --no-dev

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app /app
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

COPY src/ src/
COPY config/ config/

RUN mkdir -p data

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "src.finlens.api.main:app", "--host", "0.0.0.0", "--port", "8080"]