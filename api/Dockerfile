FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Create a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy project configuration files first for better caching
COPY pyproject.toml ./

# Generate requirements and install dependencies
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Multi-stage build for a clean runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code from the builder
COPY --from=builder /app /app

# Create non-root user for security
RUN addgroup --gid 1001 appgroup && \
    adduser --uid 1001 --gid 1001 --disabled-password --gecos "" appuser && \
    chown -R appuser:appgroup /app /opt/venv

USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
