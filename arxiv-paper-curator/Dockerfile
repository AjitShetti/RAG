FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY pyproject.toml ./
COPY airflow/requirements.txt ./airflow/

RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt && \
    uv pip install -r airflow/requirements.txt

COPY . .

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /app /app

RUN addgroup --gid 1001 appgroup && \
    adduser --uid 1001 --gid 1001 --disabled-password --gecos "" appuser && \
    chown -R appuser:appgroup /app /opt/venv

USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]