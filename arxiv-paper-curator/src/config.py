"""Application configuration using pydantic-settings.

Reads from .env file. Includes settings for database (sync + async),
arXiv ingestion, LLM provider, Redis, and Langfuse observability.
No Ollama-specific settings — all LLM access is provider-agnostic via LLM_PROVIDER.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────
    # Sync URL for Airflow tasks, scripts, CLI runs (psycopg2)
    database_url: str = Field(
        default="postgresql://rag_user:rag_password@localhost:5433/rag_db",
        description="Sync PostgreSQL URL (psycopg2 driver)",
    )
    # Async URL for FastAPI endpoints (asyncpg)
    async_database_url: str = Field(
        default="postgresql+asyncpg://rag_user:rag_password@localhost:5433/rag_db",
        description="Async PostgreSQL URL (asyncpg driver)",
    )

    # ── OpenSearch ────────────────────────────────────────────
    opensearch_url: str = Field(default="http://localhost:9200")

    # ── LLM — provider-agnostic ───────────────────────────────
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_api_key: str = Field(default="")

    # ── Redis ─────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379")

    # ── Langfuse ──────────────────────────────────────────────
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")

    # ── arXiv ingestion ───────────────────────────────────────
    arxiv_max_results: int = Field(
        default=20,
        description="Max papers to fetch per DAG run",
    )
    arxiv_rate_limit_calls: int = Field(
        default=3,
        description="Max API calls allowed per rate-limit period",
    )
    arxiv_rate_limit_period: float = Field(
        default=1.0,
        description="Rate-limit window in seconds",
    )
    arxiv_default_query: str = Field(
        default="cs.AI",
        description="Default arXiv category/query for daily ingestion",
    )

    # ── Embeddings & Chunking ──────────────────────────────────
    embeddings_provider: str = Field(default="nvidia")
    embeddings_api_key: str = Field(default="")
    embeddings_model: str = Field(default="nvidia/nv-embedqa-e5-v5")
    embeddings_api_url: str = Field(default="https://integrate.api.nvidia.com/v1/embeddings")
    embeddings_dimensions: int = Field(default=1024)
    embeddings_batch_size: int = Field(default=8)

    chunk_max_tokens: int = Field(default=350)
    chunk_overlap_tokens: int = Field(default=50)


# Module-level singleton — import this everywhere
settings = Settings()
