"""Database engine, session factory, and declarative Base.

Provides both a sync engine (for Airflow tasks and CLI scripts via psycopg2)
and an async engine (for FastAPI endpoints via asyncpg).
Call `init_db()` once at startup to create all tables via SQLAlchemy metadata.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

logger = logging.getLogger(__name__)


# ── Declarative Base (shared by all ORM models) ───────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Sync engine (psycopg2) ────────────────────────────────────────────────────
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,       # verify connection health before checkout
    pool_size=5,
    max_overflow=10,
    echo=False,               # set True to log all SQL statements
)


@event.listens_for(engine, "connect")
def set_timezone(dbapi_conn, connection_record):  # noqa: ARG001
    """Enforce UTC timezone for all connections."""
    cursor = dbapi_conn.cursor()
    cursor.execute("SET TIME ZONE 'UTC'")
    cursor.close()


# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager that provides a scoped sync session and handles commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── FastAPI dependency (sync) ─────────────────────────────────────────────────
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for sync DB session injection."""
    with get_session() as session:
        yield session


# ── Table creation ────────────────────────────────────────────────────────────
def init_db() -> None:
    """Create all tables defined in Base.metadata if they do not exist.

    Call once at application startup (FastAPI lifespan or Airflow init task).
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS semantics.
    """
    # Import all models so their metadata is registered before create_all()
    from .models import paper  # noqa: F401

    logger.info("Running init_db() — creating tables if not exist …")
    Base.metadata.create_all(bind=engine)
    logger.info("init_db() complete.")
