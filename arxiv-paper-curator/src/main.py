"""FastAPI application entrypoint for arXiv Paper Curator API."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import settings
from .database import init_db
from .routers.agentic_ask import router as agentic_ask_router
from .routers.ask import router as ask_router
from .routers.hybrid_search import router as hybrid_search_router
from .routers.ping import router as ping_router
from .routers.search import router as search_router
from .services.telegram import start_telegram_bot

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown tasks."""
    logger.info("Initializing database tables...")
    init_db()

    bot_app = None
    if settings.telegram_bot_token.strip():
        logger.info("Starting Telegram Bot task...")
        bot_app = await start_telegram_bot()

    yield

    if bot_app:
        logger.info("Stopping Telegram Bot...")
        try:
            if bot_app.updater and bot_app.updater.is_running:
                await bot_app.updater.stop()
            await bot_app.stop()
            await bot_app.shutdown()
        except Exception as exc:
            logger.warning("Error stopping Telegram Bot: %s", exc)

    logger.info("Shutting down application...")


app = FastAPI(
    title="arXiv Paper Curator API",
    description=(
        "Production-grade agentic RAG system for arXiv paper ingestion, indexing, and search."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(ping_router)
app.include_router(search_router, prefix="/api/v1")
app.include_router(hybrid_search_router, prefix="/api/v1")
app.include_router(ask_router, prefix="/api/v1")
app.include_router(agentic_ask_router, prefix="/api/v1")
