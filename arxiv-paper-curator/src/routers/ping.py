"""Health check and ping router."""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/ping")
def ping() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}
