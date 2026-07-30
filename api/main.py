import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RAG Application API", version="0.1.0")

class HealthCheckResponse(BaseModel):
    status: str
    database_url: str | None
    opensearch_url: str | None

@app.get("/", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="ok",
        database_url=os.getenv("DATABASE_URL") and "configured",
        opensearch_url=os.getenv("OPENSEARCH_URL"),
    )

@app.get("/api/ping")
async def ping():
    return {"ping": "pong"}
